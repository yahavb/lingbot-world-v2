"""lbw-dit rank 0: find a VAE at session time, link to it over RDMA, and fail over to another.

Discovery follows production-stack's K8sPodIPServiceDiscovery (src/vllm_router/service_discovery.py):
list (every POLL_S) the pods that match a label selector and keep the ones whose containers are Ready and that
are not terminating, by pod IP. A session takes one of them (round-robin); if its HTTP request or
RDMA send fails, that VAE is set aside for BAD_S seconds and the next chunk goes to another one.
With none available a session is refused with 503 VAE_SERVICE_UNAVAILABLE.

Env: VAE_SELECTOR (default app=lbw-vae), VAE_PORT (8000), plus wan/rdma_link.py's.
"""
import itertools
import logging
import os
import random
import threading
import time

import requests

BAD_S = 60
POLL_S = float(os.environ.get("VAE_POLL_S", "5"))
log = logging.getLogger("vae_pool")


class VaeUnavailable(RuntimeError):
    pass


class VaePool:
    def __init__(self):
        from wan import rdma_link as R
        self.R = R
        self.selector = os.environ.get("VAE_SELECTOR", "app=lbw-vae")
        self.port = int(os.environ.get("VAE_PORT", "8000"))
        self.endpoints = {}            # pod name -> pod IP, Ready and not terminating
        self.bad = {}                  # pod name -> time it failed
        self.lock = threading.Lock()
        self.rank = R.new_transport_rank(random)   # this process's transport rank, for its life
        self.gen = itertools.count(random.randrange(1, 1 << 30))
        self.store = R.shared_store()  # the cluster's one rendezvous store, for this process's life
        self.current = None            # (name, ip, link)
        self.fresh = False             # the current VAE has not decoded a chunk yet
        self.http = requests.Session()
        threading.Thread(target=self._watch, daemon=True).start()

    # ---- discovery (production-stack: K8sPodIPServiceDiscovery._watch_engines) ----
    def _watch(self):
        """Every POLL_S, list the pods matching the selector and keep the Ready, not-terminating ones
        (production-stack's readiness rules). A fresh list each time: a long-lived watch stream in this
        process stalled after its first event, and a list can never miss a change."""
        from kubernetes import client, config
        config.load_incluster_config()
        ns = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()
        api = client.CoreV1Api()
        while True:
            try:
                pods = api.list_namespaced_pod(namespace=ns, label_selector=self.selector, _request_timeout=10).items
                now = {}
                for pod in pods:
                    cs = pod.status.container_statuses or []
                    if pod.metadata.deletion_timestamp is None and pod.status.pod_ip and cs and all(c.ready for c in cs):
                        now[pod.metadata.name] = pod.status.pod_ip
                with self.lock:
                    for n in set(self.endpoints) - set(now):
                        log.info(f"vae gone: {n}")
                    for n, ip in now.items():
                        if self.endpoints.get(n) != ip:
                            log.info(f"vae ready: {n} {ip}")
                    self.endpoints = now
            except Exception as e:
                log.warning(f"vae list: {type(e).__name__}: {e}")
            time.sleep(POLL_S)

    def available(self):
        now = time.time()
        with self.lock:
            return [(n, ip) for n, ip in sorted(self.endpoints.items())
                    if now - self.bad.get(n, 0) > BAD_S]

    # ---- linking ----
    def url(self):
        return None if self.current is None else f"http://{self.current[1]}:{self.port}"

    def acquire(self):
        """Keep the current VAE if it is still listed; else link to the next available one."""
        if self.current is not None and self.current[0] in dict(self.available()):
            return True
        self.release()
        cands = self.available()
        random.shuffle(cands)
        for name, ip in cands:
            try:
                gen = next(self.gen) % (1 << 30)
                r = self.http.post(f"http://{ip}:{self.port}/link", params={"gen": gen, "dit_rank": self.rank},
                                   timeout=10)
                r.raise_for_status()
                vae_rank = int(r.json()["vae_rank"])
                if vae_rank == self.rank:
                    raise RuntimeError("transport rank clash")
                link = self.R.RdmaLink("dit", self.store, gen, self.rank, vae_rank)
                self.current, self.fresh = (name, ip, link), True
                log.info(f"session VAE: {name} {ip} (gen {gen}, transport ranks {self.rank}->{vae_rank})")
                return True
            except Exception as e:
                log.warning(f"vae {name} {ip}: link failed: {type(e).__name__}: {e}")
                self.fail(name)
        return False

    def fail(self, name=None):
        name = name or (self.current[0] if self.current else None)
        if name:
            with self.lock:
                self.bad[name] = time.time()
            log.warning(f"vae {name}: set aside for {BAD_S} s")
        if self.current is not None and (name is None or self.current[0] == name):
            self.release()

    def release(self):
        if self.current is not None:
            self.current[2].close()
            self.current = None

    def send(self, remote, first, x_dev, emit):
        """Decode one chunk on the session's VAE: HTTP request (metadata, frames back) + RDMA send of
        the latents. A failure moves to another VAE (which starts its decode cache fresh) and
        retries the chunk once."""
        for attempt in (0, 1):
            if remote.error is not None:                     # the last HTTP request failed
                remote.error = None
                self.fail()
            if not self.acquire():
                raise VaeUnavailable("no VAE available")
            try:
                remote.submit(first or self.fresh, None, emit, url=self.url())
                self.current[2].send(x_dev).wait()
                # Neuron reports a failed transfer asynchronously, at the next sync point; sync here so a
                # dead VAE fails this try (and fails over) instead of surfacing later inside the DiT
                import torch
                torch.neuron.synchronize()
                self.fresh = False
                return
            except VaeUnavailable:
                raise
            except Exception as e:
                log.warning(f"vae send failed: {type(e).__name__}: {e}")
                self.fail()
        raise VaeUnavailable("VAE failover did not succeed")
