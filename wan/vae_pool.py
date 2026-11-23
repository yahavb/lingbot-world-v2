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
_NIXL_TIMING = __import__('os').environ.get('LINGBOT_NIXL_TIMING', '0') == '1'

import requests

BAD_S = 60
DRAIN_S = float(os.environ.get("VAE_DRAIN_S", "6"))
POLL_S = float(os.environ.get("VAE_POLL_S", "5"))
log = logging.getLogger("vae_pool")


class VaeUnavailable(RuntimeError):
    pass


class VaePool:
    def __init__(self):
        self._wt = []   # LINGBOT_NIXL_TIMING samples
        from wan import rdma_link as R
        self.R = R
        self.selector = os.environ.get("VAE_SELECTOR", "app=lbw-vae")
        self.port = int(os.environ.get("VAE_PORT", "8000"))
        self.endpoints = {}            # pod name -> pod IP, Ready and not terminating
        self.bad = {}                  # pod name -> time it failed
        self.lock = threading.Lock()
        self.gen = itertools.count(random.randrange(1, 1 << 30))
        # transport: "nixl" (default; wan/nixl_link.py, one-sided writes) or "neuron" (wan/rdma_link.py)
        self.transport = os.environ.get("LINGBOT_VAE_TRANSPORT", "nixl")
        self.nixl = None               # wan.nixl_link.NixlDit, made at the first chunk (its shape)
        self.seq = itertools.count()
        if self.transport != "nixl":
            self.store = R.shared_store()  # the cluster's one rendezvous store, for this process's life
            self.rank = R.new_transport_rank(self.store)  # this process's transport rank, never reused
        else:
            self.rank = -1
        self.current = None            # (name, ip, link)
        self.fresh = False             # the current VAE has not decoded a chunk yet
        self.gone = False              # the current VAE disappeared from the pod list
        self._prefer = None            # relink to this VAE first (after a stale-completion failure)
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
                        if self.current is not None and self.current[0] == n:
                            self.gone = True       # the session's VAE left: the next chunk relinks
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

    def acquire(self, prefer=None):
        """Keep the current VAE if it is still listed; else link to an available one (`prefer` first)."""
        # LINGBOT_VAE_LEAST_LOADED=1: a NEW session (new_session()) re-chooses by load instead of keeping the last VAE
        if (self.current is not None and not self.gone and self.current[0] in dict(self.available())
                and not getattr(self, "_rechoose", False)):
            return True
        self._rechoose = False
        self.gone = False
        self.release()
        cands = self.available()
        random.shuffle(cands)
        if os.environ.get("LINGBOT_VAE_LEAST_LOADED", "0") == "1":
            # horizontal scaling of the decoder role: link to the VAE serving the fewest sessions (GET /load)
            def load(c):
                try:
                    return int(self.http.get(f"http://{c[1]}:{self.port}/load", timeout=1).json()["sessions"])
                except Exception:
                    return 1 << 20
            loads = {c[0]: load(c) for c in cands}
            cands.sort(key=lambda c: loads[c[0]])
            log.info(f"VAE loads: {loads}")
        cands.sort(key=lambda c: c[0] != prefer)
        for name, ip in cands:
            try:
                gen = next(self.gen) % (1 << 30)
                r = self.http.post(f"http://{ip}:{self.port}/link", params={"gen": gen, "dit_rank": self.rank},
                                   timeout=10)
                r.raise_for_status()
                info = r.json()
                if self.transport == "nixl":
                    if info.get("transport") != "nixl":
                        raise RuntimeError(f"VAE {name} does not speak NIXL")
                    conn = self.nixl.connect(info)
                    self.current, self.fresh = (name, ip, conn), True
                    log.info(f"session VAE: {name} {ip} (nixl, {len(info['slots'])} slots)")
                    return True
                vae_rank = int(info["vae_rank"])
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

    def new_session(self):
        """Called at a session's start: with LINGBOT_VAE_LEAST_LOADED=1 the next acquire() picks the least-loaded VAE."""
        if os.environ.get("LINGBOT_VAE_LEAST_LOADED", "0") == "1" and self.current is not None:
            self.release()
            self._rechoose = True

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
            if self.transport == "nixl":
                self.nixl.disconnect(self.current[2])
            else:
                self.current[2].close()
            self.current = None

    def send(self, remote, first, x_dev, emit):
        """Decode one chunk on the session's VAE: HTTP request (metadata, frames back) + RDMA send of
        the latents. On a failure: if the VAE is still listed, the failure may be a stale completion from
        a transfer to a VAE that just died (its error lands on the next transfer of the process): retire the
        link, wait DRAIN_S for that to clear, and relink to the same VAE; otherwise (or if that fails too)
        set the VAE aside and move to another one, which starts its decode cache fresh."""
        import torch
        if self.transport == "nixl":
            return self._send_nixl(remote, first, x_dev, emit)
        relinked = False
        for attempt in range(3):
            if remote.error is not None:                     # the last HTTP request failed
                remote.error = None
                self.fail()
            if not self.acquire(prefer=self._prefer):
                raise VaeUnavailable("no VAE available")
            self._prefer = None
            name = self.current[0]
            try:
                remote.submit(first or self.fresh, None, emit, url=self.url())
                self.current[2].send(x_dev).wait()
                # Neuron reports a failed transfer asynchronously, at the next sync point: sync here so the
                # failure is caught by this try (and handled) instead of surfacing later inside the DiT
                torch.neuron.synchronize()
                self.fresh = False
                return
            except VaeUnavailable:
                raise
            except Exception as e:
                log.warning(f"vae {name}: send failed: {type(e).__name__}: {e}")
                if not relinked and name in dict(self.available()):
                    relinked = True
                    self.release()
                    log.warning(f"vae {name}: relinking after {DRAIN_S} s (a stale completion from a dead peer)")
                    time.sleep(DRAIN_S)
                    self._prefer = name
                else:
                    self.fail(name)
        raise VaeUnavailable("VAE failover did not succeed")

    def _send_nixl(self, remote, first, x_dev, emit):
        """NIXL: a write that fails means that VAE is gone (no stale errors carry over, perf/nixl_ping.py):
        set it aside and retry the chunk on the next one."""
        if self.nixl is None:
            from wan.nixl_link import NixlDit
            self.nixl = NixlDit(tuple(x_dev.shape), x_dev.device)
        for attempt in range(3):
            if remote.error is not None:                     # the last HTTP request failed
                remote.error = None
                self.fail()
            if not self.acquire():
                raise VaeUnavailable("no VAE available")
            # (kiro) backpressure: at most SLOTS chunks in flight (submitted, not decoded yet). Chunk seq goes to
            # slot seq % SLOTS, which last held chunk seq - SLOTS; without this wait a DiT ahead of its decoder
            # overwrote latents the decoder had not read yet, and those chunks decoded the wrong latents
            from wan.nixl_link import SLOTS as _SLOTS
            while remote.q.unfinished_tasks >= _SLOTS and remote.error is None:
                time.sleep(0.001)
            name, seq = self.current[0], next(self.seq)
            try:
                remote.submit(first or self.fresh, None, emit, url=self.url(), seq=seq, sid=self.current[2][2])
                for retry in range(3):
                    try:
                        _t0 = time.perf_counter()
                        self.nixl.write(self.current[2], x_dev, seq)
                        if _NIXL_TIMING:
                            # LINGBOT_NIXL_TIMING=1: the hand-off's own time (device copy into the send buffer, sync,
                            # one-sided write, completion), median of every 20 chunks
                            self._wt.append(((time.perf_counter() - _t0) * 1000, getattr(self.nixl, "last_xfer_ms", -1.0)))
                            if len(self._wt) == 20:
                                w = sorted(x[0] for x in self._wt)
                                x = sorted(x[1] for x in self._wt)
                                log.info(f"[NIXL] over 20 chunks: transfer median {x[10]:.3f} ms (min {x[0]:.3f} max "
                                         f"{x[-1]:.3f}); write() incl. device sync median {w[10]:.1f} ms; "
                                         f"bytes {x_dev.numel() * x_dev.element_size()}")
                                self._wt = []
                        break
                    except Exception as e:
                        # a cancelled completion left by the dead VAE can fail this write; the new VAE is still
                        # waiting for chunk `seq`, so writing it again is safe
                        if retry == 2:
                            raise
                        log.warning(f"vae {name}: nixl write failed ({type(e).__name__}: {e}); retrying")
                        time.sleep(0.2)
                self.fresh = False
                return
            except Exception as e:
                log.warning(f"vae {name}: nixl write failed: {type(e).__name__}: {e}")
                self.fail(name)
        raise VaeUnavailable("VAE failover did not succeed")
