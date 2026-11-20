"""Generate the pod specs for one cell of the disaggregation matrix (docs/disagg/PROBLEM.md).

The base specs are the live split's (lbw-dit-deploy.yaml, lbw-vae-deploy.yaml, as JSON in base/), so the
monolith, the split and the DiT-only runs share every model flag; only the parallelism, the VAE mode and
the claim change.

    python3 perf/disagg/mk.py claim 4                   # ResourceClaimTemplate x-lnc2-efa-4 (4 device+EFA pairs)
    python3 perf/disagg/mk.py dit  NAME CHIPS MODE [--size 480*768] [--vae-app APP]
        MODE: split (VAE pods labelled app=APP), inline (monolith: DiT + VAE on the same chips), null (DiT only)
    python3 perf/disagg/mk.py vae  NAME CHIPS APP [--replicas 1] [--size 480*768] [--per-device 2]

Specs are written to perf/disagg/gen/NAME.json (kubectl apply -f). TP stays 4; CP = 4 ranks per chip x
CHIPS / TP (LNC2: 4 ranks per chip).
"""
import argparse
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "gen")
RANKS_PER_CHIP = 4  # LNC2 on trn3: 8 physical cores per chip, 4 logical ranks
TP = 4
LAT = {"480*768": (60, 96), "720*1280": (90, 160), "704*1280": (88, 160)}


def env_set(c, name, value):
    for e in c["env"]:
        if e["name"] == name:
            e.pop("valueFrom", None)
            e["value"] = str(value)
            return
    c["env"].append({"name": name, "value": str(value)})


def env_del(c, name):
    c["env"] = [e for e in c["env"] if e["name"] != name]


def claim(n):
    """n Neuron devices at LNC2 and the n EFAs of the same device group (devicegroup<n>_id: a group holds its
    devices' paired EFAs, so every device's own EFA is in the claim; 16 = the whole node). Count requests:
    the per-device pairing constraints (one request per device) time out the scheduler at 16."""
    if n <= 2:
        return claim_paired(n)
    group = f"devicegroup{n}_id"      # trn3-dev1 has devicegroup 1, 4, 8, 16
    sel = [{"cel": {"expression": "device.attributes['neuron.aws.com'].instanceType == 'trn3-dev1.48xlarge'"}}]
    reqs = [{"name": "neurons", "exactly": {"allocationMode": "ExactCount", "count": n,
                                            "deviceClassName": "neuron.aws.com", "selectors": sel}},
            {"name": "efas", "exactly": {"allocationMode": "ExactCount", "count": n,
                                         "deviceClassName": "efa.networking.k8s.aws"}}]
    cons = [{"matchAttribute": f"resource.aws.com/{group}", "requests": ["neurons", "efas"]}] if group else []
    return {"apiVersion": "resource.k8s.io/v1", "kind": "ResourceClaimTemplate",
            "metadata": {"name": f"x-lnc2-efa-{n}", "namespace": "default"},
            "spec": {"spec": {"devices": {
                "requests": reqs, "constraints": cons,
                "config": [{"requests": ["neurons"], "opaque": {
                    "driver": "neuron.aws.com",
                    "parameters": {"apiVersion": "neuron.aws.com/v1", "kind": "NeuronConfig",
                                   "logicalNeuronCore": 2}}}]}}}}


def claim_paired(n):
    """n (1 or 2) devices, each with its own EFA (devicegroup1_id), all in one devicegroup4."""
    reqs, cons = [], []
    sel = [{"cel": {"expression": "device.attributes['neuron.aws.com'].instanceType == 'trn3-dev1.48xlarge'"}}]
    for i in range(n):
        reqs += [{"name": f"neurons-{i}", "exactly": {"allocationMode": "ExactCount", "count": 1,
                                                      "deviceClassName": "neuron.aws.com", "selectors": sel}},
                 {"name": f"efas-{i}", "exactly": {"allocationMode": "ExactCount", "count": 1,
                                                   "deviceClassName": "efa.networking.k8s.aws"}}]
        cons.append({"matchAttribute": "resource.aws.com/devicegroup1_id", "requests": [f"neurons-{i}", f"efas-{i}"]})
    if n > 1:
        cons.append({"matchAttribute": "resource.aws.com/devicegroup4_id", "requests": [r["name"] for r in reqs]})
        # a directly linked pair: devices 5 + 6 (same group of 4) failed "ALLREDUCE over 8 ranks not supported";
        # linked pairs share topology_x
        cons.append({"matchAttribute": "neuron.aws.com/topology_x", "requests": [f"neurons-{i}" for i in range(n)]})
    return {"apiVersion": "resource.k8s.io/v1", "kind": "ResourceClaimTemplate",
            "metadata": {"name": f"x-lnc2-efa-{n}", "namespace": "default"},
            "spec": {"spec": {"devices": {"requests": reqs, "constraints": cons, "config": [{
                "requests": [f"neurons-{i}" for i in range(n)], "opaque": {"driver": "neuron.aws.com",
                "parameters": {"apiVersion": "neuron.aws.com/v1", "kind": "NeuronConfig", "logicalNeuronCore": 2}}}]}}}}


def rename(d, name):
    d["metadata"] = {"name": name, "namespace": "default", "labels": {"exp": "disagg"}}
    d["spec"]["selector"]["matchLabels"] = {"app": name}
    d["spec"]["template"]["metadata"]["labels"] = {"app": name, "exp": "disagg"}
    d["spec"]["template"]["spec"]["resourceClaims"][0]["resourceClaimTemplateName"] = None


def dit(a):
    d = copy.deepcopy(json.load(open(os.path.join(HERE, "base", "lbw-dit-deploy.json"))))
    rename(d, a.name)
    d["spec"]["template"]["spec"]["resourceClaims"][0]["resourceClaimTemplateName"] = f"x-lnc2-efa-{a.chips}"
    c = d["spec"]["template"]["spec"]["containers"][0]
    if a.flags == "mono":
        # one flag set for every chip count: the 4-chip monolith's DiT flags (lbw-deploy.yaml; the 2-chip t22
        # kernels are tuned to CP2 shapes and fail at CP4+), the split VAE's VAE flags (lbw-vae-deploy.yaml) so
        # inline and split decode with the same kernels, and the split's transport env
        mono = json.load(open(os.path.join(HERE, "base", "lbw-deploy.json")))["spec"]["template"]["spec"]["containers"][0]["env"]
        vae_env = json.load(open(os.path.join(HERE, "base", "lbw-vae-deploy.json")))["spec"]["template"]["spec"]["containers"][0]["env"]
        keep = [e for e in c["env"] if e["name"].startswith(("FI_", "CCOM_")) or e["name"] in
                ("VAE_URL", "VAE_SELECTOR", "LINGBOT_VAE_RDMA", "HF_TOKEN")]
        env = {e["name"]: e for e in mono}
        env.update({e["name"]: e for e in vae_env if e["name"].startswith("LINGBOT_VAE_")})
        env.update({e["name"]: e for e in keep})
        c["env"] = list(env.values())
        env_set(c, "LINGBOT_REANCHOR_FRAMES", "0")
    nproc = RANKS_PER_CHIP * a.chips
    env_set(c, "NPROC", nproc)
    env_set(c, "TP", TP)
    env_set(c, "CP", nproc // TP)
    env_set(c, "SIZE", a.size)
    if a.size != "480*768":
        env_set(c, "LINGBOT_SCENES", "03,00")          # pre-encode only the measured scene + the warmup's second
        env_set(c, "LINGBOT_PRE_ENCODE_RANK0", "1")
    env_set(c, "CACHE_VERSION", a.cache or f"{a.flags}{a.chips}")   # one cache per flag set x chip count
    # ranks loading the checkpoint at once: all 64 at 16 chips exceeded 950 GiB of host memory (OOMKilled)
    env_set(c, "LINGBOT_LOAD_CONCURRENCY", nproc if a.flags == "t22" else min(nproc, 16))
    if a.chips >= 16:
        c["resources"]["limits"]["memory"] = "1600Gi"
    if a.mode == "split":
        env_set(c, "VAE_URL", f"http://{a.vae_app}.default.svc.cluster.local:8000")
        # the DiT uses the VAE pods labelled pool=<its name> (kubectl label pod ... pool=NAME): VAEs move between
        # DiTs by relabelling, with no restart (wan/vae_pool.py lists pods by this selector every 5 s)
        env_set(c, "VAE_SELECTOR", f"pool={a.name}")
    else:
        for k in ("VAE_URL", "VAE_SELECTOR", "LINGBOT_VAE_RDMA"):
            env_del(c, k)
    if a.mode == "null":
        env_set(c, "LINGBOT_NULL_DECODE", "1")
    if a.sysprof:
        # Neuron runtime system profile of global rank 0 (perf/sysprof_rank0.sh): one span per NEFF execution
        cmd = c["command"][2]
        old = 'torchrun --nproc_per_node="$NPROC" serve.py'
        assert old in cmd
        c["command"][2] = cmd.replace(old, 'torchrun --nproc_per_node="$NPROC" --no-python bash perf/sysprof_rank0.sh python3 serve.py')
        env_set(c, "SYSPROF_DIR", "/logs/sysprof")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env_set(c, k, v)
    return d


def vae(a):
    d = copy.deepcopy(json.load(open(os.path.join(HERE, "base", "lbw-vae-deploy.json"))))
    rename(d, a.name)
    d["metadata"]["labels"]["app"] = a.app
    d["spec"]["selector"]["matchLabels"] = {"app": a.app, "vae": a.name}
    d["spec"]["template"]["metadata"]["labels"] = {"app": a.app, "vae": a.name, "exp": "disagg"}
    d["spec"]["template"]["spec"]["resourceClaims"][0]["resourceClaimTemplateName"] = f"x-lnc2-efa-{a.chips}"
    d["spec"]["replicas"] = a.replicas
    c = d["spec"]["template"]["spec"]["containers"][0]
    lh, lw = LAT[a.size]
    env_set(c, "NPROC", RANKS_PER_CHIP * a.chips)
    env_set(c, "LAT_H", lh)
    env_set(c, "LAT_W", lw)
    env_set(c, "LINGBOT_VAE_PER_DEVICE", a.per_device)
    env_set(c, "CACHE_VERSION", a.cache)
    return d


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("claim"); p.add_argument("chips", type=int)
    p = sub.add_parser("dit"); p.add_argument("name"); p.add_argument("chips", type=int)
    p.add_argument("mode", choices=["split", "inline", "null"]); p.add_argument("--vae-app", default="")
    p.add_argument("--size", default="480*768"); p.add_argument("--cache", default="")
    p.add_argument("--flags", choices=["mono", "t22"], default="mono")
    p.add_argument("--env", action="append", default=[], help="K=V, repeatable")
    p.add_argument("--sysprof", action="store_true", help="runtime system profile of rank 0")
    p = sub.add_parser("vae"); p.add_argument("name"); p.add_argument("chips", type=int); p.add_argument("app")
    p.add_argument("--replicas", type=int, default=1); p.add_argument("--size", default="480*768")
    p.add_argument("--per-device", default="2"); p.add_argument("--cache", default="5")
    a = ap.parse_args()
    os.makedirs(GEN, exist_ok=True)
    if a.cmd == "claim":
        d, name = claim(a.chips), f"x-lnc2-efa-{a.chips}"
    elif a.cmd == "dit":
        d, name = dit(a), a.name
    else:
        d, name = vae(a), a.name
    if a.cmd != "claim":
        # a Service per app, so the DiT reaches the VAE pods (and the runner reaches the DiT) by name
        app = a.app if a.cmd == "vae" else a.name
        d = {"apiVersion": "v1", "kind": "List", "items": [d, {
            "apiVersion": "v1", "kind": "Service", "metadata": {"name": app, "namespace": "default"},
            "spec": {"selector": {"app": app}, "ports": [{"port": 8000, "targetPort": 8000}]}}]}
    out = os.path.join(GEN, f"{name}.json")
    json.dump(d, open(out, "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
