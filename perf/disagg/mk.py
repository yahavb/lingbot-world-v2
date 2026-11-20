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
    """n Neuron devices at LNC2, each with its own paired EFA (devicegroup1_id), all in one device group."""
    reqs, cons = [], []
    for i in range(n):
        reqs.append({"name": f"neurons-{i}", "exactly": {
            "allocationMode": "ExactCount", "count": 1, "deviceClassName": "neuron.aws.com",
            "selectors": [{"cel": {"expression":
                "device.attributes['neuron.aws.com'].instanceType == 'trn3-dev1.48xlarge'"}}]}})
        reqs.append({"name": f"efas-{i}", "exactly": {
            "allocationMode": "ExactCount", "count": 1, "deviceClassName": "efa.networking.k8s.aws"}})
        cons.append({"matchAttribute": "resource.aws.com/devicegroup1_id", "requests": [f"neurons-{i}", f"efas-{i}"]})
    if n in (2, 4, 8):
        cons.append({"matchAttribute": f"resource.aws.com/devicegroup{n}_id",
                     "requests": [r["name"] for r in reqs]})
    return {"apiVersion": "resource.k8s.io/v1", "kind": "ResourceClaimTemplate",
            "metadata": {"name": f"x-lnc2-efa-{n}", "namespace": "default"},
            "spec": {"spec": {"devices": {
                "requests": reqs, "constraints": cons,
                "config": [{"requests": [f"neurons-{i}" for i in range(n)], "opaque": {
                    "driver": "neuron.aws.com",
                    "parameters": {"apiVersion": "neuron.aws.com/v1", "kind": "NeuronConfig",
                                   "logicalNeuronCore": 2}}}]}}}}


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
    nproc = RANKS_PER_CHIP * a.chips
    env_set(c, "NPROC", nproc)
    env_set(c, "TP", TP)
    env_set(c, "CP", nproc // TP)
    env_set(c, "SIZE", a.size)
    env_set(c, "CACHE_VERSION", a.cache)
    if a.mode == "split":
        env_set(c, "VAE_URL", f"http://{a.vae_app}.default.svc.cluster.local:8000")
        env_set(c, "VAE_SELECTOR", f"app={a.vae_app}")
    else:
        for k in ("VAE_URL", "VAE_SELECTOR", "LINGBOT_VAE_RDMA"):
            env_del(c, k)
    if a.mode == "null":
        env_set(c, "LINGBOT_NULL_DECODE", "1")
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
    p.add_argument("--size", default="480*768"); p.add_argument("--cache", default="5")
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
