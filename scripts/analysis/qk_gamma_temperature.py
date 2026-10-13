"""Which layers are too HOT for const-max attention, from the QK-RMSNorm gammas alone.

Why this exists: we run const_max_ag on all 40 blocks with a single assumed
softmax_max=15.0. Per the const-max design note, that is only valid for COLD heads.
Hot heads select a specific token, their row-max can exceed 500, and a const max far
below it makes exp(score - max) overflow -- so those layers must fall back to flash
attention. the reference implementation ports that as FLASH_ATTN_LAYERS (their PR #90, which we have NOT
ported); their measured hot set is layers 33, 34, 38, 39 for the reference model.

That set is THEIRS. lingbot is a different model with different weights, so the hot
layers must be found on lingbot's own gammas -- copying 33,34,38,39 across would be
the same mistake as inheriting softmax_max=15.0.

The note gives an offline predictor that needs no device and no forward pass:

  "Per-head gamma max and runtime row-max are strongly correlated for hot heads ...
   the converse also holds: no head with low gamma is hot. This suggests that gamma
   can be a reliable predictor of cold heads, even in the across-heads RMSNorm
   scenario."

  "Across all 9 hot heads, the non-amplified dimensions average ~1.1 (std ~0.15),
   while 5-18 dimensions spike to 2.3-4.3. Crucially, gamma_q and gamma_k spike in
   the SAME dimensions -- this aligns the tokens in those directions which results in
   higher cos(theta) and scores. Cold heads have uniform gamma (~1.2-1.5 everywhere,
   std ~0.05)."

So the discriminator is CO-SPIKING: dimensions where gamma_q AND gamma_k are both
amplified, within the same head. A head with large gammas that do not co-spike stays
cold, which is why max|gamma| alone is reported but not used to decide.

lingbot normalises ACROSS HEADS (WanRMSNorm(dim=5120), model.py:161), which the note
calls scenario 2: the Cauchy-Schwarz bound sqrt(d)*max|gq|*max|gk| is computed here
for completeness but is far too loose to classify with -- at d=5120 it exceeds the
~70 safe ceiling for almost any gamma. It is printed so that looseness is visible
rather than assumed.

This reports; it does not edit anything. Run it where the weights are (the pod has
them at /tmp/ckpt); it is CPU-only and needs no NeuronCore.

    python3 scripts/analysis/qk_gamma_temperature.py --ckpt_dir /tmp/ckpt

WHAT THIS IS NOT: a measurement of the actual row-max. It predicts cold heads from
weights. The note is explicit that offline gamma analysis is "not an absolute
guarantee", and its own hot/cold split for the reference model came from runtime row-max over 3
prompts. Treat the output as the candidate hot set to verify, not as proof.
"""
import argparse
import glob
import os
import sys

import torch


def _load_qk_gammas(ckpt_dir):
    """{block_index: (gamma_q, gamma_k)} for every block that has QK-RMSNorm."""
    state = None
    pth = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.pth"), recursive=True))
    for p in pth:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if isinstance(obj, dict) and any("norm_q" in k for k in obj):
            state, src = obj, p
            break
    if state is None:
        from safetensors.torch import load_file
        shards = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.safetensors"),
                                  recursive=True))
        if not shards:
            raise SystemExit(f"no .pth or .safetensors with norm_q under {ckpt_dir}")
        state, src = {}, f"{len(shards)} safetensors shards"
        for s in shards:
            for k, v in load_file(s).items():
                if "norm_q" in k or "norm_k" in k:
                    state[k] = v
    print(f"gammas from: {src}")

    out = {}
    for k in state:
        if not k.endswith("norm_q.weight") or "cross" in k:
            continue
        kk = k.replace("norm_q", "norm_k")
        if kk not in state:
            continue
        idx = [int(t) for t in k.split(".") if t.isdigit()]
        if not idx:
            continue
        out[idx[0]] = (state[k].float(), state[kk].float())
    if not out:
        raise SystemExit("found no self-attention norm_q/norm_k pairs")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Predict const-max-unsafe (HOT) layers from QK-RMSNorm gammas")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--spike_sigma", type=float, default=3.0,
                    help="a dim is amplified when gamma > mean + sigma*std, per head")
    ap.add_argument("--min_cospikes", type=int, default=5,
                    help="co-spike count reported per layer. NOT the classifier any "
                         "more -- on lingbot the counts span 1-6 with no gap, so a "
                         "threshold on them selects noise. Kept for the table only.")
    ap.add_argument("--max_hot", type=int, default=6,
                    help="never flag more than this many layers; each one costs HBM "
                         "and exact attention is the slow path")
    ap.add_argument("--min_gap", type=float, default=1.4,
                    help="require this relative drop in CS_bound to cut hot from cold; "
                         "below it, report no hot layers rather than guess")
    args = ap.parse_args()

    gammas = _load_qk_gammas(args.ckpt_dir)
    n_heads = args.heads
    print(f"blocks with QK-RMSNorm: {len(gammas)}   heads/block: {n_heads}")

    hot_layers, rows = [], []
    for blk in sorted(gammas):
        gq, gk = gammas[blk]
        d_full = gq.numel()
        hd = d_full // n_heads
        q = gq.view(n_heads, hd).abs()
        k = gk.view(n_heads, hd).abs()

        qm, qs = q.mean(1, keepdim=True), q.std(1, keepdim=True)
        km, ks = k.mean(1, keepdim=True), k.std(1, keepdim=True)
        spike_q = q > (qm + args.spike_sigma * qs)
        spike_k = k > (km + args.spike_sigma * ks)
        cospikes = (spike_q & spike_k).sum(1)

        worst = int(cospikes.argmax())
        cs_bound = (d_full ** 0.5) * float(q.max()) * float(k.max())
        n_hot_heads = int((cospikes >= args.min_cospikes).sum())
        # LAYER-WIDE maxima, the same ones CS_bound is computed from. This column used
        # to print q[worst].max() -- the max inside the worst-co-spike HEAD -- while
        # CS_bound used q.max() over the whole layer. The two differ by ~3x (layer 38
        # printed 1.45 while its layer max is 4.30), which made the table look like no
        # layer reached the note's 2.3-4.3 hot range when four of them do.
        rows.append((blk, n_hot_heads, int(cospikes[worst]), float(q.max()),
                     float(k.max()), float(q.mean()), float(q.std()), cs_bound))

    # RANK BY THE CAUCHY-SCHWARZ BOUND, then cut at the largest relative gap.
    #
    # The first version of this classified on co-spike COUNT with a fixed threshold, and
    # on lingbot that thresholded noise: the counts spanned only 1-6 with no separation
    # anywhere, so ">= 5" picked 7 layers (2,4,14,21,31,33,34) whose counts were 5 or 6
    # while many cold layers sat at 4. Five of those seven were wrong.
    #
    # The bound separates cleanly on the same data:
    #     38 -> 1321,  34 -> 1199,  33 -> 1010,  39 -> 660,  then 21 -> 417
    # a 1.58x drop into 5th place, and those top four are EXACTLY the set the reference implementation
    # measured on the reference model from runtime row-max over 3 prompts. Two independent methods on
    # two different models agreeing is far better evidence than my threshold was.
    #
    # The cut is the largest relative gap among the top candidates rather than an
    # absolute number, because the bound scales with d and with the gammas and no fixed
    # value would transfer between models -- which was the original mistake.
    ranked = sorted(rows, key=lambda r: -r[7])
    limit = min(args.max_hot, len(ranked) - 1)
    best_gap, cut = 0.0, 0
    for i in range(1, limit + 1):
        gap = ranked[i - 1][7] / max(ranked[i][7], 1e-9)
        if gap > best_gap:
            best_gap, cut = gap, i
    if best_gap >= args.min_gap:
        hot_layers = sorted(r[0] for r in ranked[:cut])
    else:
        hot_layers = []
    print()
    print(f"ranked by CS_bound: " + ", ".join(f"{r[0]}={r[7]:.0f}" for r in ranked[:8]))
    print(f"largest relative gap in the top {limit}: {best_gap:.2f}x after "
          f"{cut} layer(s)   (need >= {args.min_gap}x to cut)")

    print()
    print(f"{'blk':>4} {'cospk_hd':>8} {'max_cospk':>9} {'maxL|gq|':>8} "
          f"{'maxL|gk|':>8} {'mean_gq':>8} {'std_gq':>7} {'CS_bound':>10}")
    for r in rows:
        flag = "  <- HOT" if r[0] in hot_layers else ""
        print(f"{r[0]:>4} {r[1]:>9} {r[2]:>9} {r[3]:>8.2f} {r[4]:>8.2f} "
              f"{r[5]:>8.2f} {r[6]:>7.3f} {r[7]:>10.0f}{flag}")

    print()
    print(f"Cauchy-Schwarz bounds above are all >> the ~70 safe ceiling because d="
          f"{gammas[sorted(gammas)[0]][0].numel()} (across-heads norm). Too loose to "
          f"classify with -- shown so that is visible, not assumed.")
    print()
    if hot_layers:
        print(f"CANDIDATE HOT LAYERS ({len(hot_layers)}): "
              f"{','.join(str(b) for b in hot_layers)}")
        print(f"  the reference implementation's measured set for the reference model was 33,34,38,39 -- a DIFFERENT "
              f"model. Do not assume these coincide.")
    else:
        print("No layer reached the co-spike threshold: every head looks cold, so "
              "gamma structure does NOT explain the quality loss and the row-max "
              "must be measured at runtime instead.")
    print()
    print("NEXT: this is a prediction from weights, not a row-max measurement. "
          "Confirm before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
