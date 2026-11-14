#!/usr/bin/env python3
"""Build demo/talk-deck-trn3.html: every validated slide as a BEFORE / AFTER pair.

BEFORE = the slide exactly as in the original deck (~/Downloads/talk-deck_v2.html, md5 a03a9769),
         red BEFORE banner, data-ref "<id>-BEFORE". Delete these after the review.
AFTER  = the same slide with the numbers measured on LingBot-World v2, 8 trn3 chips
         (docs/talk/validate/RESULTS.md), green AFTER banner, data-ref "<id>".
Before/after slides get a two-lane (or n-lane) race animation: bars grow at the same rate,
drawn to scale from the measured numbers, so the slower way is still running when the faster ends.
B6b keeps its hand-made animation (anim/b6b_copy_vs_in_place.html).

  python3 docs/talk/validate/build_trn3_deck.py [--orig ~/Downloads/talk-deck_v2.html]
"""
import argparse
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--orig", default=os.path.expanduser("~/Downloads/talk-deck_v2.html"))
ap.add_argument("--out", default=os.path.join(REPO, "demo", "talk-deck-trn3.html"))
args = ap.parse_args()

orig = open(args.orig, encoding="utf-8").read()
first = orig.rfind("<", 0, orig.index('data-ref="S0"'))
last_ref = [m.start() for m in re.finditer(r'data-ref="', orig)][-1]
k = orig.find("<script", last_ref)
end_cells = orig.index("</div><!-- /stage -->", last_ref)   # keep the stage close, #jump, #hud and #notes in SUFFIX
PREFIX, SUFFIX = orig[:first], orig[end_cells:]


def cell_of(ref):
    i = orig.index('data-ref="%s"' % ref)
    st = orig.rfind("<", 0, i)
    nxt = orig.find('<div class="cell" data-ref="', i)
    if nxt < 0:
        nxt = end_cells
    seg = orig[st:nxt]
    e = seg.rfind("</div></div>") + len("</div></div>")
    return seg[:e]


CSS = """
<style>
.bab{position:absolute;top:20px;right:28px;z-index:5;padding:7px 18px;border-radius:9px;font-weight:800;
     font-size:21px;letter-spacing:.12em;color:#fff}
.bab.before{background:#d03b3b}
.bab.after{background:#0c8f0c}
.bab small{display:block;font-size:11px;letter-spacing:.04em;font-weight:600;opacity:.9}
.tbl td.ok{color:#4cd04c}
.tbl td.no{color:#ff6b6b}
</style>
"""


def banner(kind, why):
    lab = "BEFORE" if kind == "before" else "AFTER"
    return '<div class="bab %s">%s<small>%s</small></div>' % (kind, lab, html.escape(why))


def before(ref, why):
    c = cell_of(ref)
    c = c.replace('data-ref="%s"' % ref, 'data-ref="%s-BEFORE"' % ref, 1)
    c = re.sub(r'(<section class="slide[^"]*">)', lambda m: m.group(1) + banner("before", why), c, count=1)
    c = c.replace('<div class="nt">', '<div class="nt"><p class="chk">BEFORE: the original slide, kept for the review. '
                  'Delete this slide once the AFTER version is approved.</p>', 1)
    return c


RACE = open(os.path.join(HERE, "anim", "race_template.html"), encoding="utf-8").read()


def race(cfg, title):
    doc = RACE.replace("/*CONFIG*/null", json.dumps(cfg))
    return ('<div class="anim" style="width:94%;margin:0 auto;height:auto;aspect-ratio:1600/' + str(cfg.get("h", 560)) + '">'
            '<iframe title="%s" aria-label="%s" tabindex="-1" srcdoc="%s"></iframe></div>'
            % (html.escape(title), html.escape(title), html.escape(doc, quote=True)))


def stats(items, cols=None):
    cols = cols or len(items)
    out = '<div class="stats c%d">' % cols
    for v, unit, label in items:
        out += '<div class="stat g"><div class="v">%s <small>%s</small></div><div class="l">%s</div></div>' % (v, unit, label)
    return out + "</div>"


def table(head, rows, cls="tbl"):
    out = '<table class="%s"><tr>' % cls + "".join("<th>%s</th>" % h for h in head) + "</tr>"
    for r in rows:
        out += "<tr>" + "".join("<td>%s</td>" % c for c in r) + "</tr>"
    return out + "</table>"


def after(ref, h1, body, takeaway, notes, why="LingBot-World v2 · 8 trn3 chips"):
    tk = ('<p class="takeaway"><span class="k">Takeaway</span><span>%s</span></p>' % takeaway) if takeaway else ""
    nts = "".join("<p>%s</p>" % n for n in notes)
    return ('<div class="cell" data-ref="%s"><section class="slide">%s<h1>%s</h1><div class="body"><div class="stack" '
            'style="gap:14px">%s</div></div>%s</section><div class="nt"><p class="meta">AFTER · measured on LingBot-World '
            'v2, 8 trn3 chips (xl-lnc2, TP4 x CP8 unless noted), 81 frames 480x768 · source docs/talk/validate/RESULTS.md'
            '</p>%s</div></div>\n' % (ref, banner("after", why), h1, body, tk, nts))


RED, GRN, GRY, BLU, YEL = "#e5484d", "#3fbf3f", "#55544f", "#3987e5", "#fab219"
DIT, VAE = "#3987e5", "#199e70"

# ───────────────────────── the pairs ─────────────────────────
P = []


def pair(ref, why_before, after_html):
    P.append(before(ref, why_before))
    P.append(after_html)


title = ('<div class="cell" data-ref="T0"><section class="slide title-slide"><h1>LingBot-World v2 on Trainium3: '
         'the validated numbers</h1><p class="byline kind">Every slide twice: <b style="color:#ff6b6b">BEFORE</b> '
         '(the original) then <b style="color:#4cd04c">AFTER</b> (measured on LingBot, 8 trn3 chips). Delete the '
         'BEFORE slides once reviewed.</p></section><div class="nt"><p>Built by docs/talk/validate/build_trn3_deck.py '
         'from the original deck and RESULTS.md. Search for data-ref="...-BEFORE" to remove the originals.</p></div></div>\n')

pair("S0a", "Wan2.2-TI2V-5B, one trn2 device",
     after("S0a", "Four ideas, one worked example",
           stats([("21.6 → 4.09", "s", "LingBot-World v2, one 81-frame 480×768 video, 8 trn3 chips"),
                  ("3.75 → 19.8", "fps", "eager on the device → all four ideas")], 2),
           "The same four ideas, measured on the world model itself.",
           ["21.6 s: eager, nothing compiled, native SDPA (4x8_eager). 4.09 s: loop_engineering_8dev flags without "
            "chained calls (4x8_d8nochain), 19.81 / 19.92 fps in two processes, bitwise repeatable."]))

pair("S1a", "the counts were right; the forward count needs one word",
     after("S1a", "The pipeline we optimize, end to end",
           table(["", "LingBot-World v2 (measured config)"],
                 [["T5", "once per prompt"], ["DiT", "14B, 40 blocks, 7 chunks × 3 latent frames, 4,320 tokens per chunk, "
                   "KV cache up to 17,280 keys, 4 steps, no CFG"],
                  ["forwards per video", "35 (28 denoise + 7 cache commits); 34 when the last commit is skipped "
                   "(<code>LINGBOT_SKIP_LAST_COMMIT=1</code>)"],
                  ["VAE", "decodes 21 latent frames into 81 frames"]]),
           None, ["35 forwards counted by the [BENCH] DiT forwards line of 4x8_eager and 4x8_compile."]))

pair("S3a", "Wan2.2-TI2V-5B, one trn2 device: 76.6 s",
     after("S3a", "Eager first: move the model to the device",
           stats([("21.6", "s", "per video, eager, native SDPA"), ("19.2", "s", "DiT (35 forwards, 0.54 s each)"),
                  ("2.03", "s", "VAE (eager, 32 ranks)"), ("0.0129", "", "step-0 error vs fp32 CPU")], 4),
           "Correct first, fast second: every optimization must reproduce this output.",
           ["Config 4x8_eager: LINGBOT_COMPILE_DIT=0, ATTENTION_BACKEND=sdpa, every loop flag off. First pass "
            "(per-op compile) 184.8 s; GEN 19.59 s steady. VAE vs an fp32 CPU decode 56.4 dB.",
            "Code: generate.py with the eager settings above (the BEFORE slide's inference_neuron_ti2v.py is Wan2.2)."]))

pair("S4b", "Wan2.2 models/fast_dit.py, FFN 3072 → 14336 → 3072",
     after("S4b", "One graph per transformer block",
           '<pre class="code" style="font-size:17px">for i, block in enumerate(self.blocks):          # wan/modules/model_fast.py\n'
           '    self.blocks[i] = torch.compile(block, backend="neuron",\n'
           '                                   fullgraph=True, dynamic=False)</pre>' +
           stats([("40", "blocks", "LingBot-World v2 14B"), ("5120 → 13824 → 5120", "", "FFN")], 2),
           "Compile one graph per repeated block.", ["COMPILE_SCOPE=blocks, model_fast.py around line 2985."]))

pair("S4c", "Wan2.2: 4.4 → 1.91 s/step, 76.6 → 36.8 s",
     after("S4c", "What one graph per block buys",
           race({"unit": "s", "lanes": [
               {"name": "eager", "sub": "every op launched alone", "color": RED,
                "segs": [[19.59, DIT, "DiT"], [2.03, VAE, "VAE"]], "end": "21.6 s", "note": "0.54 s per DiT forward"},
               {"name": "one graph per block", "sub": "torch.compile, fullgraph", "color": GRN,
                "segs": [[6.07, DIT, "DiT"], [2.02, VAE, "VAE"]], "end": "8.09 s", "note": "0.173 s per DiT forward"}]},
                "eager vs one compiled graph per block") +
           stats([("0.54 → 0.173", "s per forward", "DiT, 3.1×"), ("21.6 → 8.09", "s per video", "2.7×")], 2),
           "Compile one graph per repeated block, with the collectives inside it.",
           ["4x8_eager vs 4x8_compile: same flags, only LINGBOT_COMPILE_DIT. VAE eager in both."]))

pair("S4e", "Wan2.2 VAE; the conclusion flips for LingBot",
     after("S4e", "On LingBot, compiling the VAE wins",
           race({"unit": "s", "lanes": [
               {"name": "eager", "sub": "upstream decoder, per conv", "color": RED, "segs": [[2.03, VAE, ""]], "end": "2.03 s", "note": "56.3 dB vs fp32"},
               {"name": "compiled blocks", "sub": "LINGBOT_VAE_FUSED=1", "color": YEL, "segs": [[1.38, VAE, ""]], "end": "1.38 s", "note": "56.3 dB"},
               {"name": "+ bf16, -O1, 2-D tiles, 6 frames", "sub": "today", "color": GRN, "segs": [[0.41, VAE, ""]], "end": "0.41 s", "note": "52.7 dB"}]},
                "VAE eager vs compiled") ,
           "On LingBot the VAE compiles well: 5× faster than eager.",
           ["Decode-only A/B on the same latents (talk-pipeline and talk-ladder jobs). The Wan2.2 result (compile loses) does not hold here."]))

pair("S5a", "Wan2.2 block: 56.1 ms, 3 heads",
     after("S5a", "What is still slow? Profile before writing a kernel",
           stats([("8.98", "ms", "one LingBot block, SDPA as matmul → softmax → matmul"),
                  ("6", "gaps", "Tensor idle while the softmax runs (1.54 ms)"),
                  ("10", "heads", "per rank (TP4)")], 3),
           "Profile first: Tensor waits for the softmax.",
           ["TORCH_NEURONX_ENABLE_NKI_SDPA=0, otherwise torch-neuronx swaps in an NKI kernel by itself. Pair "
            "pairs/1_sdpa_math_rank1 (rank 1, +2% vs the system trace). Whole pipeline in this mode: 13.4 s."]))

pair("S5b", "Wan2.2 profile screenshot",
     after("S5b", "Select one PyTorch call: every instruction it made",
           table(["", "LingBot"], [["source line of every attention instruction", "<code>wan/modules/attention.py:802</code>, F.scaled_dot_product_attention"],
                                   ["demo pair (one core, attention only)", "<code>pairs/demo_sdpa</code>: Tensor idle 0.71 → 1.88 ms while Vector/Scalar run the softmax"],
                                   ["script", "<code>demo/sdpa_vs_nki.py --mode sdpa</code>"]]),
           "Map the profile back to your code.", ["Open pairs/demo_sdpa/sdpa.neff + sdpa.ntff in Neuron Explorer; demo/README.md says what to point at."]))

pair("S5d", "Wan2.2: 39% / 36% / 16%",
     after("S5d", "Measured: Tensor waits, engines take turns",
           stats([("45.3%", "", "Tensor busy"), ("28.5%", "", "exactly one engine busy"), ("16.6%", "", "every engine idle"),
                  ("2.93 / 1.87", "ms", "softmax on Vector / Scalar")], 4),
           "Engines taking turns is time a kernel can win back.", ["engine_overlap.py on pairs/1_sdpa_math_rank1."]))

pair("S5f", "Wan2.2: 18.2 → 5.3 ms per call",
     after("S5f", "Attention: the operator that dominates",
           race({"unit": "ms", "lanes": [
               {"name": "F.scaled_dot_product_attention", "sub": "matmul → softmax → matmul", "color": RED,
                "segs": [[1.275, RED, "Tensor idle"], [1.412, GRY, "rest"]], "end": "2.69 ms", "note": "1.17 ms Tensor gap"},
               {"name": "nkilib attention_cte", "sub": "NKI flash attention", "color": GRN,
                "segs": [[0.202, RED, ""], [1.372, GRY, ""]], "end": "1.57 ms", "note": "softmax overlaps nc_matmul"}]},
                "SDPA vs attention_cte, one call") ,
           "nkilib first: production kernels, called like any torch op.",
           ["One LingBot rank's shape on one core: q [1,10,640,128], k/v [1,10,17280,128]. demo/sdpa_vs_nki.py. Red = softmax work while Tensor is idle."]))

pair("S5g", "Wan2.2: 16.2 → 13.2 s; for LingBot the result flips",
     after("S5g", "Calling attention_cte yourself can be slower",
           race({"unit": "s", "lanes": [
               {"name": "our own cte call", "sub": "copies KV into a padded buffer", "color": RED,
                "segs": [[2.91, RED, "the copy"], [6.07, DIT, "DiT"], [2.02, VAE, "VAE"]], "end": "11.0 s", "note": "error 0.034 (zeros in the softmax)"},
               {"name": "native SDPA", "sub": "the compiler puts in attention_cte", "color": GRN,
                "segs": [[6.07, DIT, "DiT"], [2.02, VAE, "VAE"]], "end": "8.09 s", "note": "error 0.012"}]},
                "explicit cte vs compiler cte"),
           "Check what the compiler already does before adding a kernel.",
           ["4x8_compilecte vs 4x8_compile. The extra time is one block graph at 17.9 ms (Tensor 10.6%) spent in the KV copy at attention.py:449. Red segment = 8.98 − 6.07 s GEN."]))

pair("S5h", "Wan2.2: 56.1 → 17.3 ms",
     after("S5h", "The same block with the kernel: 8.98 → 4.37 ms",
           race({"unit": "ms", "lanes": [
               {"name": "matmul → softmax → matmul", "sub": "SDPA, NKI swap off", "color": RED,
                "segs": [[2.32, RED, "softmax, Tensor idle"], [6.66, GRY, "rest of the block"]], "end": "8.98 ms", "note": ""},
               {"name": "attention_cte", "sub": "nc_matmul in a K-tile loop", "color": GRN,
                "segs": [[0.21, RED, ""], [4.16, GRY, "rest of the block"]], "end": "4.37 ms", "note": "2.1× faster"}]},
                "one LingBot block, before and after the kernel"),
           "Tile so the engines overlap.", ["Same rank, same shard: pairs/1_sdpa_math_rank1 and pairs/2_attention_cte_rank1."]))

pair("S5i", "Wan2.2 table",
     after("S5i", "Native SDPA vs attention_cte",
           table(["", "matmul → softmax → matmul", "attention_cte"],
                 [["block", "8.98 ms", "4.37 ms"], ["softmax while Tensor idle", "2.32 ms", "0.21 ms"],
                  ["Tensor busy", "45.3%", "56.0%"], ["3+ engines busy", "22.1%", "28.1%"],
                  ["every engine idle", "16.6%", '17.1% (collective waits and transposes stay)']], "tbl big"),
           "The kernel removes the softmax gaps; collectives keep some idle time.", []))

pair("S5j", "Wan2.2 nkilib conv3d",
     after("S5j", "LingBot's VAE: compiled blocks, not conv3d",
           stats([("2.03 → 1.38", "s", "whole decoder blocks compiled"), ("1.38 → 0.41", "s", "bf16, -O1, 2-D tiles, 6 frames per step")], 2),
           "Where the compiler is strong, give it the whole block.", ["See B4h for every step."]))

pair("S7a", "trn2: 18.64 and 25.02 fps",
     after("S7a", "The frame rate decides how many chips",
           stats([("19.8", "fps", "8 trn3 chips (best flags, no softmax shortcut)"), ("18.1", "fps", "8 trn3 chips, main (q20)"),
                  ("not run", "", "16 chips (over the 8-chip budget)")], 3),
           None, ["18.64 / 25.02 fps used chained kernel calls, which assert on trn3 (kernels/attention_cte_lb.py:767)."]))

pair("S7d", "Wan2.2 VAE 9.0 → 6.0 s; 17.88 fps",
     after("S7d", "Sequence parallelism: split the work across cores",
           race({"unit": "s", "lanes": [
               {"name": "VAE on one core", "sub": "eager", "color": RED, "segs": [[18.2, VAE, ""]], "end": "18.2 s", "note": ""},
               {"name": "one tile per core, 32 cores", "sub": "eager", "color": GRN, "segs": [[2.03, VAE, ""]], "end": "2.03 s", "note": "9×"}]},
                "VAE one core vs 32") +
           stats([("17.88 → 18.08", "fps", "half a trn3, TP4 × CP8, q20: re-run confirms")], 1),
           "Collectives are plain torch.distributed calls.", []))

pair("S9a", "VAE 44.6 dB is Wan2.2",
     after("S9a", "Validate every step against the eager reference",
           table(["config", "step-0 vs fp32", "VAE vs fp32 decode"],
                 [["eager, native SDPA", "0.0129", "56.4 dB"], ["compiled blocks", "0.0123", "56.4 dB"],
                  ["q20 (main)", "0.0108", "52.7 dB"], ["best flags (8dev, no chain)", "0.0148", "52.7 dB"],
                  ['<span style="color:#ff6b6b">our own cte call (S5g)</span>', '<span style="color:#ff6b6b">0.034</span>', "56.3 dB"]]),
           "Put a number on correctness at every step.", []))

journey = {"unit": "s", "lanes": [
    {"name": "eager", "sub": "", "color": RED, "segs": [[19.59, DIT, "GEN"], [2.03, VAE, "VAE"]], "end": "21.6 s · 3.75 fps", "note": ""},
    {"name": "+ compiled blocks", "sub": "", "color": YEL, "segs": [[6.07, DIT, "GEN"], [2.02, VAE, "VAE"]], "end": "8.09 s · 10.0 fps", "note": ""},
    {"name": "+ q20 flags (main)", "sub": "", "color": BLU, "segs": [[4.07, DIT, "GEN"], [0.41, VAE, ""]], "end": "4.48 s · 18.1 fps", "note": ""},
    {"name": "+ 8dev flags, no chain", "sub": "", "color": GRN, "segs": [[3.68, DIT, "GEN"], [0.39, VAE, ""]], "end": "4.07 s · 19.9 fps", "note": ""}]}
pair("S9b", "the Wan2.2 journey",
     after("S9b", "The journey: 21.6 → 4.07 s per video",
           race(journey, "LingBot journey on 8 trn3 chips"), None,
           ["Every row measured in the talk jobs. No A100 number exists for LingBot."]))
pair("S9c", "16 and 8 trn2 devices",
     after("S9c", "LingBot-World v2 on 8 trn3 chips: 19.9 fps",
           race(journey, "LingBot journey on 8 trn3 chips"),
           "With the softmax shortcut off; the shortcut needs chained calls, which do not run on trn3.",
           ["16 devices not re-run (over budget). The 18.40 fps 'with the check' cannot run on trn3."]))

pair("S9d", "17.3 fps / 1.07 s used a separate VAE deployment (more than 8 chips)",
     after("S9d", "LingBot-World v2: a live stream",
           race({"unit": "s", "lanes": [
               {"name": "one chunk, VAE on the same 8 chips", "sub": "serve.py, today's main", "color": YEL,
                "segs": [[0.554, DIT, "DiT"], [0.276, VAE, "VAE"], [0.179, GRY, ""]], "end": "1.01 s", "note": "11.8 fps · key press 2.05 s"}]},
                "one live chunk") +
           stats([("0.554", "s", "DiT per chunk (deck: ~0.58)"), ("2.05", "s", "key press → new frames, VAE inline"),
                  ("11.8", "fps", "stream, VAE inline")], 3),
           "In a stream, measure what the user waits for.",
           ["17.3 fps / 1.07 s need the VAE on other chips (over the 8-chip budget): not measured. live_latency.py through kubectl port-forward."]))

pair("S11c", "Wan2.2 VAE warm-up 186 / 21 / 3.6 s",
     after("S11c", "Warm-up is part of deployment",
           stats([("151.0", "s", "GEN pass 1 (compile, load)"), ("4.28", "s", "pass 2"), ("4.27", "s", "pass 3")], 3),
           "Warm up until the timing settles, then report ready.", ["4x8_final, today's main flags."]))

pair("B4h", "64 ranks on trn2",
     after("B4h", "LingBot VAE, step by step (32 ranks, trn3)",
           race({"unit": "s", "lanes": [
               {"name": "upstream, per conv", "sub": "56.3 dB", "color": RED, "segs": [[2.03, VAE, ""]], "end": "2.03 s", "note": ""},
               {"name": "whole blocks compiled", "sub": "56.3 dB", "color": YEL, "segs": [[1.38, VAE, ""]], "end": "1.38 s", "note": ""},
               {"name": "bf16 convolutions", "sub": "52.7 dB", "color": YEL, "segs": [[0.86, VAE, ""]], "end": "0.86 s", "note": ""},
               {"name": "-O1, norm as matmul", "sub": "52.7 dB", "color": BLU, "segs": [[0.65, VAE, ""]], "end": "0.65 s", "note": ""},
               {"name": "2-D tiles", "sub": "52.7 dB", "color": BLU, "segs": [[0.46, VAE, ""]], "end": "0.46 s", "note": ""},
               {"name": "6 frames per step", "sub": "52.7 dB", "color": GRN, "segs": [[0.41, VAE, ""]], "end": "0.41 s", "note": ""}]},
                "VAE ladder"), "bf16 is where the 3.6 dB goes; the padding bug is not present at this tiling.", []))

pair("B4i", "64 ranks on trn2",
     after("B4i", "Where compiling whole VAE blocks won",
           race({"unit": "s", "lanes": [
               {"name": "upstream decoder", "sub": "per-conv graphs, eager ops", "color": RED, "segs": [[2.03, VAE, ""]], "end": "2.03 s", "note": ""},
               {"name": "one graph per decoder block", "sub": "LINGBOT_VAE_FUSED=1", "color": GRN, "segs": [[1.38, VAE, ""]], "end": "1.38 s", "note": "→ 0.41 s after tuning"}]},
                "VAE upstream vs compiled blocks"), "The graph's scope is the decision.", []))

pair("B6a", "trn2, 16 devices",
     after("B6a", "Attention over cached chunks: a copy before every call",
           stats([("7.37 → 4.34", "s GEN", "read the cache in place (LINGBOT_CTE_MERGE)"),
                  ("0.032 → 0.011", "", "error vs fp32: the zero keys are gone"),
                  ("17.9", "ms", "the block graph that does the copy (Tensor 10.6%)")], 3),
           "Don't copy data to fit a kernel.", ["Copy block from the explicit-cte profile (job 3). Per-block copy share at q20: pending job talk-b6-nomerge."]))

b6b = cell_of("B6b")  # the BEFORE is the original; the AFTER is the edited slide in demo/talk-deck.html
P.append(before("B6b", "trn2: 8.95 → 5.73 s, 3.28 → 2.88 ms"))
cur = open(os.path.join(REPO, "demo", "talk-deck.html"), encoding="utf-8").read()
i = cur.index('data-ref="B6b"'); st = cur.rfind("<", 0, i); nx = cur.find('<div class="cell" data-ref="', i)
b6b_after = cur[st:nx]; b6b_after = b6b_after[:b6b_after.rfind("</div></div>") + 12]
b6b_after = re.sub(r'(<section class="slide[^"]*">)', lambda m: m.group(1) + banner("after", "LingBot-World v2 · 8 trn3 chips"), b6b_after, count=1)
P.append(b6b_after + "\n")

pair("B6c", "trn2 numbers",
     after("B6c", "The win was data movement, not new math",
           stats([("7.37 → 4.34", "s", "1 no copies: read in place"), ("4.34 → 4.08", "s", "2 fewer calls: two chunks per call"),
                  ("4.32 / 4.20 → 4.08", "s", "3 fewer DMAs: K layout / 8 V tiles per DMA")], 3), None,
           ["q20 with one flag off at a time (talk-ladder job)."]))

na = "cannot run on trn3: needs chained kernel calls (LINGBOT_CTE_CHAIN), which assert on trn3 (kernels/attention_cte_lb.py:767)"
for ref, h in [("B6d", "A softmax shortcut that checks itself"), ("B6g", "Why one constant works in 34 of 40 layers"),
               ("B6h", "The check's runtime branch in the kernel")]:
    pair(ref, "trn2, branch loop_engineering_8dev",
         after(ref, h, '<div class="card k2"><b>Not measurable on trn3 today</b><p>%s. The shortcut only switches on '
               'with chaining (model_fast.py:1389 on loop_engineering_8dev).%s</p></div>'
               % (na, " The code on this slide matches attention_cmax2_lb.py:389-403." if ref == "B6h" else ""),
               "Keep as a trn2 result, or drop.", [], why="trn2 only · cannot run on trn3"))

pair("B6e", "trn2: 3.94 → 3.74 → 3.66 s",
     after("B6e", "Changes to a copy of the kernel, one at a time",
           table(["q20 with one flag off", "GEN", "cost of turning it off"],
                 [["8 V tiles per DMA (LINGBOT_CTE_VGROUP=8)", "4.20 s", "+0.12 s"],
                  ["K in the all-gather's layout (LINGBOT_K_BLOCKED)", "4.32 s", "+0.24 s"],
                  ["cross-attention cache in the kernel's layout (LINGBOT_CROSS_NMAJOR)", "4.12 s", "+0.04 s"],
                  ["runtime-built DMA descriptors (LINGBOT_CTE_KV_DGE=none)", "pending", ""]]), None, ["q20 all on: 4.08 s."]))

pair("B6f", "trn2, TP4 × CP16, 2.53-2.60 ms",
     after("B6f", "One block now (q20, TP4 × CP8)",
           stats([("2.92", "ms", "steady block (540 tokens per rank)"), ("64.5%", "", "Tensor busy"), ("8.1%", "", "every engine idle"),
                  ("1.22 / 1.09", "ms", "attention kernel / projections + FFN")], 4), None,
           ["Not 1:1 with the BEFORE: CP8 holds twice the tokens per rank of CP16. The 19.9 fps block: pending job talk-b6f-8dev."]))

pair("B7c", "Wan2.2 layouts on one trn2 device",
     after("B7c", "Layouts for LingBot on 8 trn3 chips",
           race({"unit": "s", "lanes": [
               {"name": "TP2 × CP16", "sub": "", "color": RED, "segs": [[5.05, DIT, "GEN"], [0.41, VAE, ""]], "end": "5.46 s · 14.9 fps", "note": ""},
               {"name": "TP8 × CP4", "sub": "TP across two chips", "color": YEL, "segs": [[4.40, DIT, "GEN"], [0.41, VAE, ""]], "end": "4.81 s · 16.8 fps", "note": ""},
               {"name": "TP4 × CP8", "sub": "TP inside a chip", "color": GRN, "segs": [[4.08, DIT, "GEN"], [0.42, VAE, ""]], "end": "4.50 s · 18.0 fps", "note": ""}]},
                "LingBot layouts"), "Keep the chatty collective inside a chip.", []))

cells = title + "".join(P)
out = PREFIX.replace("</style>", CSS.replace("<style>", "").replace("</style>", "") + "</style>", 1)
out = out.replace("<title>", "<title>[trn3 validated] ", 1) + cells + SUFFIX
open(args.out, "w", encoding="utf-8").write(out)
print("wrote", args.out, "slides:", cells.count('<div class="cell"'))
