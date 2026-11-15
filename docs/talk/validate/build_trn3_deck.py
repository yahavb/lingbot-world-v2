#!/usr/bin/env python3
"""Build demo/talk-deck-trn3.html: every validated slide as a BEFORE / AFTER pair.

BEFORE = the slide exactly as in the original deck (~/Downloads/talk-deck_v2.html), red banner,
         data-ref "<id>-BEFORE". Delete these after the review.
AFTER  = THE SAME SLIDE (same title, layout, figure, point), with only the Wan2.2 / trn2 numbers,
         file names and code replaced by the LingBot-World v2 ones measured on 8 trn3 chips
         (docs/talk/validate/RESULTS.md). Green banner, data-ref "<id>-AFTER". Its notes list every
         change, old -> new. Every edit is asserted: the build fails if the original text moved.
B6b AFTER is the approved two-lane animation (demo/talk-deck.html).

  python3 docs/talk/validate/build_trn3_deck.py
"""
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ORIG = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/talk-deck_v2.html")
OUT = os.path.join(REPO, "demo", "talk-deck-trn3.html")
GH = "https://github.com/yahavb/lingbot-world-v2/blob/main/"

orig = open(ORIG, encoding="utf-8").read()
first = orig.rfind("<", 0, orig.index('data-ref="S0"'))
last_ref = [m.start() for m in re.finditer(r'data-ref="', orig)][-1]
end_cells = orig.index("</div><!-- /stage -->", last_ref)
PREFIX, SUFFIX = orig[:first], orig[end_cells:]


def cell_of(src, ref):
    i = src.index('data-ref="%s"' % ref)
    st = src.rfind("<", 0, i)
    nx = src.find('<div class="cell" data-ref="', i)
    seg = src[st:nx if nx > 0 else len(src)]
    return seg[:seg.rfind("</div></div>") + len("</div></div>")]


CSS = """
.bab{position:absolute;top:20px;right:28px;z-index:5;padding:7px 18px;border-radius:9px;font-weight:800;
     font-size:21px;letter-spacing:.12em;color:#fff;text-align:left}
.bab.before{background:#d03b3b}
.bab.after{background:#0c8f0c}
.bab small{display:block;font-size:11px;letter-spacing:.04em;font-weight:600;opacity:.92}
.tl-lbl{font-size:15px;fill:#c3c2b7;font-weight:600}
.tl-ax{font-size:12px;fill:#898781}
"""


def banner(kind, why):
    return '<div class="bab %s">%s<small>%s</small></div>' % (kind, kind.upper(), html.escape(why))


def with_banner(cell, ref, kind, why, notes_html=None):
    c = cell.replace('data-ref="%s"' % ref, 'data-ref="%s-%s"' % (ref, kind.upper()), 1)
    c = re.sub(r'(<section class="slide[^"]*">)', lambda m: m.group(1) + banner(kind, why), c, count=1)
    if notes_html is not None:
        a = c.index('<div class="nt">') + len('<div class="nt">')
        b = c.rindex("</div></div>")
        c = c[:a] + notes_html + c[b:]
    return c


# ───────────── figure builders, in the deck's own SVG classes ─────────────
def hbars(cols, rows, x0=440, maxw=622, rowh=66, w=1152, vmax=None):
    """cols: [(x, LABEL)]; rows: dict(label, v, text, bar, t, chips=[], step=None, extra=[(x, text, cls)])."""
    vmax = vmax or max(r["v"] for r in rows)
    h = 22 + rowh * len(rows) + 8
    s = '<svg viewBox="0 0 %d %d" width="%d" height="%d" role="img" aria-label="%s">' % (
        w, h, w, h, html.escape("; ".join("%s %s" % (r["label"], r["text"]) for r in rows)))
    for x, lab in cols:
        s += '<text class="lbl" x="%d" y="12">%s</text>' % (x, lab)
    s += '<line class="ax" x1="0" y1="22" x2="%d" y2="22"/>' % w
    for i, r in enumerate(rows):
        y = 22 + i * rowh + rowh / 2
        bw = maxw * r["v"] / vmax
        g = '<g%s>' % (' data-step="%d"' % r["step"] if r.get("step") else "")
        g += '<text class="tk" x="0" y="%.1f">%s</text>' % (y + 6, r["label"])
        for k, ch in enumerate(r.get("chips", [])):
            cx = 352 + 34 * k
            g += ('<rect class="ichip" x="%d" y="%.1f" width="28" height="26" rx="7"/><text class="ichip-t" x="%d" y="%.1f" '
                  'text-anchor="middle">%s</text>' % (cx, y - 14, cx + 14, y + 4, ch))
        g += '<rect class="%s" x="%d" y="%.1f" width="%.1f" height="32" rx="4"/>' % (r["bar"], x0, y - 17, bw)
        g += '<text class="%s" x="%.1f" y="%.1f">%s</text>' % (r["t"], x0 + bw + 10, y + 4, r["text"])
        for ex, et, ec in r.get("extra", []):
            g += '<text class="%s" x="%d" y="%.1f">%s</text>' % (ec, ex, y + 4, et)
        s += g + "</g>"
    return s + "</svg>"


TLROWS = ["Tensor", "Vector", "Scalar", "GpSimd", "Sync"]
TLCOL = {"Tensor": "#3987e5", "Vector": "#199e70", "Scalar": "#c98a2b", "GpSimd": "#7c6bd6", "Sync": "#6b6a64"}


def timeline(prof, w, h, sel_label, gap_ms=0.05):
    """Engine timeline drawn from the profile's instruction table (anim/profiles/*.json):
    each engine's busy time; the selected call's instructions in red; Tensor gaps > gap_ms boxed."""
    d = json.load(open(os.path.join(HERE, "anim", "profiles", prof)))
    ms = d["ms"]; X0, X1 = 92, w - 12; top = 34; rh = (h - top - 40) / len(TLROWS)
    sx = lambda t: X0 + (X1 - X0) * t / ms
    s = ('<svg viewBox="0 0 %d %d" width="%d" height="%d" role="img" aria-label="Engine timeline of one LingBot block, '
         'drawn from its profile; red: %s">' % (w, h, w, h, html.escape(sel_label)))
    s += '<rect x="0" y="0" width="%d" height="%d" rx="8" fill="#101012" stroke="#2c2c2a"/>' % (w, h)
    s += '<text class="tl-lbl" x="12" y="22">drawn from the profile · red = %s</text>' % html.escape(sel_label)
    for i, e in enumerate(TLROWS):
        y = top + i * rh
        s += '<text class="tl-lbl" x="12" y="%.1f">%s</text>' % (y + rh * 0.62, e)
        s += '<rect x="%d" y="%.1f" width="%d" height="%.1f" fill="#18181b"/>' % (X0, y + 3, X1 - X0, rh - 6)
        for a, b in d["eng"].get(e, []):
            s += '<rect x="%.1f" y="%.1f" width="%.2f" height="%.1f" fill="%s"/>' % (sx(a), y + 5, max(.6, sx(b) - sx(a)), rh - 10, TLCOL[e])
        for a, b in d["att"].get(e, []):
            s += '<rect x="%.1f" y="%.1f" width="%.2f" height="%.1f" fill="#e5484d"/>' % (sx(a), y + 5, max(.6, sx(b) - sx(a)), rh - 10)
    # Tensor gaps inside the selected call's span
    att = [iv for e in TLROWS for iv in d["att"].get(e, [])]
    if att:
        a0, a1 = min(a for a, _ in att), max(b for _, b in att)
        ten = sorted(d["eng"]["Tensor"])
        for (p0, p1), (q0, q1) in zip(ten, ten[1:]):
            vs = sum(max(0, min(y, q0) - max(x, p1)) for x, y in d["att"].get("Vector", []))
            if q0 - p1 > gap_ms and p1 >= a0 and q0 <= a1 and vs > 0.5 * (q0 - p1):
                s += ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" stroke="#fab219" stroke-width="2.5" '
                      'stroke-dasharray="6 4"/>' % (sx(p1), top + 1, sx(q0) - sx(p1), rh - 2))
    for t in range(0, int(ms) + 1):
        if ms > 4 or True:
            pass
    step = 1 if ms > 3 else 0.5
    t = 0.0
    while t <= ms + 1e-9:
        s += '<text class="tl-ax" x="%.1f" y="%d" text-anchor="middle">%g ms</text>' % (sx(t), h - 14, t)
        t += step
    return s + "</svg>"


def code_panel(cap, path, line, lines, hl=0, size=16):
    pre = "".join('<span class="ln%s">%s</span>' % (" hl" if i == hl else "", html.escape(l)) for i, l in enumerate(lines))
    return ('<div class="code"><div class="cap"><span>%s</span><span class="lnks"><a class="lnk" href="%s%s#L%s" '
            'target="_blank" rel="noopener">%s</a></span></div><pre style="font-size:%dpx">%s</pre></div>'
            % (cap, GH, path, line, path, size, pre))


def s2c_svg():
    """16 trn3 chips as neuron-ls on trn3-dev1 lists them: each chip links to its partner (d xor 1)
    and to every other chip of the same parity (rows 0-6 printed; the pattern holds for all of them)."""
    W, H = 636, 424
    o = ('<svg viewBox="0 0 %d %d" width="%d" height="%d" role="img" aria-label="A Trainium3 node as neuron-ls on '
         'trn3-dev1 lists it: 16 chips of 8 cores. Each chip links to its partner (0 and 1, 2 and 3, ...) and to every '
         'other chip of the same parity: the 8 even chips are all linked to each other, and so are the 8 odd chips.">' % (W, H, W, H))
    o += '<text class="lbl" x="0" y="14">TRAINIUM3 NODE · 16 CHIPS · 128 CORES · LINKS FROM neuron-ls</text>'
    cw, gap, x0 = 66, 8, 26
    ys = {0: 92, 1: 300}
    for par, y in ys.items():
        o += '<rect class="bx-w" x="%d" y="%d" width="%d" height="%d" rx="12" style="fill:none"/>' % (x0 - 12, y - 30, 8 * cw + 7 * gap + 24, 104)
        lab = "even chips: every one links to every other" if par == 0 else "odd chips: every one links to every other"
        o += '<text class="t-s" x="%d" y="%d">%s</text>' % (x0 - 4, y - 38 if par == 0 else y + 92, lab)
    for k in range(8):
        x = x0 + k * (cw + gap)
        o += '<line class="br" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (x + cw / 2, ys[0] + 48, x + cw / 2, ys[1])
        for par in (0, 1):
            d = 2 * k + par
            cls = "bx-a" if d in (0, 1) else "bx"
            o += '<rect class="%s" x="%d" y="%d" width="%d" height="48" rx="8"/>' % (cls, x, ys[par], cw)
            o += '<text class="t" x="%d" y="%d" text-anchor="middle">chip %d</text>' % (x + cw / 2, ys[par] + 30, d)
    ly = (ys[0] + 48 + ys[1]) / 2
    o += '<rect x="%d" y="%d" width="236" height="30" rx="6" style="fill:#141416"/>' % (x0 + 3 * (cw + gap) - 10, ly - 15)
    o += '<text class="t-s" x="%d" y="%d">partner links: 0–1, 2–3, … 14–15</text>' % (x0 + 3 * (cw + gap), ly + 6)
    return o + "</svg>"


# ───────────── edit machinery ─────────────
class Miss(Exception):
    pass


def edit(cell, ops):
    """ops: ("r", old, new) | ("re", pattern, new) | ("svg", n, new) | ("code", n, new) | ("img", new) | ("table", new)."""
    body_end = cell.index('<div class="nt">')
    b, notes = cell[:body_end], cell[body_end:]
    for op in ops:
        k = op[0]
        if k == "r":
            if op[1] not in b:
                raise Miss(op[1][:80])
            b = b.replace(op[1], op[2])
        elif k == "re":
            b2 = re.sub(op[1], op[2], b, count=1, flags=re.S)
            if b2 == b:
                raise Miss(op[1][:80])
            b = b2
        elif k in ("svg", "code", "img", "table"):
            pat = {"svg": r"<svg\b.*?</svg>", "code": r'<div class="code">.*?</pre></div>',
                   "img": r"<img\b[^>]*>", "table": r"<table\b.*?</table>"}[k]
            ms = list(re.finditer(pat, b, flags=re.S))
            n = op[1] if k in ("svg", "code") else 0
            new = op[2] if k in ("svg", "code") else op[1]
            if len(ms) <= n:
                raise Miss("%s #%d" % (k, n))
            m = ms[n]
            b = b[:m.start()] + new + b[m.end():]
    return b + notes


def notes(changes, extra=()):
    li = "".join("<li>%s</li>" % c for c in changes)
    ex = "".join("<p>%s</p>" % e for e in extra)
    return ('<p class="meta">AFTER · the same slide, LingBot-World v2 on 8 trn3 chips (xl-lnc2, TP4 × CP8), 81 frames '
            '480×768 · numbers from docs/talk/validate/RESULTS.md</p><p><b>Changed from the original:</b></p><ul>%s</ul>%s'
            % (li, ex))


PAIRS = []


def pair(ref, why_before, ops, changes, extra=(), why_after="LingBot-World v2 · 8 trn3 chips", src=None):
    c = cell_of(src or orig, ref)
    PAIRS.append(with_banner(c, ref, "before", why_before,
                             None))
    try:
        a = edit(c, ops) if ops else c
    except Miss as e:
        sys.exit("EDIT MISS on %s: %r" % (ref, str(e)))
    PAIRS.append(with_banner(a, ref, "after", why_after, notes(changes, extra)))


# ═══════════════════ the slides ═══════════════════
pair("S0a", "Wan2.2-TI2V-5B, one trn2 device: 76.6 → 8.7 s",
     [("r", "Wan2.2-TI2V-5B", "LingBot-World v2"), ("r", "704×544", "480×768"),
      ("r", "one Trainium2 device", "8 Trainium3 chips"), ("r", "from 76.6 s to 8.7 s", "from 21.6 s to 4.07 s")],
     ["Wan2.2-TI2V-5B → LingBot-World v2", "704×544 → 480×768", "one Trainium2 device → 8 Trainium3 chips",
      "76.6 → 8.7 s → 21.6 → 4.07 s (eager → loop_engineering_8dev flags without chained calls)"])

pair("S1a", "numbers on this slide are already LingBot's", [],
     ["no number on the slide changes: 14B, 40 blocks, 7 chunks, 4,320 tokens, 17,280 keys, 4 steps are LingBot's",
      "measured: 35 forwards per video (28 denoise + 7 commits), 34 with LINGBOT_SKIP_LAST_COMMIT=1"])

pair("S2c", "the 4 × 4 torus was read off neuron-ls on a trn2.48xlarge",
     [("r", "One node: 16 chips in a 2-D torus", "One node: 16 chips, each linked to 8 others"),
      ("svg", 0, s2c_svg()),
      ("r", "4 <small>neighbours per chip</small>", "8 <small>linked chips per chip</small>"),
      ("r", "over NeuronLink · one each way in its row and in its column, wrapping at the edges",
       "its partner (0–1, 2–3, …) and every other chip of the same parity"),
      ("r", "8 <small>cores in a chip</small>", "8 <small>cores in a chip</small>"),
      ("re", r'(<p class="foot">Source: )', r'\1neuron-ls on trn3-dev1 (CONNECTED DEVICES) · ')],
     ["title: \"16 chips in a 2-D torus\" → \"16 chips, each linked to 8 others\"",
      "figure: 4 × 4 torus, 4 neighbours (read on a trn2.48xlarge) → the links neuron-ls lists on trn3-dev1: chip 0 → 1, 2, 4, 6, 8, 10, 12, 14; chip 1 → 0, 3, 5, …, 15",
      "4 neighbours per chip → 8 linked chips per chip; 144 GB HBM per chip confirmed"],
     ["neuron-ls printed rows 0-6 in the job logs (head -20); every row follows the pattern partner + same parity. Check rows 7-15 on the box before re:Invent.",
      "This also explains kiro/core_order.py: each CP8 group must sit on chips that are all linked."])

pair("S3a", "Wan2.2-TI2V-5B, one trn2 device: 76.6 s",
     [("code", 0, code_panel("Move the model to the device", "generate.py", 663,
                             ['torch.neuron.set_device(local_rank)', 'dist.init_process_group(backend="neuron", ...)',
                              'text_encoder.model.to(device)'], size=17)),
      ("r", '<div class="v">2 <small>changes</small></div><div class="l">RoPE with real cos/sin tables · no float64</div>',
       '<div class="v">19.2 + 2.03 <small>s</small></div><div class="l">DiT (35 forwards) + VAE, eager, native SDPA</div>'),
      ("r", "76.6 s <small>baseline</small>", "21.6 s <small>baseline</small>"),
      ("r", "Correct, and saved as the reference", "Correct (0.0129 vs fp32), and saved as the reference")],
     ["code: Wan2.2 inference_neuron_ti2v.py → LingBot generate.py:663 / :341",
      "\"2 changes: RoPE with real cos/sin tables · no float64\" (Wan2.2 port) → 19.2 s DiT + 2.03 s VAE",
      "76.6 s → 21.6 s baseline (config 4x8_eager: nothing compiled, native SDPA, every loop flag off)",
      "added: step-0 error vs fp32 CPU 0.0129"])

pair("S4b", "Wan2.2 models/fast_dit.py, FFN 3072 → 14336 → 3072",
     [("code", 0, code_panel("One graph per block", "wan/modules/model_fast.py", 2985,
                             ['_scope = os.environ.get("COMPILE_SCOPE", "blocks")',
                              'for _i in range(len(self.blocks)):',
                              '    self.blocks[_i] = torch.compile(',
                              '        self.blocks[_i], backend="neuron", fullgraph=True, dynamic=False)'], hl=2))],
     ["code: Wan2.2 models/fast_dit.py → LingBot wan/modules/model_fast.py:2985 (COMPILE_SCOPE=blocks)",
      "notes: LingBot's FFN is 5120 → 13824 → 5120 (Wan2.2-5B: 3072 → 14336 → 3072)"])

pair("S4c", "Wan2.2: 4.4 → 1.91 s/step, 76.6 → 36.8 s",
     [("r", "shared by all 30 DiT blocks", "shared by all 40 DiT blocks"),
      ("r", "4.4 → 1.91 <small>s/step</small>", "0.54 → 0.173 <small>s/forward</small>"),
      ("r", "76.6 → 36.8 <small>s</small>", "21.6 → 8.09 <small>s</small>")],
     ["30 → 40 DiT blocks", "DiT 4.4 → 1.91 s/step → 0.54 → 0.173 s per forward (no CFG in LingBot)",
      "76.6 → 36.8 s → 21.6 → 8.09 s per video (4x8_eager vs 4x8_compile)"])

pair("S4e", "Wan2.2 VAE; for LingBot the result is the opposite",
     [("r", "Compile is not always the answer: the VAE", "On LingBot, compiling the VAE wins"),
      ("table", '<table class="tbl big"><thead><tr><th>VAE</th><th class="r">Run time</th><th class="r">vs fp32 decode</th></tr></thead><tbody>'
                '<tr class="bad"><td>Eager, upstream decoder (per conv)</td><td class="r">2.03 s</td><td class="r">56.3 dB</td></tr>'
                '<tr><td>Whole decoder blocks compiled</td><td class="r">1.38 s</td><td class="r">56.3 dB</td></tr>'
                '<tr class="win"><td><b>+ bf16, -O1, 2-D tiles, 6 frames per step</b></td><td class="r"><b>0.41 s</b></td><td class="r">52.7 dB</td></tr>'
                '</tbody></table>'),
      ("code", 0, code_panel("One graph per decoder block", "wan/modules/vae2_1_fused.py", 325,
                             ['return torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)'])),
      ("r", "Compile where shapes are static and the compiler is strong; elsewhere, call a kernel.",
       "Measure both: on LingBot the compiler wins on the VAE, 5× over eager.")],
     ["title: \"Compile is not always the answer\" → \"On LingBot, compiling the VAE wins\" (the measured result is the opposite of Wan2.2)",
      "table: Wan2.2 compile 11.5-12.9 s / static 12.8-14.2 s / eager+NKI 1.7 s → LingBot eager 2.03 s, compiled blocks 1.38 s, tuned 0.41 s",
      "column Compile time (not measured for LingBot) → accuracy vs fp32 decode",
      "code: Wan2.2 vae2_2.py _rms_silu → LingBot vae2_1_fused.py:325", "takeaway rewritten to match"])

pair("S5a", "Wan2.2 block: 56.1 ms, 3 heads, Explorer screenshot",
     [("r", "One compiled DiT block, native SDPA, one core: 56.1 ms",
       "One compiled LingBot DiT block, SDPA as matmul → softmax → matmul, one core: 8.98 ms"),
      ("img", timeline("iv_blk_sdpa.json", 726, 478, "F.scaled_dot_product_attention (attention.py:802)")),
      ("r", "<b>Three long Tensor gaps:</b> ~22 of 56 ms", "<b>Five Tensor gaps:</b> ~0.3 ms each, 1.46 of 8.98 ms (dashed)"),
      ("r", "<b>Three times,</b> once per head", "<b>All inside attention:</b> 2.32 ms of softmax with Tensor idle")],
     ["screenshot of the Wan2.2 block → engine timeline drawn from the LingBot block's profile (pairs/1_sdpa_math_rank1, rank 1)",
      "56.1 ms → 8.98 ms; three gaps ~22 ms → five softmax gaps of ~0.3 ms, 1.46 ms in all (1.37-1.67, 2.52-2.82, 3.67-3.97, 4.84-5.12, 6.34-6.61 ms from the block start; Vector 93-99% and Scalar 87-97% busy in them, all from attention.py:802). A sixth 85 us gap at 7.83 ms is not softmax.", "\"three times, once per head\" → 2.32 ms softmax with Tensor idle"],
     ["TORCH_NEURONX_ENABLE_NKI_SDPA=0, otherwise torch-neuronx (SDK 2.33) swaps in an NKI kernel by itself.",
      "<b>Profile:</b> docs/talk/validate/pairs/block_WITHOUT_kernel_8.98ms (S5a)/s5a_lingbot_block_sdpa_8.98ms.neff + .ntff (rank 1 of 32, TP4 x CP8, the steady block, 1,600 calls per pass), source pairs/block_WITHOUT_kernel_8.98ms (S5a)/source.tar.gz (unpacks to /tmp/lingbot, the path in the NEFF). Job talk-sdpa-math-job.yaml (kiro/e2e/profile.sh)."])

pair("S5b", "Wan2.2 Explorer screenshot",
     [("img", timeline("iv_demo_sdpa.json", 726, 478, "every instruction of F.scaled_dot_product_attention")),
      ("r", "red from ~5 to ~34 ms", "red from 0 to 2.7 ms"),
      ("r", "short red bursts with gaps", "idle from 0.71 to 1.88 ms (dashed)")],
     ["screenshot → timeline drawn from pairs/demo_sdpa (one LingBot rank's attention, one core, demo/sdpa_vs_nki.py)",
      "Vector, Scalar red ~5-34 ms → 0-2.7 ms", "Tensor: one 1.17 ms gap, 0.71-1.88 ms"],
     ["<b>Profile:</b> docs/talk/validate/pairs/attention_only_WITHOUT_kernel_2.69ms (S5b)/s5b_lingbot_attention_sdpa_2.69ms.neff + .ntff (one LingBot rank's self-attention, one trn3 core, q [1,10,640,128], k/v [1,10,17280,128]), source pairs/attention_only_WITHOUT_kernel_2.69ms (S5b)/source.tar.gz. Select the F.scaled_dot_product_attention line of sdpa_vs_nki.py in the code view. Job demo/demo_job.yaml; script demo/sdpa_vs_nki.py --mode sdpa."])

pair("S5d", "Wan2.2: 39% / 36% / 16%",
     [("r", "One native-SDPA block, 56.1 ms, measured from the profile", "One LingBot block, SDPA as matmul → softmax → matmul, 8.98 ms, measured from the profile"),
      ("re", r">39%<", ">45%<"), ("re", r">36%<", ">29%<"), ("re", r">16%<", ">17%<")],
     ["56.1 ms → 8.98 ms", "Tensor busy 39% → 45%", "one engine busy 36% → 29% (28.5%)", "every engine idle 16% → 17% (16.6%)"])

pair("S5f", "Wan2.2: 18.2 → 5.3 ms per call, 7,854 tokens",
     [("svg", 0, hbars([(0, "SELF-ATTENTION, ONE LINGBOT RANK: 640 × 17,280"), (298, "PER CALL")],
                       [dict(label="Native SDPA", v=2.687, text="2.69 ms", bar="bar-x", t="t"),
                        dict(label="nkilib attention_cte", v=1.574, text="1.57 ms", bar="bar-g", t="t-g", step=1)],
                       x0=298, maxw=744, rowh=120))],
     ["7,854 tokens (Wan2.2) → one LingBot rank: 640 queries × 17,280 keys, 10 heads",
      "18.2 → 5.3 ms → 2.69 → 1.57 ms device time per call (demo/sdpa_vs_nki.py, one core)"])

pair("S5g", "Wan2.2: 16.2 → 13.2 s; for LingBot calling it yourself is slower",
     [("code", 0, code_panel("Inside the compiled block", "wan/modules/attention.py", 350,
                             ['from nkilib.core.attention.attention_cte import attention_cte',
                              '_attention_cte = wrap_nki(attention_cte)',
                              'out = _attention_cte(q_c, k_c, v_c, scale=1.0, causal_mask=False,',
                              '                     tp_q=True, tp_k=True, tp_out=False)'], hl=2)),
      ("r", "16.2 → 13.2 <small>s</small>", "8.09 → 11.0 <small>s</small>"),
      ("r", "per video, self- and cross-attention", "per video: slower than native SDPA, which already uses it"),
      ("r", "Reach for nkilib first: production kernels, called like any torch op.",
       "Check what the compiler already does: native SDPA already runs attention_cte here.")],
     ["code: Wan2.2 models/fast_dit.py → LingBot wan/modules/attention.py:350",
      "16.2 → 13.2 s → 8.09 s (native SDPA, the compiler puts in attention_cte) vs 11.0 s (our own call: it copies each KV chunk into a padded buffer, attention.py:449)",
      "takeaway changed: the LingBot result is the opposite"])

pair("S5h", "Wan2.2: 56.1 → 17.3 ms, Explorer screenshot",
     [("r", "The same block with the kernel: 56.1 → 17.3 ms", "The same block with the kernel: 8.98 → 4.37 ms"),
      ("img", timeline("iv_blk_cte.json", 730, 480, "attention_cte instructions")),
      ("r", "<b>Self-attention ~5 ms</b> (was ~32)", "<b>Attention 2.52 ms</b> (was 7.16)"),
      ("r", "<b>No Tensor gaps:</b> the softmax overlaps the matmuls", "<b>Softmax with Tensor idle 0.21 ms</b> (was 2.32)")],
     ["56.1 → 17.3 ms → 8.98 → 4.37 ms (same rank, same shard: pairs/1_sdpa_math_rank1 vs pairs/2_attention_cte_rank1)",
      "screenshot → timeline drawn from the LingBot profile", "self-attention ~32 → ~5 ms → attention span 7.16 → 2.52 ms",
      "\"no Tensor gaps\" → softmax with Tensor idle 2.32 → 0.21 ms (a few short gaps remain)"])

pair("S5i", "Wan2.2 table",
     [("table", '<table class="tbl big"><thead><tr><th></th><th class="r">Native SDPA</th><th class="r">attention_cte</th></tr></thead><tbody>'
                '<tr><td>Block</td><td class="r">8.98 ms</td><td class="r"><b>4.37 ms</b></td></tr>'
                '<tr><td>Attention</td><td class="r">7.16 ms</td><td class="r"><b>2.52 ms</b></td></tr>'
                '<tr><td>Every engine idle</td><td class="r">16.6%</td><td class="r">17.1%</td></tr>'
                '<tr><td>Tensor busy</td><td class="r">45%</td><td class="r"><b>56%</b></td></tr>'
                '</tbody></table>')],
     ["block 56.1 / 17.3 ms → 8.98 / 4.37 ms", "self-attention ~32 / ~5 ms → attention span 7.16 / 2.52 ms",
      "every engine idle 16% / 1.1% → 16.6% / 17.1% (collective waits and transposes stay in LingBot's block)",
      "Tensor busy 39% / 79% → 45% / 56%"])

pair("S5j", "Wan2.2 nkilib conv3d VAE",
     [("r", "The same approach for the VAE convolutions", "The VAE: compile whole blocks, then tune"),
      ("code", 0, code_panel("Whole decoder blocks compiled", "wan/modules/vae2_1_fused.py", 325,
                             ['return torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)'])),
      ("r", "6.0 → 2.6 <small>s</small>", "2.03 → 0.41 <small>s</small>")],
     ["title: conv3d kernels (Wan2.2) → LingBot compiles whole VAE blocks", "code: kernels/vae_conv3d.py → vae2_1_fused.py:325",
      "VAE 6.0 → 2.6 s → 2.03 → 0.41 s (B4h has every step)"])

pair("S7a", "trn2: 18.64 and 25.02 fps",
     [("svg", 0, hbars([(0, "LINGBOT-WORLD v2 · ONE 81-FRAME CLIP, 4 DENOISE STEPS · TRN3")],
                       [dict(label="8 chips, main", v=18.08, text="18.08 fps", bar="bar", t="t-v"),
                        dict(label="8 chips, best flags", v=19.9, text="19.9 fps", bar="bar-g", t="t-g")],
                       x0=300, maxw=700, rowh=76))],
     ["8 chips 18.64 fps (trn2) → 18.08 fps (main, q20) and 19.9 fps (loop_engineering_8dev flags, no chained calls) on trn3",
      "16 chips 25.02 fps (trn2) removed: not measured (over the 8-chip budget), and it needs chained calls, which assert on trn3"])

pair("S7d", "Wan2.2 VAE 9.0 → 6.0 s",
     [("code", 0, code_panel("Sequence parallelism inside the block graph", "wan/modules/model_fast.py", 792,
                             ['g = fc.all_gather_tensor(h[0].t().contiguous(), 0, m)   # token shard -> all tokens',
                              'u = torch.matmul(_w(ffn[0]), g) ...',
                              'r = fc.reduce_scatter_tensor(y2, "sum", 0, m)            # back to the shard'])),
      ("r", "9.0 → 6.0 <small>s</small>", "18.2 → 2.03 <small>s</small>"),
      ("r", "17.88 <small>fps</small>", "18.08 <small>fps</small>")],
     ["code: Wan2.2 fast_dit.py → LingBot model_fast.py:792 (feature-major FFN under SP)",
      "VAE 9.0 → 6.0 s → 18.2 s on one core → 2.03 s with one width tile per core, 32 cores",
      "17.88 fps → 18.08 fps (q20, re-run on trn3)"])

pair("S9a", "VAE 44.6 dB is Wan2.2",
     [("r", "<b>Within the bf16 floor</b>", "<b>0.0108</b>"), ("r", "<b>44.6 dB</b>", "<b>52.7 dB</b>")],
     ["\"within the bf16 floor\" → 0.0108 rel-RMS, step-0 noise prediction vs an fp32 CPU forward (q20)",
      "44.6 dB (Wan2.2) → 52.7 dB, VAE frames vs an fp32 CPU decode of the same latents"])

pair("S9b", "the Wan2.2 journey",
     [("r", "The journey: 76.6 s → 8.7 s per video", "The journey: 21.6 s → 4.07 s per video"),
      ("svg", 0, hbars([(0, "LINGBOT-WORLD v2 · 8 TRN3 CHIPS"), (352, "IDEAS"), (440, "PER VIDEO")],
                       [dict(label="Eager", v=21.62, text="21.6 s", bar="bar-m", t="t-v", chips=["1"]),
                        dict(label="+ torch.compile", v=8.09, text="8.09 s", bar="bar", t="t-v", chips=["2"]),
                        dict(label="+ NKI attention, SP, fused VAE", v=4.48, text="4.48 s", bar="bar", t="t-v", chips=["3", "4"]),
                        dict(label="+ 8dev flags, no chained calls", v=4.07, text="4.07 s", bar="bar-g", t="t-g", chips=["3", "4"])]))],
     ["76.6 / 36.8 / 16.1 / 10.8 / 8.7 s (Wan2.2) → 21.6 / 8.09 / 4.48 / 4.07 s (LingBot)",
      "A100 16.92 s removed: no A100 number for LingBot"])

pair("S9c", "16 and 8 trn2 devices",
     [("r", "LingBot-World v2: 25 fps on 16 devices, 18.4 on 8", "LingBot-World v2: 19.9 fps on 8 trn3 chips"),
      ("svg", 0, hbars([(0, "8 TRN3 CHIPS (32 RANKS)"), (440, "GEN + VAE PER VIDEO"), (1080, "FPS")],
                       [dict(label="Eager", v=21.62, text="21.6 s", bar="bar-m", t="t-v", extra=[(1080, "3.75", "t-v")]),
                        dict(label="+ compiled blocks", v=8.09, text="8.09 s", bar="bar", t="t-v", extra=[(1080, "10.0", "t-v")]),
                        dict(label="+ q20 flags (main)", v=4.48, text="4.48 s", bar="bar", t="t-v", extra=[(1080, "18.1", "t-v")]),
                        dict(label="today's main flags", v=4.68, text="4.68 s", bar="bar", t="t-v", extra=[(1080, "17.3", "t-v")]),
                        dict(label="+ 8dev flags, no chained calls", v=4.07, text="4.07 s", bar="bar-g", t="t-g", extra=[(1080, "19.9", "t-g")])],
                       maxw=560))],
     ["16 trn2 devices table (11.90 → 3.24 s, 25.02 fps) → 8 trn3 chips: 21.6 → 4.07 s, 19.9 fps",
      "8 trn2 devices with the check 4.40 s / 18.40 fps → not measurable on trn3 (the check needs chained calls)"])

pair("S9d", "17.3 fps / 1.07 s used a separate VAE deployment",
     [("r", "half a trn3 · ~0.58 s per chunk", "half a trn3 · 0.554 s per chunk"),
      ("r", ">Fused VAE<", ">Fused VAE (same chips)<"),
      ("r", "17.3 <small>fps</small>", "11.8 <small>fps</small>"), ("r", "~1.07 <small>s</small>", "2.05 <small>s</small>")],
     ["~0.58 s per chunk → 0.554 s (DiT, measured server-side)",
      "fused VAE on its own deployment → on the same 8 chips (a separate VAE needs more than the 8-chip budget)",
      "17.3 fps → 11.8 fps; ~1.07 s → 2.05 s key press to new frames (VAE inline: DiT and VAE run one after the other)"],
     ["With the VAE on other chips the chunk period would be about the DiT's 0.55 s: not measured."])

pair("S11c", "Wan2.2 VAE warm-up 186 / 21 / 3.6 s",
     [("svg", 0, hbars([(0, "WARM-UP RUN"), (128, "GEN TIME")],
                       [dict(label="Run 1", v=151.0, text="151 s", bar="bar-x", t="t"),
                        dict(label="Run 2", v=4.28, text="4.28 s", bar="bar-w", t="t", step=1),
                        dict(label="Run 3", v=4.27, text="4.27 s", bar="bar-g", t="t-g", step=2)],
                       x0=128, maxw=944, rowh=84))],
     ["VAE time 186 / 21 / 3.6 s (Wan2.2) → LingBot GEN time 151 / 4.28 / 4.27 s (today's main flags, 4x8_final)"])

pair("B4h", "64 ranks on trn2",
     [("svg", 0, hbars([(0, "VAE, 32 RANKS, TRN3"), (440, "TIME"), (1010, "VS FP32 DECODE")],
                       [dict(label="Upstream, per conv", v=2.03, text="2.03 s", bar="bar-m", t="t-v", extra=[(1010, "56.3 dB", "t-v")]),
                        dict(label="Whole blocks compiled", v=1.38, text="1.38 s", bar="bar", t="t-v", extra=[(1010, "56.3 dB", "t-v")]),
                        dict(label="bf16 convolutions", v=0.86, text="0.86 s", bar="bar", t="t-v", extra=[(1010, "52.7 dB", "t-v")]),
                        dict(label="-O1, norm as a matmul", v=0.65, text="0.65 s", bar="bar", t="t-v", extra=[(1010, "52.7 dB", "t-v")]),
                        dict(label="2-D tiles", v=0.46, text="0.46 s", bar="bar", t="t-v", extra=[(1010, "52.7 dB", "t-v")]),
                        dict(label="6 frames per step", v=0.41, text="0.41 s", bar="bar-g", t="t-g", extra=[(1010, "52.7 dB", "t-g")])],
                       maxw=480, rowh=56))],
     ["64 ranks trn2 → 32 ranks trn3", "3.01 / 1.36 / 0.80 / 0.55 / 0.33 / 0.31 s → 2.03 / 1.38 / 0.86 / 0.65 / 0.46 / 0.41 s",
      "43.5 → 56.6 dB → 56.3 dB from the start (no padding bug at this tiling); bf16 costs 3.6 dB (56.3 → 52.7)"])

pair("B4i", "64 ranks on trn2",
     [("r", "UPSTREAM DECODER, 64 RANKS", "UPSTREAM DECODER, 32 RANKS"), ("r", "~3 s", "2.03 s"),
      ("r", "~3 → 1.36 <small>s</small>", "2.03 → 1.38 <small>s</small>"), ("r", "0.31 <small>s</small>", "0.41 <small>s</small>")],
     ["64 → 32 ranks", "upstream ~3 s → 2.03 s", "~3 → 1.36 s → 2.03 → 1.38 s", "0.31 s after tuning → 0.41 s",
      "\"convolution: 0.2 s\" kept from the original: not re-measured on trn3"])

pair("B6a", "trn2, 16 devices",
     [("r", "1.7 ms of a 10.4 ms block", "14.4 ms of a 17.1 ms block"),
      ("r", "heads scaled by 0.94–0.99", "error vs fp32 0.032 (vs 0.011 read in place)")],
     ["copy 1.7 ms of a 10.4 ms block (trn2) → 14.4 ms of a 17.1 ms block (trn3, LINGBOT_CTE_MERGE off: lbw-talk-b6-nomerge rank-0 pair rankr0-6681ms, attention.py:440-459); +3.03 s per video (7.37 vs 4.34 s)",
      "heads scaled by 0.94-0.99 → step-0 error vs fp32 0.032 vs 0.011 (the zero keys' effect, measured end to end)"],
     [])

PAIRS.append(with_banner(cell_of(orig, "B6b"), "B6b", "before", "trn2: 8.95 → 5.73 s, 3.28 → 2.88 ms"))
cur = open(os.path.join(REPO, "demo", "talk-deck.html"), encoding="utf-8").read()
PAIRS.append(with_banner(cell_of(cur, "B6b"), "B6b", "after", "LingBot-World v2 · 8 trn3 chips · approved"))

pair("B6c", "trn2 numbers",
     [("r", "The copy took 1.7 of 10.4 ms per block.", "The copy took 14.4 of 17.1 ms per block."),
      ("r", "3.28 → 2.88 ms per block.", "3.11 → 2.85 ms per block."),
      ("r", "Generation 3.94 → 3.66 s per video.", "K layout 4.32 → 4.08 s, V tiles 4.20 → 4.08 s.")],
     ["1.7 of 10.4 ms → 14.4 of 17.1 ms (7.37 → 4.34 s per video)", "3.28 → 2.88 ms per block → 3.11 → 2.85 ms per steady block, system trace (4.34 → 4.08 s per video)",
      "3.94 → 3.66 s → 4.32 → 4.08 s (LINGBOT_K_BLOCKED) and 4.20 → 4.08 s (LINGBOT_CTE_VGROUP=8)"])

NA = ('<p class="dek" style="color:#fab219">Not measurable on trn3: the shortcut needs chained kernel calls, which assert '
      'on trn3 (kernels/attention_cte_lb.py:767). Numbers below are trn2.</p>')
for ref in ("B6d", "B6g", "B6h"):
    pair(ref, "trn2, branch loop_engineering_8dev",
         [("re", r"(</h1>)", r"\1" + NA.replace("\\", "\\\\"))],
         ["added: not measurable on trn3 (chained calls assert, attention_cte_lb.py:767; the shortcut only switches on with chaining, model_fast.py:1389)",
          "numbers unchanged: they stay trn2 results" + (" · the code matches kernels/attention_cmax2_lb.py:389-403" if ref == "B6h" else "")],
         why_after="trn2 only · cannot run on trn3")

pair("B6e", "trn2: 3.94 → 3.74 → 3.66 s",
     [("r", "GEN 3.94 → 3.74 s", "GEN 4.20 → 4.08 s"), ("r", "3.74 → 3.66 s", "4.32 → 4.08 s"),
      ("r", "−250 µs per block", "4.12 → 4.08 s")],
     ["8 V tiles per DMA: 3.94 → 3.74 s → 4.20 → 4.08 s", "K in the all-gather's layout: 3.74 → 3.66 s → 4.32 → 4.08 s",
      "cross-attention cache layout: −250 µs per block → 4.12 → 4.08 s per video",
      "each is q20 with that one flag off vs all on, 8 trn3 chips", "runtime-built K/V DMA descriptors (LINGBOT_CTE_KV_DGE=none): 4.67 vs 4.08 s, slower (BEFORE notes: 4.26 vs 3.94 s)"])

pair("B6f", "trn2, TP4 × CP16, 2.5 ms",
     [("r", "Backup: one block now, 2.5 ms", "Backup: one block now, 2.74 ms"),
      ("r", "TP4 × CP16, LingBot-World v2", "TP4 × CP8, LingBot-World v2, 19.9 fps flags, 8 trn3 chips"),
      ("r", '~350 <small>µs</small></div><div class="l">idle, waiting for K/V all-gathers', '232 <small>µs</small></div><div class="l">every engine idle'),
      ("r", '~310 <small>µs</small></div><div class="l">FFN', '1.02 <small>ms</small></div><div class="l">projections, FFN, norms'),
      ("r", '~150 <small>µs</small></div><div class="l">merging (data layout)', '0.13 <small>ms</small></div><div class="l">attention merge and prep'),
      ("r", '~100 <small>µs</small></div><div class="l">K/V loads', '1.08 <small>ms</small></div><div class="l">attention kernel')],
     ["2.5 ms at TP4 × CP16 (trn2, 270 tokens per rank) → 2.74 ms at TP4 × CP8 (trn3, 540 tokens per rank), the 19.9 fps config (2.71 ms per call in the system trace)",
      "the four parts re-measured on lbw-talk-b6f-8dev/lnc2_4x8_d8nochain (rank 0 pair); they are busy times and overlap. q20 block: 2.92 ms"])

pair("B7c", "Wan2.2 layouts on one trn2 device",
     [("re", r"<h1>[^<]*</h1>", "<h1>Backup: three layouts on 8 trn3 chips</h1>"),
      ("svg", 0, hbars([(0, "LAYOUT, LINGBOT, 8 TRN3 CHIPS"), (440, "PER VIDEO"), (1080, "FPS")],
                       [dict(label="TP2 × CP16", v=5.46, text="5.46 s", bar="bar-m", t="t-v", extra=[(1080, "14.9", "t-v")]),
                        dict(label="TP8 × CP4 (TP across chips)", v=4.81, text="4.81 s", bar="bar", t="t-v", extra=[(1080, "16.8", "t-v")]),
                        dict(label="TP4 × CP8 (TP inside a chip)", v=4.50, text="4.50 s", bar="bar-g", t="t-g", extra=[(1080, "18.0", "t-g")])],
                       maxw=560))],
     ["four Wan2.2 layouts inside one trn2 device (29.4 → 20.6 ms per block) → three LingBot layouts on 8 trn3 chips, per video",
      "per-block column removed: not measured per layout"])

title = ('<div class="cell" data-ref="T0"><section class="slide title-slide"><h1>LingBot-World v2 on Trainium3: '
         'the validated numbers</h1><p class="byline kind">Every slide twice: <b style="color:#ff6b6b">BEFORE</b> '
         '(the original, Wan2.2 or trn2) then <b style="color:#4cd04c">AFTER</b> (the same slide, the same point, '
         'LingBot numbers measured on 8 trn3 chips). Delete the BEFORE slides once reviewed.</p></section>'
         '<div class="nt"><p>Built by docs/talk/validate/build_trn3_deck.py. Each AFTER slide\'s notes list what changed.</p></div></div>\n')
cells = title + "\n".join(PAIRS)
out = PREFIX.replace("</style>", CSS + "</style>", 1).replace("<title>", "<title>[trn3 BEFORE/AFTER] ", 1) + cells + SUFFIX
open(OUT, "w", encoding="utf-8").write(out)
print("wrote", OUT, "slides:", cells.count('<div class="cell"'))
