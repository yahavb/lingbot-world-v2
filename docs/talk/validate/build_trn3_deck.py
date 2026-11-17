#!/usr/bin/env python3
"""Build demo/talk-deck-trn3.html (changed slides only) and demo/talk-deck.html (the full talk, merged).

BEFORE = the slide exactly as in the original deck (~/Downloads/talk-deck_v2.html), red banner,
         data-ref "<id>-BEFORE". Delete these after the review.
AFTER  = THE SAME SLIDE (same title, layout, figure, point), with only the Wan2.2 / trn2 numbers,
         file names and code replaced by the LingBot-World v2 ones measured on 8 trn3 chips
         (docs/talk/validate/RESULTS.md). Green banner, data-ref "<id>-AFTER". Its notes list every
         change, old -> new. Every edit is asserted: the build fails if the original text moved.
B6b AFTER is the approved two-lane animation (demo/talk-deck.html).

  python3 docs/talk/validate/build_trn3_deck.py
"""
import base64
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
.slide a{pointer-events:auto;cursor:pointer}
.xlink{color:#4da3ff;text-decoration:underline;font-weight:700}
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


def kernel_side_by_side():
    """One slide right after S5a-AFTER: the same LingBot block WITH the kernel (S5a-AFTER shows it without)."""
    img = base64.b64encode(open(os.path.join(HERE, "anim", "S5h_kernel_marked.jpg"), "rb").read()).decode()
    body = ('<div style="display:flex;flex-direction:column;gap:10px;align-items:center;width:100%%">'
            '<div style="font-size:30px;font-weight:800;color:#4cd04c">WITH the kernel: 4.37 ms</div>'
            '<a href="%s" target="_blank" rel="noopener"><img src="data:image/jpeg;base64,%s" style="width:760px;height:auto;border-radius:8px"></a>'
            '<a class="xlink" href="%s" target="_blank" rel="noopener">Open this profile in Neuron Explorer</a>'
            '<div style="font-size:21px;color:#c3c2b7">green box: no waiting, the softmax runs alongside the matmuls</div></div>' % (S5H_URL, img, S5H_URL))
    cell = ('<div class="cell" data-ref="S5a-KERNEL"><section class="slide"><p class="kicker">Idea 3 · NKI · Keep the engines busy</p>'
            '<h1>The same LingBot block with the kernel</h1><div class="body">%s</div>'
            '<p class="takeaway"><span class="k">Takeaway</span><span>2.1× faster: the kernel overlaps the softmax with the matmuls, so Tensor stops waiting.</span></p>'
            '</section><div class="nt">%s</div></div>\n') % (body, notes(
                ["new slide, right after S5a-AFTER (which is labelled WITHOUT the kernel: 8.98 ms), so the fix sits next to the problem"],
                ["pairs/block_WITH_kernel_4.37ms (S5h). Same rank, same shard, same block as S5a; only the attention changes."]))
    return with_banner(cell, "S5a-KERNEL", "after", "LingBot-World v2 · 8 trn3 chips").replace('data-ref="S5a-KERNEL-AFTER"', 'data-ref="S5a-KERNEL"')


def _b6b_one(ref, h1, color, label, jpg, alt, stats, takeaway, note, url=None):
    img = base64.b64encode(open(os.path.join(HERE, "anim", jpg), "rb").read()).decode()
    st = "".join('<div class="stat"><div class="v" style="color:%s">%s <small>%s</small></div><div class="l">%s</div></div>' % (c, v, u, l) for c, v, u, l in stats)
    body = ('<div style="display:flex;gap:36px;align-items:center;width:100%%">'
            '<img src="data:image/jpeg;base64,%s" alt="%s" style="max-height:455px;max-width:780px;width:auto;height:auto;border-radius:8px">'
            '<div style="display:flex;flex-direction:column;gap:20px">'
            '<div style="font-size:28px;font-weight:800;color:%s">%s</div>%s%s</div></div>') % (img, alt, color, label, st,
            ('<p><a class="xlink" href="%s" target="_blank" rel="noopener">Open this profile in Neuron Explorer</a></p>' % url) if url else "")
    if url:
        body = body.replace('<img src=', '<a href="%s" target="_blank" rel="noopener"><img src=' % url, 1).replace('border-radius:8px">', 'border-radius:8px"></a>', 1)
    cell = ('<div class="cell" data-ref="%s"><section class="slide"><h1>%s</h1><div class="body">%s</div>'
            '<p class="takeaway"><span class="k">Takeaway</span><span>%s</span></p></section><div class="nt">%s</div></div>\n'
            % (ref, h1, body, takeaway, notes(["new slide after B6b-AFTER"], [note])))
    return with_banner(cell, ref, "after", "LingBot-World v2 · 8 trn3 chips").replace('data-ref="%s-AFTER"' % ref, 'data-ref="%s"' % ref)


def b6b_profile():
    return (_b6b_one("B6b-COPY", "Backup: the copy, in the profile", "#ff6b6b", "COPY FIRST: 17.1 ms block", "B6b_line448_marked.jpg",
                     "copy-first block: line 448 selected, its copy is the red bar on Sync",
                     [("#ff6b6b", "7.4", "ms", "line 448 alone: copies of K on the Sync engine"),
                      ("#ff6b6b", "14.4", "of 17.1 ms", "copying K and V (lines 448-449)"),
                      ("#ffffff", "~0", "compute", "Tensor, Vector, Scalar idle meanwhile")],
                     "While the memory is copied, the chip computes nothing.",
                     "Profile: pairs/copy_first_17.1ms (B6b), job lbw-talk-b6-nomerge (LINGBOT_CTE_MERGE=0), rank 0. Line 448/449 of wan/modules/attention.py "
                     "become HLO scatter ops, run on Sync as DMA_DIRECT2D + EVENT_SEMAPHORE.",
                     url="https://neuron-explorer.harmony.a2z.com/profile/copy_first_17.1ms_B6b?shared=H4sIAN0c-WoAA6tWSs4vqIxPyywqLok3NNczzC2OdzJLUrKqViouSSwqCcnMTVWyMtBRSs1LgbANzQ0sDcxMDGprASegFG48AAAA")
            + _b6b_one("B6b-INPLACE", "Backup: read in place, in the profile", "#4cd04c", "READ IN PLACE: 2.92 ms block", "B6b_line210_marked.jpg",
                       "read-in-place block: line 210 selected, short loads on GpSimd, compute busy",
                       [("#4cd04c", "2.92", "ms", "the same block, no copy"),
                        ("#ff6b6b", "~0.25", "ms", "line 210: K and V read in place, short loads on GpSimd"),
                        ("#ffffff", "busy", "", "Tensor, Vector, Scalar keep working")],
                       "Read the memory where it is, and the chip keeps working.",
                       "Profile: pairs/read_in_place_2.92ms (B6b), job lbw-talk-b6-layouts (q20, LINGBOT_CTE_MERGE=1), rank 0, the steady block. "
                       "The read in place: kernels/attention_cte_lb.py:210 (_kv_dma, called for K at 3090 and V at 3207), on the GpSimd engine, ~0.25 ms.",
                       url="https://neuron-explorer.harmony.a2z.com/profile/read_in_place_2.92ms_B6b?shared=H4sIANcZ-WoAA33MsQ6CMBSF4XfpTAiWtlhGJ50cxDjeFLjRhnJr4C7G9N1t4sDGdobz_V-x6hcET88-MvQhDhM4ZiT2kWBgBFXWzbyKNj_ZLdz5GUVbFQJp_G9VN7LSKhVig5HCBx6X7ny9dzDhQhhAlsbOK9x0vxOT5mh0ZXNsQTeCJ3gHN2DGVmZ8MrvYHpS2NqUfWhj5KNYAAAA"))


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
S5F_URL = "https://neuron-explorer.harmony.a2z.com/profile/attention_only_WITH_kernel_1.57ms_S5f?shared=H4sIABYU-WoAA33MvQqDMBAA4He5WYJRLj8-QTt1aErHENsUxHgBvaVI3r1g6VLE7Zu-FfqUH6O_n93pcnN-jDPF5I2wZlr8FQN0KywcZnbDFKGrK4j0_NpYjRbbUkFgjsRDJp8pvf-zRii7Zf1B1iijsLb72W-SAvU2vQ4mibpV0pTyAVkbucjcAAAA"
S5B_URL = "https://neuron-explorer.harmony.a2z.com/profile/attention_only_WITHOUT_kernel_2.69ms_S5b?shared=H4sIAKwS-WoAA6tWSsrJT86OD_cM8fAPDYnPTi3KS82Jt9CztMgtjg82TVSyqlYqLkksKgnJzE1VsjLQUUrNS4GwLSzNTS1NjWt1lBJLSlLzSjLz8-Lz83Iq0Q0z0jOzBBuWhMcwIzMLM1MDy9paACJ_RF6SAAAA"
S5H_URL = "https://neuron-explorer.harmony.a2z.com/profile/s5h_lingbot_block_attention_cte_4.37ms"
S5A_URL = "https://neuron-explorer.harmony.a2z.com/profile/block_WITHOUT_kernel_8.98ms_S5a?shared=H4sIAFoF-WoAA6tWSsrJT86OD_cM8fAPDYnPTi3KS82Jt9CztMgtjg82TVSyqlYqLkksKgnJzE1VsjLQUUrNS4GwLSzNTS1NjWtrAdB7B0lFAAAA"


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
      "This also explains perf/core_order.py: each CP8 group must sit on chips that are all linked."])

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
                '<tr class="win"><td><b>+ bf16, each core decodes one tile, 6 frames per call</b></td><td class="r"><b>0.41 s</b></td><td class="r">52.7 dB</td></tr>'
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
      ("img", '<div><div style="font-size:30px;font-weight:800;color:#ff6b6b;margin-bottom:8px">WITHOUT the kernel: 8.98 ms</div><a href="%s" target="_blank" rel="noopener"><img src="data:image/jpeg;base64,%s" alt="LingBot block without the kernel in Neuron Explorer, attention.py:802 selected; five Tensor gaps marked" style="width:726px;height:auto"></a></div>'
              % (S5A_URL, base64.b64encode(open(os.path.join(HERE, "anim", "S5a_code_view_marked.jpg"), "rb").read()).decode())),
      ("r", "<b>Three long Tensor gaps:</b> ~22 of 56 ms", "<b>Five Tensor gaps:</b> ~0.3 ms each, 1.46 of 8.98 ms (yellow)"),
      ("r", "<b>Three times,</b> once per head", "<b>All inside attention:</b> 2.32 ms of softmax with Tensor idle"),
      ("r", "<b>All inside attention:</b> 2.32 ms of softmax with Tensor idle</p></div>",
       '<b>All inside attention:</b> 2.32 ms of softmax with Tensor idle</p></div><div class="o"><span class="n">↗</span><p><a class="xlink" href="%s" target="_blank" rel="noopener">Open this profile in Neuron Explorer</a></p></div>' % S5A_URL)],
     ["screenshot of the Wan2.2 block → Neuron Explorer screenshot of the LingBot block (pairs/block_WITHOUT_kernel_8.98ms (S5a)), attention.py:802 selected in the code view, the five gaps marked (anim/S5a_code_view_marked.png)",
      "56.1 ms → 8.98 ms; three gaps ~22 ms → five softmax gaps of ~0.3 ms, 1.46 ms in all (1.37-1.67, 2.52-2.82, 3.67-3.97, 4.84-5.12, 6.34-6.61 ms from the block start; Vector 93-99% and Scalar 87-97% busy in them, all from attention.py:802). A sixth 85 us gap at 7.83 ms is not softmax.", "\"three times, once per head\" → 2.32 ms softmax with Tensor idle"],
     ["TORCH_NEURONX_ENABLE_NKI_SDPA=0, otherwise torch-neuronx (SDK 2.33) swaps in an NKI kernel by itself.",
      "<b>Profile:</b> docs/talk/validate/pairs/block_WITHOUT_kernel_8.98ms (S5a)/s5a_lingbot_block_sdpa_8.98ms.neff + .ntff (rank 1 of 32, TP4 x CP8, the steady block, 1,600 calls per pass), source pairs/block_WITHOUT_kernel_8.98ms (S5a)/source.tar.gz (unpacks to /tmp/lingbot, the path in the NEFF). Job talk-sdpa-math-job.yaml (perf/e2e/profile.sh)."])

PAIRS.append(kernel_side_by_side())

pair("S5b", "Wan2.2 screenshot · no AFTER: merged into S5a-AFTER",
     [("img", '<a href="%s" target="_blank" rel="noopener"><img src="data:image/jpeg;base64,%s" alt="Neuron Explorer: F.scaled_dot_product_attention selected in the code view; every instruction it made is red" style="width:726px;height:auto"></a>'
              % (S5A_URL, base64.b64encode(open(os.path.join(HERE, "anim", "S5a_code_view_marked.jpg"), "rb").read()).decode())),
      ("r", "red from ~5 to ~34 ms", "red through the whole attention, ~1 to 7 ms"),
      ("r", "short red bursts with gaps", "red bursts with 5 gaps (yellow)")],
     ["screenshot → your Neuron Explorer screenshot of the LingBot block (pairs/block_WITHOUT_kernel_8.98ms (S5a)), line 802 selected in the code view (blue box)",
      "Vector, Scalar red ~5-34 ms → red through the attention, ~1-7 ms", "Tensor: red bursts with 5 gaps (yellow boxes)"],
     ["<b>Profile:</b> docs/talk/validate/pairs/attention_only_WITHOUT_kernel_2.69ms (S5b)/s5b_lingbot_attention_sdpa_2.69ms.neff + .ntff (one LingBot rank's self-attention, one trn3 core, q [1,10,640,128], k/v [1,10,17280,128]), source pairs/attention_only_WITHOUT_kernel_2.69ms (S5b)/source.tar.gz. Select the F.scaled_dot_product_attention line of sdpa_vs_nki.py in the code view. Job demo/demo_job.yaml; script demo/sdpa_vs_nki.py --mode sdpa."])

PAIRS.pop()   # S5b-AFTER removed: S5a-AFTER's one screenshot already shows "select one call, see every instruction"

pair("S5d", "Wan2.2 numbers · no AFTER: same point as S5a",
     [("r", "One native-SDPA block, 56.1 ms, measured from the profile", "One LingBot block, SDPA as matmul → softmax → matmul, 8.98 ms, measured from the profile"),
      ("re", r">39%<", ">45%<"), ("re", r">36%<", ">29%<"), ("re", r">16%<", ">17%<")],
     ["56.1 ms → 8.98 ms", "Tensor busy 39% → 45%", "one engine busy 36% → 29% (28.5%)", "every engine idle 16% → 17% (16.6%)"])

PAIRS.pop()   # S5d-AFTER removed: it repeats S5a in numbers

pair("S5f", "Wan2.2: 18.2 → 5.3 ms per call, 7,854 tokens",
     [("svg", 0, '<div style="display:flex;gap:32px;justify-content:center;align-items:flex-start;width:100%%">'
                 '<div style="display:flex;flex-direction:column;gap:8px;width:560px">'
                 '<div style="font-size:28px;font-weight:800;color:#ff6b6b">WITHOUT the kernel: 2.69 ms</div>'
                 '<a href="%s" target="_blank" rel="noopener"><img src="data:image/jpeg;base64,%s" alt="attention alone without the kernel: one 1.17 ms Tensor pause" style="width:560px;height:auto;border-radius:8px"></a>'
                 '<div style="font-size:19px;color:#c3c2b7">the matmul engine stops for 1.17 ms · <a class="xlink" href="%s" target="_blank" rel="noopener">open in Neuron Explorer</a></div></div>'
                 '<div style="display:flex;flex-direction:column;gap:8px;width:560px" data-step="1">'
                 '<div style="font-size:28px;font-weight:800;color:#4cd04c">WITH the kernel: 1.57 ms</div>'
                 '<a href="%s" target="_blank" rel="noopener"><img src="data:image/jpeg;base64,%s" alt="attention alone with the NKI kernel: Tensor never stops for the softmax" style="width:560px;height:auto;border-radius:8px"></a>'
                 '<div style="font-size:19px;color:#c3c2b7">the matmuls never stop for the softmax · <a class="xlink" href="%s" target="_blank" rel="noopener">open in Neuron Explorer</a></div>'
                 '<div style="font-size:14px;color:#898781">attention alone, one call over 17,280 keys: a simplified test, not the block\'s exact attention</div></div>'
                 '</div>'
                 % (S5B_URL, base64.b64encode(open(os.path.join(HERE, "anim", "S5f_without_code_marked.jpg"), "rb").read()).decode(), S5B_URL,
                    S5F_URL, base64.b64encode(open(os.path.join(HERE, "anim", "S5f_with_kernel_marked.jpg"), "rb").read()).decode(), S5F_URL))],
     ["chart → your Neuron Explorer screenshot of attention alone without the kernel (line 55 selected), linked to the shared profile; next to it the with-kernel screenshot (attention_only_WITH_kernel_1.57ms (S5f))",
      "7,854 tokens (Wan2.2) → one LingBot rank: 640 queries × 17,280 keys, 10 heads",
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
      ("img", '<img src="data:image/jpeg;base64,%s" alt="LingBot block with the attention kernel in Neuron Explorer; the kernel region and the remaining pause marked" style="width:730px;height:auto">'
              % base64.b64encode(open(os.path.join(HERE, "anim", "S5h_kernel_marked.jpg"), "rb").read()).decode()),
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
                        dict(label="tile per core", v=0.46, text="0.46 s", bar="bar", t="t-v", extra=[(1010, "52.7 dB", "t-v")]),
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
# The approved B6b animation, saved from demo/talk-deck.html before that file became this script's output.
cur = open(os.path.join(HERE, "B6b_approved.html"), encoding="utf-8").read()
PAIRS.append(with_banner(cell_of(cur, "B6b"), "B6b", "after", "LingBot-World v2 · 8 trn3 chips · approved"))
PAIRS.append(b6b_profile())

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

# ═══════════════════ the full talk: original deck, each validated slide replaced by its AFTER ═══════════════════
# BEFORE slides are dropped. A BEFORE with no AFTER (S5b, S5d) is dropped from the talk too: its point
# is merged into S5a. New slides (S5a-KERNEL, B6b-COPY, B6b-INPLACE) follow the AFTER they belong to.
FULL = os.path.join(REPO, "demo", "talk-deck.html")   # the full talk
CELL = re.compile(r'<div class="cell" data-ref="([^"]+)"')


def split_cells(text):
    pos = [m.start() for m in CELL.finditer(text)]
    return [(CELL.match(text, p).group(1), text[p:pos[i + 1] if i + 1 < len(pos) else len(text)].rstrip() + "\n")
            for i, p in enumerate(pos)]


def unbanner(cell, ref):
    cell = re.sub(r'<div class="bab (?:before|after)">.*?</small></div>', "", cell, count=1, flags=re.S)
    return cell.replace('data-ref="%s-AFTER"' % ref, 'data-ref="%s"' % ref, 1)


title = ('<div class="cell" data-ref="T0"><section class="slide title-slide"><h1>LingBot-World v2 on Trainium3: '
         'the changed slides</h1><p class="byline kind">Only the slides that changed, with LingBot numbers measured '
         'on 8 trn3 chips. The full talk with these merged in: demo/talk-deck.html.</p></section>'
         '<div class="nt"><p>Built by docs/talk/validate/build_trn3_deck.py. Each AFTER slide\'s notes list what changed.</p></div></div>\n')
cells = title + "".join(c for r, c in split_cells("\n".join(PAIRS)) if not r.endswith("-BEFORE"))
out = PREFIX.replace("</style>", CSS + "</style>", 1).replace("<title>", "<title>[trn3 changed slides] ", 1) + cells + SUFFIX
open(OUT, "w", encoding="utf-8").write(out)
print("wrote", OUT, "slides:", cells.count('<div class="cell"'))


after, extras, dropped, last = {}, {}, set(), None
for ref, cell in split_cells("\n".join(PAIRS)):
    if ref.endswith("-BEFORE"):
        dropped.add(ref[:-7])
    elif ref.endswith("-AFTER"):
        last = ref[:-6]
        after[last] = unbanner(cell, last)
    else:
        extras.setdefault(last, []).append(unbanner(cell, ref))

# S4f, S4g: the Wan2.2 VAE "compiled is slower" example; for LingBot compiling wins (S4e). Dropped.
dropped.update(("S4f", "S4g"))   # S4g: the same Wan2.2 example (12% vs 75% of peak)
full_cells, kept = [], 0
for ref, cell in split_cells(orig[first:end_cells]):
    if ref in after:
        full_cells.append(after[ref])
        full_cells.extend(extras.get(ref, []))
    elif ref in dropped:
        continue
    else:
        full_cells.append(cell)
        kept += 1
assert not set(after) - {r for r, _ in split_cells(orig[first:end_cells])}, "AFTER without an original slide"

# ───────────── S7b: right panel redrawn as the trn3 node (Neuron docs, trn3-arch) ─────────────
# One Trn3 server: 16 Trainium3 chips. Chips are paired ([0]-[1] ... [14]-[15]) by a direct NeuronLink-v4
# link, 1,024 GB/s; every chip also connects to a NeuronSwitch-v1, 1,024 GB/s. No 2-D torus on trn3.
# The left panel (one chip, 8 NeuronCore-v4) already matches the Trainium3 page.
def s7b_node_svg():
    out = ['<text class="lbl" x="584" y="12">SEQUENCE PARALLEL · ACROSS CHIPS</text>',
           '<text class="t" x="867" y="44" text-anchor="middle">one Trn3 server · 16 Trainium3 chips</text>']
    xs = []
    for pr in range(4):
        x0 = 584 + pr * 148
        xs += [x0, x0 + 66]
    sw_y, sw_h = 152, 30
    for row, y in ((0, 70), (1, 220)):
        for k, x in enumerate(xs):
            used = row == 0
            out.append('<rect class="%s" x="%d" y="%d" width="56" height="44" rx="6"/>' % ("bx" if used else "bx", x, y))
            if used:
                out.append('<rect class="bx-a" x="%d" y="%d" width="20" height="20" rx="4"/>' % (x + 18, y + 12))
            # to the switch
            out.append('<line class="br" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (x + 28, y + 44 if row == 0 else y, x + 28, sw_y if row == 0 else sw_y + sw_h))
        for k in range(0, 8, 2):   # the direct pair link
            out.append('<line stroke="#199e70" stroke-width="3" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (xs[k] + 56, y + 22, xs[k + 1], y + 22))
    out.append('<rect class="bx-h" x="584" y="%d" width="566" height="%d" rx="6"/>' % (sw_y, sw_h))
    out.append('<text class="t-m" x="867" y="%d" text-anchor="middle">NeuronSwitch-v1</text>' % (sw_y + 20))
    out.append('<text class="t-s" x="584" y="292"><tspan fill="#199e70">━</tspan> paired chips: direct NeuronLink-v4 · '
               '│ every chip: the switch · 1,024 GB/s each</text>')
    out.append('<text class="t-s" x="584" y="312">highlighted: the 8 chips of one video (TP4 × CP8)</text>')
    out.append('<text class="t-s" x="584" y="336">the K/V all-gather crosses NeuronLink — the slower link</text>')
    return "\n    ".join(out)


def s7b(cell):
    svg = re.search(r"<svg\b.*?</svg>", cell, re.S).group(0)
    k = svg.index('<text class="tk" x="480" y="190" text-anchor="middle">×</text>')
    head = svg[:k + len('<text class="tk" x="480" y="190" text-anchor="middle">×</text>')]
    head = head.replace("Right: sixteen chips in a torus, one core highlighted in each, hold the sequence-parallel group, "
                        "whose key and value all-gather crosses NeuronLink between chips.",
                        "Right: one Trn3 server, sixteen Trainium3 chips, paired by direct NeuronLink links and all joined by "
                        "a NeuronSwitch; eight highlighted chips hold the sequence-parallel group, whose key and value "
                        "all-gather crosses NeuronLink between chips.")
    head = head.replace('<text class="lbl" x="580" y="12">SEQUENCE PARALLEL · ONE CORE PER CHIP</text>', '')
    assert "sixteen chips in a torus" not in head
    return cell.replace(svg, head + "\n    " + s7b_node_svg() + "\n  </svg>", 1)


full_cells = [s7b(c) if CELL.match(c).group(1) == "S7b" else c for c in full_cells]

# ───────────── S11b / S11d: LingBot's server (serve.py) and deployment, not Wan2.2's serve_ti2v.py ─────────────
GHM = "https://github.com/yahavb/lingbot-world-v2/blob/main/"
S11_OPS = {
    "S11b": [
        ("on all 8 ranks", "on all 32 ranks"),
        ("8 ranks, one torchrun", "32 ranks, one torchrun"),
        ('<b>Fixed shape</b><p>no compile per request</p>',
         '<b>Every graph compiled at start-up</b><p>a warm-up session runs each prompt switch and camera move</p>'),
        ('<b>Collectives on one thread</b><p>the one that compiled</p>',
         '<b>Every rank in lockstep</b><p>rank 0 broadcasts each chunk’s camera and prompt</p>'),
        ('<b>T5 on the CPU</b><p>with a prompt cache</p>',
         '<b>One graph for any prompt</b><p>text padded to one length, embeddings cached</p>'),
        ("Fix the input shapes, so no request ever triggers a compile.",
         "Compile every graph at start-up, so no request waits for the compiler."),
    ],
    "S11d": [
        ("Close the host gaps, then deploy like any pod", "Stream every chunk, then deploy like any pod"),
        ('<b>Scheduler step on the device:</b> busy 96.9% → 99.5%',
         '<b>VAE in its own pod:</b> decodes a chunk while the DiT makes the next'),
        ('<b>Encode the mp4</b> while the VAE decodes', '<b>Frames stream</b> to the browser as each one decodes'),
        ('<b>ffmpeg with vfork:</b> no 90 ms stall', '<b>Controls apply next chunk:</b> about 1.1 s to see a key press'),
        ('NeuronCore claim<br>weights cached on a PVC<br>readiness probes',
         'Neuron devices by resource claim<br>weights and graphs cached on the node<br>startup and readiness probes'),
        ("Find host gaps in the system trace, and move that work off the device’s path.",
         "Stream each chunk as it is ready, so a key press shows on screen in about a second."),
    ],
}
S11_NOTES = {
    "S11b": ('<p class="meta">Section 8 · about 2 minutes · 5 clicks</p>'
             '<p><a href="%sserve.py">serve.py</a> runs FastAPI on rank 0 (POST /session, /control, /health, /readiness), with all '
             '32 ranks (TP4 × CP8 on 8 chips) in one torchrun.</p><ol>'
             '<li>Every graph compiled at start-up: a warm-up session runs three segments, switching the prompt at both '
             'boundaries and stepping through every camera action, so each graph a real session uses compiles before the pod is ready.</li>'
             '<li>Every rank in lockstep: rank 0 decides each chunk’s camera move and prompt and broadcasts it, so all ranks '
             'switch at the same chunk.</li>'
             '<li>One graph for any prompt: T5 embeddings are zero-padded to the text length, so a new prompt length is not a new '
             'graph, and cached by prompt (the scene prompts are encoded at start-up).</li>'
             '<li>Health before readiness: /health answers while the model loads; /readiness turns true only after warm-up.</li></ol>'
             '<p>Click 5, the takeaway.</p>' % GHM),
    "S11d": ('<p class="meta">Section 8 · about 2 minutes · 2 clicks</p><ol>'
             '<li>The VAE runs in its own pod (<a href="%slbw-v2-vae-deploy.yaml">lbw-v2-vae-deploy.yaml</a>). Rank 0 sends each '
             'chunk’s latents to it from a background thread, off the DiT’s path (RemoteDecoder in serve.py).</li>'
             '<li>Each decoded frame goes to the browser as a JPEG over server-sent events.</li>'
             '<li>A camera key or a prompt applies from the next chunk: about 1.1 s from the key press to its frames at the browser, '
             'at 17.3 fps (<a href="%slbw-v2-dit-t5-deploy.yaml">lbw-v2-dit-t5-deploy.yaml</a> header).</li></ol>'
             '<p>Click 1: deploy on EKS: a DRA resource claim for the Neuron devices, the checkpoint and compiled graphs cached on the node, '
             'startup and readiness probes.</p><p>Click 2, the takeaway.</p>' % (GHM, GHM)),
}


def s11(cell):
    ref = CELL.match(cell).group(1)
    for a, b in S11_OPS[ref]:
        assert a in cell, (ref, a[:50])
        cell = cell.replace(a, b)
    k = cell.index('<div class="nt">') + len('<div class="nt">')
    return cell[:k] + S11_NOTES[ref] + "</div></div>\n"


full_cells = [s11(c) if CELL.match(c).group(1) in S11_OPS else c for c in full_cells]

# ───────────── S11c removed; S11m: S1c's parts, each mapped to the EKS object that runs it ─────────────
# The objects are the live KEEP-trn3-dev1 deployment: dev1-ing.yaml, lbw-web, lbw-v2-dit-t5, lbw-v2-vae.
def s11m():
    cols = [("you", "prompt and camera keys", "Ingress → ALB", "AWS Load Balancer Controller", "path /lbw"),
            ("web app", "page, relays the stream", "Deployment lbw-web", "CPU node", "FastAPI"),
            ("T5 + DiT", "KV cache in HBM", "Deployment lbw-v2-dit-t5", "claim xl-lnc2 · 8 trn3 chips", "32 ranks, TP4 × CP8"),
            ("VAE", "latents → frames", "Deployment lbw-v2-vae", "claim l-lnc1 · 4 trn3 chips", "32 ranks")]
    W, G = 252, 48
    o = ['<text class="lbl" x="0" y="12">THE MODEL (S1c)</text>',
         '<g data-step="1"><text class="lbl" x="0" y="168">ON AMAZON EKS</text></g>']
    for i, (m1, m2, e1, e2, e3) in enumerate(cols):
        x = i * (W + G)
        o.append('<rect class="bx" x="%d" y="26" width="%d" height="84" rx="9"/>' % (x, W))
        o.append('<text class="tk" x="%d" y="62" text-anchor="middle">%s</text>' % (x + W // 2, m1))
        o.append('<text class="t-s" x="%d" y="90" text-anchor="middle">%s</text>' % (x + W // 2, m2))
        g = ['<line class="br" stroke-dasharray="4 4" x1="%d" y1="110" x2="%d" y2="182"/>' % (x + W // 2, x + W // 2),
             '<rect class="bx-a" x="%d" y="182" width="%d" height="118" rx="9"/>' % (x, W),
             '<text class="tk" x="%d" y="218" text-anchor="middle">%s</text>' % (x + W // 2, e1),
             '<text class="t-s" x="%d" y="250" text-anchor="middle">%s</text>' % (x + W // 2, e2),
             '<text class="t-s" x="%d" y="276" text-anchor="middle">%s</text>' % (x + W // 2, e3)]
        if i < 3:
            g.append('<path class="ar" d="M%d,241 L%d,241"/>' % (x + W + 4, x + W + G - 4))
        o.append('<g data-step="1">%s</g>' % "".join(g))
    o.append('<g data-step="1"><text class="t-s" x="0" y="338">each arrow is a Kubernetes Service · '
             'on the trn3 node the Neuron DRA driver hands each claim its chips · weights and compiled graphs stay cached on the node</text></g>')
    svg = ('<svg viewBox="0 0 1152 350" width="1152" height="350" role="img" aria-label="Top: the world model parts from S1c: '
           'you, the web app, T5 and the DiT with its KV cache, the VAE. Below each, the EKS object that runs it: an Ingress and ALB, '
           'the lbw-web Deployment on a CPU node, the lbw-v2-dit-t5 Deployment with a claim for 8 trn3 chips, and the lbw-v2-vae '
           'Deployment with a claim for 4 trn3 chips, joined by Services.">%s</svg>' % "".join(o))
    return ('<div class="cell" data-ref="S11m"><section class="slide"><p class="kicker">Cloud deployment · on Amazon EKS</p>'
            '<h1>The world model, mapped to EKS</h1><div class="body">%s</div>'
            '<p class="takeaway" data-step="2"><span class="k">Takeaway</span><span>Each part of the model is a Deployment, '
            'and a resource claim gives it its Trainium chips.</span></p></section>'
            '<div class="nt"><p class="meta">Section 8 · about 1 minute · 2 clicks</p>'
            '<p>The same parts as S1c, top row. Click 1: the EKS object under each one, as deployed on the demo cluster.</p><ol>'
            '<li>The browser reaches an Ingress; the AWS Load Balancer Controller turns it into an ALB that routes /lbw.</li>'
            '<li>The web app (lbw-web) is a plain Deployment on a CPU node: it serves the page and relays the frame stream.</li>'
            '<li>T5 and the DiT run in one Deployment, lbw-v2-dit-t5: a resource claim (xl-lnc2) for 8 trn3 chips, 32 ranks, '
            'TP4 × CP8. The KV cache lives in its HBM.</li>'
            '<li>The VAE is its own Deployment, lbw-v2-vae: a claim (l-lnc1) for 4 trn3 chips. The DiT sends each chunk’s latents to it.</li></ol>'
            '<p>Each arrow is a Service. The Neuron DRA driver on the trn3 node allocates the chips for each claim; the checkpoint and '
            'compiled graphs are cached on the node, so a restart does not download or recompile. Click 2, the takeaway.</p></div></div>\n'
            % svg)


# S11c removed (user, 2026-11-15); S11m was tried in its place and dropped too. s11m() kept for reference.
full_cells = [c for c in full_cells if CELL.match(c).group(1) != "S11c"]

# ───────────── B6b and its two profile slides move from the backups to right after S7d ─────────────
MOVE = ("B6b", "B6b-COPY", "B6b-INPLACE")
moved = [c for c in full_cells if CELL.match(c).group(1) in MOVE]
assert len(moved) == 3
moved = [re.sub(r"<h1>Backup: (\w)", lambda m: "<h1>" + m.group(1).upper(), c, count=1) for c in moved]
full_cells = [c for c in full_cells if CELL.match(c).group(1) not in MOVE]
i = [CELL.match(c).group(1) for c in full_cells].index("S7d") + 1
# Main slides now, so main-slide refs: S7e, S7f, S7g (typing "7e" jumps there; no "B" in the counter).
for a, b in zip(MOVE, ("S7e", "S7f", "S7g")):
    moved = [c.replace('data-ref="%s"' % a, 'data-ref="%s"' % b, 1) for c in moved]
full_cells[i:i] = moved

# ───────────── S1a removed; S1c takes its place (right after the S1 divider) ─────────────
_refs = [CELL.match(c).group(1) for c in full_cells]
_s1c = full_cells.pop(_refs.index("S1c"))
_refs = [CELL.match(c).group(1) for c in full_cells]
full_cells[_refs.index("S1a")] = _s1c

# ───────────── S5h: the same 4.37 ms profile as S5a-KERNEL, so the same Explorer link ─────────────
def s5h_link(cell):
    k = cell.index('<div class="nt">')
    b, n = cell[:k], cell[k:]
    m = re.search(r"<img\b[^>]*>", b)
    a = ('<div style="display:flex;flex-direction:column"><a href="%s" target="_blank" rel="noopener">%s</a>'
         '<a class="xlink" href="%s" target="_blank" rel="noopener" style="display:block;text-align:center;margin-top:6px">'
         'Open this profile in Neuron Explorer</a></div>') % (S5H_URL, m.group(0), S5H_URL)
    return b[:m.start()] + a + b[m.end():] + n


full_cells = [s5h_link(c) if CELL.match(c).group(1) == "S5h" else c for c in full_cells]

# S5h right after S5a-KERNEL: S5a (without, 8.98 ms) -> S5a-KERNEL (with, 4.37 ms) -> S5h (8.98 -> 4.37 ms)
_refs = [CELL.match(c).group(1) for c in full_cells]
_s5h = full_cells.pop(_refs.index("S5h"))
_refs = [CELL.match(c).group(1) for c in full_cells]
full_cells.insert(_refs.index("S5a-KERNEL") + 1, _s5h)

# ───────────── S5c: the "why it is slow" animation, before S5a; no Wan2.2 numbers left in it ─────────────
def s5c(cell):
    m = re.search(r'srcdoc="([^"]*)"', cell)
    src = html.unescape(m.group(1))
    for a, b in [("  [0, 10, 20, 30, 40, 50, 56].forEach(t => g.fillText(t + ' ms', xOf(t), LANE_Y + 3 * (LANE_H + LANE_GAP) + 12));\n", ""),
                 ("g.fillText('Tensor idle ≈ 21 ms of 56', W / 2, H - 28);",
                  "g.fillText('Without the kernel, Tensor sits idle while the softmax runs', W / 2, H - 28);")]:
        assert a in src, a[:40]
        src = src.replace(a, b)
    cell = cell[:m.start(1)] + html.escape(src, quote=True) + cell[m.end(1):]
    for a, b in [("<h1>Simplified: Tensor waits for the softmax</h1>", "<h1>Without the kernel, the block is slow</h1>"),
                 ('<p class="dek">In the trace, max, exp and sum run tile by tile, so Vector and Scalar overlap. Either way, Tensor waits.</p>',
                  '<p class="dek">Simplified: per head, Tensor computes Q×K, then waits while Vector and Scalar run the softmax.</p>')]:
        assert a in cell, a[:40]
        cell = cell.replace(a, b)
    k = cell.index('<div class="nt">') + len('<div class="nt">')
    return cell[:k] + ('<p class="meta">Section 5 · about 1 minute · the animation plays when the slide opens (R restarts it)</p>'
                       '<p>Why the block without the kernel is slow, simplified. Per attention head, Tensor computes Q×K, then sits '
                       'idle while Vector (row max), Scalar (exp) and Vector (row sum) run the softmax, then multiplies by V. '
                       'After the heads, cross-attention and the FFN keep Tensor busy.</p>'
                       '<p>The animation draws 3 heads and no times; LingBot runs 10 heads per rank. The next slide is the real '
                       'profile: 8.98 ms without the kernel. Then the same block with the kernel: 4.37 ms.</p>') + "</div></div>\n"


_refs = [CELL.match(c).group(1) for c in full_cells]
_c = s5c(full_cells.pop(_refs.index("S5c")))
_refs = [CELL.match(c).group(1) for c in full_cells]
full_cells.insert(_refs.index("S5a"), _c)

# No "idea" wording on the slides (kickers, divider labels, "→ Idea N:" pointers, titles). Notes untouched.
NO_IDEA = [(r'(<p class="kicker">)Idea \d+ · ', r'\1'),
           (r'after more tuning: bf16 convolutions, 2-D tiles', 'after more tuning: bf16 convolutions, one tile per core'),
           (r'\s*<p class="sec-k">Idea</p>', ''),
           (r'→ Ideas? \d+(?: and \d+)?: (\w)', lambda m: '→ ' + m.group(1).upper()),
           (r' — (?:both are|that is) idea \d\.', '.'),
           (r'Four ideas, one worked example', 'One worked example'),
           (r'Backup: four ideas for Trainium performance', 'Backup: Trainium performance'),
           (r'The same four ideas', 'The same techniques'),
           # S9b chart: the IDEAS column (1, 2, 3, 3) goes with the framing
           (r'<text class="lbl" x="352" y="12">IDEAS</text>', ''),
           (r'<rect class="ichip"[^>]*/>', ''),
           (r'<text class="ichip-t"[^>]*>[^<]*</text>', ''),
           (r'<text[^>]*x="366"[^>]*>\d</text>', ''),
           # OI: the to-do about the four-ideas framing is moot once the framing is gone
           (r'\s*<li class="new"><b>Four ideas:</b>.*?</li>', '')]


def no_idea(cell):
    k = cell.index('<div class="nt">')
    b = cell[:k]
    for pat, rep in NO_IDEA:
        b = re.sub(pat, rep, b)
    return b + cell[k:]


full_cells = [no_idea(c) for c in full_cells]
for c in full_cells:
    vis = re.sub(r"data:[^\"')]+", "", c[:c.index('<div class="nt">')])
    assert not re.search(r"\bideas?\b", re.sub(r"<[^>]+>", " ", vis), re.I), CELL.match(c).group(1)

# ───────────── re:Invent template (template/*.jpg, from the RIV26 speaker template) ─────────────
TITLE = "AIM323 - Serving World Model at scale with AWS Trainium and Native PyTorch"


def _jpg(name):
    return "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(HERE, "template", name + ".jpg"), "rb").read()).decode()


# Slide is 1280x720; the 16:10 template images are cover-fit (cropped evenly top and bottom).
# Title: text sits under the re:Invent logo, left of the color blocks. Divider: text sits in the band
# between the white box (left) and the color blocks (right). Agenda: the word "Agenda" is in the image.
RIV_CSS = (".slide.riv{background:#000 center/cover no-repeat}"
           ".slide.riv-title{background-image:url(%s);justify-content:flex-start;padding:340px 0 0 96px}"
           ".slide.riv-title h1{font-size:34px;line-height:1.2;max-width:440px}"
           ".slide.riv-title .byline{max-width:440px}"
           ".slide.riv-divider{background-image:url(%s);justify-content:center;padding:30px 280px 0 150px}"
           ".slide.riv-agenda{background-image:url(%s);justify-content:center;padding:0 80px 40px 520px}"
           ".slide.riv-agenda ol{margin:0;padding-left:28px;font-size:25px;line-height:1.55}"
           ".slide.riv-agenda li span{color:var(--ink-2);font-size:17px;letter-spacing:.12em;text-transform:uppercase;margin-right:12px}"
           ".slide.riv-thanks{background-image:url(%s)}.slide.riv-thanks>*{visibility:hidden}"
           % (_jpg("riv-title"), _jpg("riv-divider"), _jpg("riv-agenda"), _jpg("riv-thanks")))


def riv(cell, kind):
    return re.sub(r'<section class="slide ([^"]*)"', lambda m: '<section class="slide %s riv riv-%s"' % (m.group(1), kind), cell, count=1)


agenda_items = []
for i, c in enumerate(full_cells):
    ref = CELL.match(c).group(1)
    if ref == "S0":
        h1 = re.search(r"<h1>.*?</h1>", c, re.S).group(0)
        spk = ('<div class="riv-spk" style="margin-top:22px;font-size:19px;line-height:1.45;color:#fff">'
               '<b>Lingfan Yu</b><br><span style="color:var(--ink-2)">Senior Applied Scientist</span><br>'
               '<b style="display:inline-block;margin-top:10px">Yahav Biran</b><br><span style="color:var(--ink-2)">Principal Solutions Architect</span></div>')
        c = c.replace(h1, "<h1>%s</h1>" % html.escape(TITLE), 1)
        c = c.replace("</section>", spk + "</section>", 1)
        full_cells[i] = riv(c, "title")
    elif ref == "S12":
        qr = "".join('<div style="display:flex;flex-direction:column;align-items:center;gap:10px">'
                     '<img src="data:image/png;base64,%s" alt="QR code: %s" style="width:170px;height:170px;border-radius:6px">'
                     '<div style="font-size:17px;color:#fff;text-align:center;line-height:1.3">%s</div></div>'
                     % (base64.b64encode(open(os.path.join(HERE, "template", f + ".png"), "rb").read()).decode(), url, cap)
                     for f, url, cap in [("qr-neuron-quickstarts", "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/quickstarts.html",
                                          "Neuron quickstarts"),
                                         ("qr-lingbot-world-v2", "https://github.com/yahavb/lingbot-world-v2", "LingBot-World v2 on Trainium")])
        qr = ('<div class="riv-qr" style="position:absolute;left:760px;top:430px;display:flex;gap:56px;visibility:visible">%s</div>' % qr)
        c = c.replace("</section>", qr + "</section>", 1)
        full_cells[i] = riv(c, "thanks")
    elif 'class="slide divider"' in c:
        full_cells[i] = riv(c, "divider")
        if ref.startswith("S"):
            k = re.search(r'<p class="sec-k">(.*?)</p>', c, re.S)
            agenda_items.append('<li><span>%s</span>%s</li>' % (k.group(1) if k else "", re.search(r"<h1>(.*?)</h1>", c, re.S).group(1)))
agenda = ('<div class="cell" data-ref="AG"><section class="slide riv riv-agenda"><ol>%s</ol></section>'
          '<div class="nt"><p class="meta">Agenda</p><p>The sections, in order. Built from the section slides.</p></div></div>\n'
          % "".join(agenda_items))
full_cells.insert(1, agenda)
full = PREFIX.replace("</style>", CSS + RIV_CSS + "</style>", 1) + "".join(full_cells) + SUFFIX
assert "-BEFORE" not in full and 'class="bab' not in full
open(FULL, "w", encoding="utf-8").write(full)
print("wrote", FULL, "slides:", len(full_cells), "· replaced:", len(after), "· new:", sum(map(len, extras.values())),
      "· removed:", sorted(dropped - set(after)), "· unchanged:", kept)
