"""figures.py: the paper's two first figures as static SVG (docs/disagg/fig/), from measured cells (RESULTS.md).

  fig/fps_vs_chips.svg   LingBot frame rate vs total chips, co-located vs split (DiT + 1-chip VAE), 480x768 and 720p
  fig/architecture.svg   role-disaggregated serving: DiT pool, decoder pool, NIXL hand-off, placement
  fig/fps_vs_chips.csv   the plotted numbers (table view)
Palette: dataviz reference instance, slots 1-2 (validated: CVD dE 24.7, normal 33.6, contrast >= 3:1 on #fcfcfb).
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "disagg", "fig")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FONT = "font-family='Helvetica, Arial, sans-serif'"

SERIES = [  # (name, color, dash, [(total chips, fps)])
    ("Co-located, 480x768", BLUE, "", [(4, 15.54), (8, 22.63), (16, 26.70)]),
    ("Split: DiT + 1-chip VAE, 480x768", ORANGE, "", [(3, 10.07), (5, 17.54), (9, 25.76), (17, 32.14)]),
    ("Co-located, 720p", BLUE, "6 4", [(16, 11.38)]),
    ("Split: DiT + 1-chip VAE, 720p", ORANGE, "6 4", [(9, 8.46), (17, 12.68)]),
]


def fps_chart(series=None, title='LingBot-World v2: frame rate vs chips, one stream', note720=True):
    series = series or SERIES
    W, H, L, R, T, B = 760, 460, 64, 150, 48, 56
    xmax, ymax = 18, 35
    X = lambda c: L + (W - L - R) * c / xmax
    Y = lambda f: H - B - (H - T - B) * f / ymax
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='{L}' y='26' font-size='16' font-weight='600' fill='{INK}'>{title}</text>"]
    for f in range(0, ymax + 1, 5):
        s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(f)}' y2='{Y(f)}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{L - 8}' y='{Y(f) + 4}' font-size='11' text-anchor='end' fill='{INK2}'>{f}</text>")
    for c in (2, 4, 6, 8, 10, 12, 14, 16, 18):
        s.append(f"<text x='{X(c)}' y='{H - B + 18}' font-size='11' text-anchor='middle' fill='{INK2}'>{c}</text>")
    s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(0)}' y2='{Y(0)}' stroke='{INK2}' stroke-width='1'/>")
    s.append(f"<text x='{(L + W - R) / 2}' y='{H - 14}' font-size='12' text-anchor='middle' fill='{INK2}'>total chips (Trainium3)</text>")
    s.append(f"<text x='16' y='{(T + H - B) / 2}' font-size='12' fill='{INK2}' transform='rotate(-90 16 {(T + H - B) / 2})' text-anchor='middle'>frames per second</text>")
    for name, col, dash, pts in series:
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(c):.1f},{Y(f):.1f}" for i, (c, f) in enumerate(pts))
            s.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='2' stroke-dasharray='{dash}' stroke-linejoin='round'/>")
        for c, f in pts:
            s.append(f"<circle cx='{X(c):.1f}' cy='{Y(f):.1f}' r='4.5' fill='{col if not dash else SURF}' stroke='{col}' "
                     f"stroke-width='2'><title>{name}: {c} chips, {f} fps</title></circle>")
    # direct labels at the line ends (text in ink, the mark carries the color)
    s.append(f"<text x='{X(17) + 10}' y='{Y(32.14) + 4}' font-size='12' fill='{INK}'>split 480p 32.1</text>")
    s.append(f"<text x='{X(16) + 10}' y='{Y(26.70) + 4}' font-size='12' fill='{INK}'>co-located 480p 26.7</text>")
    if note720:
        s.append(f"<text x='{X(17) + 10}' y='{Y(12.68) + 4}' font-size='12' fill='{INK}'>split 720p 12.7</text>")
        s.append(f"<text x='{X(16) + 10}' y='{Y(11.38) + 14}' font-size='12' fill='{INK}'>co-located 720p 11.4</text>")
        # co-located 8 chips at 720p: no point (does not fit)
        s.append(f"<text x='{X(8)}' y='{Y(2.2)}' font-size='11' text-anchor='middle' fill='{INK2}'>720p co-located, 8 chips:</text>")
        s.append(f"<text x='{X(8)}' y='{Y(0.6)}' font-size='11' text-anchor='middle' fill='{INK2}'>does not fit in device memory</text>")
    # legend (always present for >= 2 series)
    lx, ly = L + 14, T + 14
    for i, (name, col, dash, _) in enumerate(series):
        y = ly + i * 22
        s.append(f"<line x1='{lx}' x2='{lx + 22}' y1='{y}' y2='{y}' stroke='{col}' stroke-width='2' stroke-dasharray='{dash}'/>")
        s.append(f"<circle cx='{lx + 11}' cy='{y}' r='4' fill='{col if not dash else SURF}' stroke='{col}' stroke-width='2'/>")
        s.append(f"<text x='{lx + 30}' y='{y + 4}' font-size='11' fill='{INK}'>{name}</text>")
    s.append(f"<text x='{lx}' y='{ly + len(series) * 22 + 14}' font-size='10' fill='{INK2}'>Split = DiT on n chips + VAE on 1 chip</text>")
    s.append(f"<text x='{lx}' y='{ly + len(series) * 22 + 28}' font-size='10' fill='{INK2}'>(total n + 1). Same flags both sides.</text>")
    s.append("</svg>")
    return "\n".join(s)


def box(x, y, w, h, title, lines, stroke):
    s = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='#ffffff' stroke='{stroke}' stroke-width='2'/>",
         f"<text x='{x + 12}' y='{y + 22}' font-size='13' font-weight='600' fill='{INK}'>{title}</text>"]
    for i, l in enumerate(lines):
        s.append(f"<text x='{x + 12}' y='{y + 42 + i * 16}' font-size='11' fill='{INK2}'>{l}</text>")
    return s


def arrow(x1, y1, x2, y2, label="", col=INK2, dy=-6):
    s = [f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='2' marker-end='url(#a)'/>"]
    if label:
        s.append(f"<text x='{(x1 + x2) / 2}' y='{(y1 + y2) / 2 + dy}' font-size='11' text-anchor='middle' fill='{INK}'>{label}</text>")
    return s


def architecture():
    W, H = 980, 520
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
         f"<path d='M0,0 L10,5 L0,10 z' fill='{INK2}'/></marker></defs>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='30' font-size='16' font-weight='600' fill='{INK}'>Role-disaggregated serving on Amazon EKS + Trainium</text>"]
    s += box(24, 200, 150, 90, "Clients", ["sessions: prompt, scene,", "actions (keys / robot)", "frames back (HTTP)"], INK2)
    # DiT pool
    s.append(f"<rect x='220' y='60' width='300' height='400' rx='12' fill='none' stroke='{BLUE}' stroke-width='2' stroke-dasharray='5 4'/>")
    s.append(f"<text x='236' y='84' font-size='13' font-weight='600' fill='{INK}'>Denoiser role (DiT + T5): R_d replicas</text>")
    for i, y in enumerate((100, 210, 320)):
        s += box(240, y, 260, 96, f"DiT replica {i + 1}",
                 ["DRA claim: n_d Neuron devices,", "each paired with its EFA", "TP x CP, KV cache per stream", "rank 0: NIXL agent"], BLUE)
    # decoder pool
    s.append(f"<rect x='610' y='60' width='340' height='400' rx='12' fill='none' stroke='{ORANGE}' stroke-width='2' stroke-dasharray='5 4'/>")
    s.append(f"<text x='626' y='84' font-size='13' font-weight='600' fill='{INK}'>Decoder role (VAE): R_v replicas</text>")
    for i, y in enumerate((100, 250)):
        s += box(630, y, 300, 120, f"VAE replica {i + 1}",
                 ["DRA claim: n_v devices (1 chip)", "K sessions: per-session causal cache", "(427 MiB each) + NIXL slot block",
                  "GET /load: sessions served", "one VAE serves ~ T_d / T_v streams"], ORANGE)
    s += box(630, 382, 300, 70, "Placement + failover", ["least-loaded VAE per session; a failed write:",
                                                         "set aside, retry on another (~3 s switch)"], INK2)
    s += arrow(174, 245, 236, 245, "session", dy=-8)
    s += arrow(500, 160, 626, 160, "", ORANGE)
    s.append(f"<text x='563' y='138' font-size='11' text-anchor='middle' fill='{INK}'>NIXL write</text>")
    s.append(f"<text x='563' y='152' font-size='10' text-anchor='middle' fill='{INK2}'>1.1 MB, 0.17 ms</text>")
    s += arrow(500, 258, 626, 300, "", ORANGE)
    s += arrow(500, 368, 626, 320, "", ORANGE)
    s.append(f"<text x='24' y='{H - 40}' font-size='11' fill='{INK2}'>Each role is sized (n_d, n_v), replicated (R_d, R_v) and placed on its own; "
             f"rate matching: R_v = ceil(R_d * T_v / T_d).</text>")
    s.append(f"<text x='24' y='{H - 22}' font-size='11' fill='{INK2}'>Co-located baseline: every chip runs the DiT, then the VAE: T_d(N) + T_v(N) per chunk; "
             f"split: max(T_d(n_d), T_v(n_v)).</text>")
    s.append("</svg>")
    return "\n".join(s)


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "fps_vs_chips.svg"), "w").write(fps_chart())
    open(os.path.join(OUT, "fps_vs_chips_480.svg"), "w").write(
        fps_chart(SERIES[:2], "LingBot-World v2 at 480x768: frame rate vs chips, one stream", note720=False))
    open(os.path.join(OUT, "architecture.svg"), "w").write(architecture())
    with open(os.path.join(OUT, "fps_vs_chips.csv"), "w") as f:
        f.write("series,total_chips,fps\n")
        for name, _, _, pts in SERIES:
            for c, v in pts:
                f.write(f"{name},{c},{v}\n")
        f.write("Co-located, 720p,8,does not fit\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()


# ---------------- more figures (all from measured cells, docs/disagg/RESULTS.md and perf/disagg/results/) -------------
AQUA, YELLOW = "#1baf7a", "#eda100"   # dataviz slots 3-4


def _axes(W, H, L, R, T, B, xs, ys, xlabel, ylabel, title, xticks, yticks, xfmt=str, yfmt=str, logx=False):
    import math
    X0, X1 = xs; Y0, Y1 = ys
    def X(v):
        if logx:
            return L + (W - L - R) * (math.log2(v) - math.log2(X0)) / (math.log2(X1) - math.log2(X0))
        return L + (W - L - R) * (v - X0) / (X1 - X0)
    Y = lambda v: H - B - (H - T - B) * (v - Y0) / (Y1 - Y0)
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='{L}' y='26' font-size='15' font-weight='600' fill='{INK}'>{title}</text>"]
    for v in yticks:
        s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(v)}' y2='{Y(v)}' stroke='{GRID}'/>")
        s.append(f"<text x='{L - 8}' y='{Y(v) + 4}' font-size='11' text-anchor='end' fill='{INK2}'>{yfmt(v)}</text>")
    for v in xticks:
        s.append(f"<text x='{X(v)}' y='{H - B + 18}' font-size='11' text-anchor='middle' fill='{INK2}'>{xfmt(v)}</text>")
    s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(Y0)}' y2='{Y(Y0)}' stroke='{INK2}'/>")
    s.append(f"<text x='{(L + W - R) / 2}' y='{H - 12}' font-size='12' text-anchor='middle' fill='{INK2}'>{xlabel}</text>")
    s.append(f"<text x='16' y='{(T + H - B) / 2}' font-size='12' fill='{INK2}' transform='rotate(-90 16 {(T + H - B) / 2})' text-anchor='middle'>{ylabel}</text>")
    return s, X, Y


def _line(s, X, Y, pts, col, name, dash=""):
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
    s.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='2' stroke-dasharray='{dash}'/>")
    for x, y in pts:
        s.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='4.5' fill='{col}' stroke='{SURF}' stroke-width='2'><title>{name}: {x}, {y}</title></circle>")


def _legend(s, x, y, items):
    for i, (name, col, dash) in enumerate(items):
        yy = y + i * 20
        s.append(f"<line x1='{x}' x2='{x + 22}' y1='{yy}' y2='{yy}' stroke='{col}' stroke-width='2' stroke-dasharray='{dash}'/>")
        s.append(f"<text x='{x + 30}' y='{yy + 4}' font-size='11' fill='{INK}'>{name}</text>")


def fig_scaling():
    s, X, Y = _axes(720, 420, 64, 40, 48, 56, (1, 16), (0, 1300), "chips (log scale)", "ms per 12-frame chunk",
                    "Per-stage scaling, LingBot 480x768", [1, 2, 4, 8, 16], range(0, 1301, 200), logx=True)
    _line(s, X, Y, [(2, 1192), (4, 686), (8, 464), (16, 375)], BLUE, "DiT alone (period)")
    _line(s, X, Y, [(1, 324), (2, 213), (4, 173), (8, 153)], ORANGE, "VAE pod")
    _line(s, X, Y, [(4, 151), (8, 133), (16, 143)], ORANGE, "VAE inline", "6 4")
    _legend(s, 430, 70, [("DiT alone: 3.2x from 2 to 16 chips", BLUE, ""), ("VAE pod: 2.1x from 1 to 8 chips", ORANGE, ""),
                         ("VAE inline on the DiT's chips: flat", ORANGE, "6 4")])
    s.append("</svg>")
    return "\n".join(s)


def fig_steps():
    # speedup of the split over co-located vs denoising steps (SA3, 2-chip DiT, 1800 s of audio)
    mono = {2: 3.06, 4: 4.07, 8: 6.15, 25: 14.88, 50: 27.81}
    v1 = {2: 3.75, 4: 3.89, 8: 5.48, 25: 14.39, 50: 27.02}
    v2 = {2: 2.51, 4: 3.32, 8: 5.37, 25: 14.16, 50: 26.80}
    s, X, Y = _axes(720, 420, 64, 40, 48, 56, (2, 50), (0.7, 1.3), "denoising steps (log scale)",
                    "speedup over co-located (x)", "Stable Audio 3: the gain follows the stage ratio",
                    [2, 4, 8, 25, 50], [0.8, 0.9, 1.0, 1.1, 1.2, 1.3], yfmt=lambda v: f"{v:.1f}", logx=True)
    s.append(f"<line x1='{X(2)}' x2='{X(50)}' y1='{Y(1.0)}' y2='{Y(1.0)}' stroke='{INK2}' stroke-dasharray='2 3'/>")
    _line(s, X, Y, [(k, round(mono[k] / v1[k], 3)) for k in mono], ORANGE, "2 DiT + 1 VAE chip")
    _line(s, X, Y, [(k, round(mono[k] / v2[k], 3)) for k in mono], AQUA, "2 DiT + 2 VAE chips")
    _legend(s, 420, 70, [("2 DiT + 1 VAE chip", ORANGE, ""), ("2 DiT + 2 VAE chips", AQUA, "")])
    s.append(f"<text x='{X(2.3)}' y='{Y(0.76)}' font-size='11' fill='{INK2}'>2 steps: the 1-chip VAE binds (loses); 2 VAE chips win 1.22x</text>")
    s.append(f"<text x='{X(12)}' y='{Y(1.17)}' font-size='11' fill='{INK2}'>many steps: the DiT dominates, gain -> 1</text>")
    s.append("</svg>")
    return "\n".join(s)


def fig_nm():
    s, X, Y = _axes(720, 420, 64, 40, 48, 56, (0.5, 3.5), (0, 20), "streams sharing one 1-chip VAE",
                    "fps per stream", "One VAE, several streams: no slowdown (LingBot 480)", [1, 2, 3], range(0, 21, 5))
    _line(s, X, Y, [(1, 10.07), (2, 10.06), (3, 10.08)], BLUE, "2-chip DiT streams")
    _line(s, X, Y, [(1, 17.52), (2, 17.53)], ORANGE, "4-chip DiT streams")
    _legend(s, 420, 70, [("2-chip DiT streams (limit ~3.7)", BLUE, ""), ("4-chip DiT streams (limit ~2.1)", ORANGE, "")])
    s.append(f"<text x='{X(0.6)}' y='{Y(2)}' font-size='11' fill='{INK2}'>3 decoders x 3 streams: 9 streams at 10.00-10.07 fps; 427 MiB per session</text>")
    s.append("</svg>")
    return "\n".join(s)


def fig_latency():
    rows = [("co-located 4", 498, 552, 62, "15.5 fps, p90 1416"), ("4 + 1 VAE chip", 476, 433, 314, "17.5 fps, p90 1471"),
            ("4 + 4 VAE chips", 470, 433, 209, "17.6 fps, p90 1371")]
    W, H, L = 760, 300, 150
    sc = (W - L - 130) / 1300
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='26' font-size='15' font-weight='600' fill='{INK}'>Press-to-photon latency (median, 100 presses): a 1-chip VAE adds 107 ms; a 4-chip VAE removes it</text>"]
    for i, (name, w, d, v, note) in enumerate(rows):
        y = 60 + i * 56
        s.append(f"<text x='{L - 10}' y='{y + 22}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        x = L
        for val, col, lab in ((w, GRID.replace('#e4e3df', '#c9c7c0'), "wait"), (d, BLUE, "DiT"), (v, ORANGE, "decode")):
            s.append(f"<rect x='{x:.1f}' y='{y}' width='{val * sc - 2:.1f}' height='34' rx='4' fill='{col}'><title>{name} {lab}: {val} ms</title></rect>")
            if val > 120:
                s.append(f"<text x='{x + 6:.1f}' y='{y + 22}' font-size='11' fill='{'#ffffff' if col != '#c9c7c0' else INK}'>{lab} {val}</text>")
            x += val * sc
        s.append(f"<text x='{x + 6:.1f}' y='{y + 16}' font-size='11' fill='{INK}'>= {w + d + v} ms</text>")
        s.append(f"<text x='{x + 6:.1f}' y='{y + 30}' font-size='10' fill='{INK2}'>{note}</text>")
    s.append(f"<text x='{L}' y='{H - 34}' font-size='11' fill='{INK2}'>A decoder sized for latency (4 chips) matches the co-located median (1114 ms)</text>")
    s.append(f"<text x='{L}' y='{H - 18}' font-size='11' fill='{INK2}'>and beats its p90 (1371 vs 1416 ms) at 13% more fps (17.6 vs 15.5).</text>")
    s.append("</svg>")
    return "\n".join(s)


def fig_model():
    import importlib.util, io, contextlib, os as _os
    pts = [(14.33, 14.33), (20.09, 20.03), (23.17, 23.37), (10.07, 10.07), (10.07, 10.06), (17.49, 17.54), (17.49, 17.52),
           (17.49, 17.47), (25.84, 25.76), (25.84, 25.95), (25.84, 25.48), (25.84, 25.60), (31.97, 32.14), (31.97, 32.21), (31.97, 32.17)]
    s, X, Y = _axes(520, 480, 64, 30, 48, 56, (8, 34), (8, 34), "predicted fps (from per-stage times)", "measured fps",
                    "Cost model vs measurement: 15 cells, within 1.4%", range(10, 35, 5), range(10, 35, 5))
    s.append(f"<line x1='{X(8)}' y1='{Y(8)}' x2='{X(34)}' y2='{Y(34)}' stroke='{INK2}' stroke-dasharray='3 3'/>")
    for i, (p, m) in enumerate(pts):
        col = BLUE if i < 3 else ORANGE
        s.append(f"<circle cx='{X(p):.1f}' cy='{Y(m):.1f}' r='5' fill='{col}' stroke='{SURF}' stroke-width='2'><title>predicted {p}, measured {m}</title></circle>")
    _legend(s, 90, 70, [("co-located (3)", BLUE, ""), ("disaggregated (12)", ORANGE, "")])
    s.append("</svg>")
    return "\n".join(s)


def fig_bounds():
    rows = [("DiT attention block", 6.38, 1.84, 0.60, 4.37, 5.25), ("DiT MXFP8 graph (/4)", 23.95 / 4, 3.81 / 4, 3.89 / 4, 12.99 / 4, 20.89 / 4),
            ("VAE halo conv", 5.71, 2.53, 0.23, 5.02, 2.89), ("VAE halo conv", 5.47, 2.48, 0.33, 4.96, 3.27)]
    W, H, L = 760, 380, 170
    sc = (W - L - 40) / 7
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='26' font-size='15' font-weight='600' fill='{INK}'>What bounds each stage: floors vs measured time (ms per execution)</text>"]
    keys = [("measured", INK2), ("compute floor", BLUE), ("HBM floor", AQUA), ("tensor busy", YELLOW), ("DMA busy", ORANGE)]
    for i, r in enumerate(rows):
        y = 56 + i * 70
        s.append(f"<text x='{L - 10}' y='{y + 30}' font-size='12' text-anchor='end' fill='{INK}'>{r[0]}</text>")
        for j, ((k, col), v) in enumerate(zip(keys, r[1:])):
            s.append(f"<rect x='{L}' y='{y + j * 12}' width='{max(v * sc, 2):.1f}' height='10' rx='3' fill='{col}'><title>{r[0]} {k}: {v:.2f} ms</title></rect>")
    lx = L
    for j, (k, col) in enumerate(keys):
        s.append(f"<rect x='{lx}' y='{H - 40}' width='12' height='10' rx='2' fill='{col}'/>")
        s.append(f"<text x='{lx + 16}' y='{H - 31}' font-size='11' fill='{INK}'>{k}</text>")
        lx += 110
    s.append(f"<text x='24' y='{H - 8}' font-size='11' fill='{INK2}'>DiT: DMA busy far above its HBM floor (on-chip movement). VAE: tensor busy ~ measured, compute floor ~half (array under-filled).</text>")
    s.append("</svg>")
    return "\n".join(s)


def main2():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("scaling", fig_scaling), ("steps", fig_steps), ("nm", fig_nm),
                     ("model", fig_model)):
        open(os.path.join(OUT, f"{name}.svg"), "w").write(fn())
    print("wrote 6 more figures")


if __name__ == "__main__":
    main2()


def fig_timeline():
    """Schematic (measured durations, LingBot 480, 16 chips): co-located DiT then VAE per chunk vs disaggregated overlap."""
    W, H, L = 900, 260, 150
    sc = (W - L - 190) / 1500      # ms -> px
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='26' font-size='15' font-weight='600' fill='{INK}'>One stream, three chunks: co-located vs disaggregated (16-chip DiT, measured times)</text>"]
    def bar(x0, y, ms, col, lab):
        s.append(f"<rect x='{L + x0 * sc:.1f}' y='{y}' width='{ms * sc - 2:.1f}' height='26' rx='4' fill='{col}'><title>{lab}: {ms} ms</title></rect>")
        if ms * sc > 40:
            s.append(f"<text x='{L + x0 * sc + 5:.1f}' y='{y + 17}' font-size='10' fill='#ffffff'>{lab}</text>")
    y = 60
    s.append(f"<text x='{L - 10}' y='{y + 17}' font-size='12' text-anchor='end' fill='{INK}'>co-located, 16 chips</text>")
    t = 0
    for k in range(3):
        bar(t, y, 322, BLUE, f"DiT {k}"); t += 322
        bar(t, y, 127, ORANGE, f"VAE {k}"); t += 127
    s.append(f"<text x='{L}' y='{y + 46}' font-size='12' fill='{INK}'>co-located: a chunk every 449 ms = 26.7 fps</text>")
    y2 = 140
    s.append(f"<text x='{L - 10}' y='{y2 + 17}' font-size='12' text-anchor='end' fill='{INK}'>DiT pod, 16 chips</text>")
    s.append(f"<text x='{L - 10}' y='{y2 + 51}' font-size='12' text-anchor='end' fill='{INK}'>VAE pod, 1 chip</text>")
    for k in range(3):
        bar(375 * k, y2, 375, BLUE, f"DiT {k}")
        bar(375 * (k + 1), y2 + 34, 324, ORANGE, f"VAE {k}")
    s.append(f"<text x='{L}' y='{y2 + 80}' font-size='12' fill='{INK}'>disaggregated: a chunk every 375 ms = 32 fps (the VAE decodes chunk n while the DiT denoises chunk n + 1)</text>")
    s.append(f"<text x='24' y='{H - 14}' font-size='11' fill='{INK2}'>Co-located: P = T_d(N) + T_v(N). Disaggregated: P = max(T_d(n_d), T_v(n_v)); the 1.1 MB latent moves over NIXL/EFA in 0.17 ms.</text>")
    s.append("</svg>")
    return "\n".join(s)


def fig_pareto():
    """Streams per chip vs fps per stream (LingBot 480): every measured configuration."""
    co = [(4, 1, 15.54), (8, 1, 22.63), (16, 1, 26.70)]                 # chips, streams, fps (last: async emit)
    sp = [(3, 1, 10.07), (5, 1, 17.54), (9, 1, 25.76), (17, 1, 32.14), (7, 3, 10.08), (9, 2, 17.53), (14, 6, 10.03), (21, 9, 10.04)]
    s, X, Y = _axes(720, 440, 64, 40, 48, 56, (0, 35), (0, 0.5), "fps per stream", "streams per chip",
                    "Throughput vs interactivity: every measured LingBot configuration (480x768)", range(0, 36, 5),
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5], yfmt=lambda v: f"{v:.1f}")
    for c, n, f in co:
        s.append(f"<circle cx='{X(f):.1f}' cy='{Y(n / c):.1f}' r='6' fill='{BLUE}' stroke='{SURF}' stroke-width='2'><title>co-located {c} chips: {f} fps</title></circle>")
    for c, n, f in sp:
        s.append(f"<circle cx='{X(f):.1f}' cy='{Y(n / c):.1f}' r='6' fill='{ORANGE}' stroke='{SURF}' stroke-width='2'><title>split {c} chips, {n} streams: {f} fps</title></circle>")
    for c, n, f, lab in ((21, 9, 10.04, "N:M, 3 streams per 1-chip VAE (3/7, 6/14, 9/21 chips)"), (3, 1, 10.07, "2+1"),
                         (9, 2, 17.53, "2 streams / 1 VAE"), (5, 1, 17.54, "4+1"), (17, 1, 32.14, "16+1"),
                         (9, 1, 25.76, "8+1"), (16, 1, 26.70, "co-located 16"), (8, 1, 22.63, "co-located 8"),
                         (4, 1, 15.54, "co-located 4")):
        dy = 16 if lab in ("4+1", "2+1") else -8
        s.append(f"<text x='{X(f) + 9:.1f}' y='{Y(n / c) + dy:.1f}' font-size='11' fill='{INK}'>{lab}</text>")
    _legend(s, X(1), Y(0.12), [("co-located", BLUE, ""), ("disaggregated (incl. N:M)", ORANGE, "")])
    s.append("</svg>")
    return "\n".join(s)


def main3():
    for name, fn in (("timeline", fig_timeline), ("pareto", fig_pareto)):
        open(os.path.join(OUT, f"{name}.svg"), "w").write(fn())
    print("wrote timeline, pareto")


if __name__ == "__main__":
    main3()


def fig_bounds2():
    """Where each stage's measured time goes: useful math (the compute floor) + the time lost, with its cause."""
    rows = [("DiT attention block", 6.38, 1.84, "data movement (memory / IO)", "on-chip data-movement engines busy 82% of the time"),
            ("VAE convolution", 5.71, 2.53, "underused compute", "math engine busy ~90% of the time at ~45% of its peak")]
    W, H, L = 760, 270, 160
    sc = (W - L - 90) / 6.5
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='26' font-size='15' font-weight='600' fill='{INK}'>What each stage spends its time on (ms per call, 4-chip, 480x768)</text>"]
    for i, (name, t, useful, kind, why) in enumerate(rows):
        y = 56 + i * 70
        s.append(f"<text x='{L - 10}' y='{y + 22}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        s.append(f"<rect x='{L}' y='{y}' width='{useful * sc - 2:.1f}' height='32' rx='4' fill='{BLUE}'><title>useful math {useful} ms</title></rect>")
        s.append(f"<text x='{L + 6}' y='{y + 21}' font-size='11' fill='#ffffff'>compute {useful}</text>")
        s.append(f"<rect x='{L + useful * sc:.1f}' y='{y}' width='{(t - useful) * sc - 2:.1f}' height='32' rx='4' fill='{ORANGE}'><title>headroom {t - useful:.2f} ms</title></rect>")
        s.append(f"<text x='{L + useful * sc + 6:.1f}' y='{y + 21}' font-size='11' fill='#ffffff'>{kind} {t - useful:.1f}</text>")
        s.append(f"<text x='{L + t * sc + 6:.1f}' y='{y + 21}' font-size='11' fill='{INK}'>= {t} ms</text>")
        s.append(f"<text x='{L}' y='{y + 48}' font-size='11' fill='{INK2}'>{why}</text>")
    s.append(f"<text x='24' y='{H - 44}' font-size='11' fill='{INK2}'>Blue: compute at full speed. Orange: the rest of the time, headroom a better kernel could reclaim.</text>")
    s.append(f"<text x='24' y='{H - 28}' font-size='11' fill='{INK2}'>Each stage's headroom has a different cause, so they scale differently with chips (the cause).</text>")
    s.append(f"<text x='24' y='{H - 12}' font-size='11' fill='{INK2}'>Disaggregation does not change it (that is kernel work); Figure chiptime shows what disaggregation saves.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds2())
    print("bounds.svg redrawn")


def fig_chiptime():
    """Chip time per 12-frame chunk, co-located 16 chips vs the 8 + 1 split (measured: chips x stage time; the rest of
    chips x period is waiting / overhead)."""
    rows = [("co-located, 16 chips (26.7 fps)", 4786, 2032, 7186 - 4786 - 2032), ("split 8 + 1 (25.8 fps)", 2976, 324, 4194 - 2976 - 324)]
    W, H, L = 780, 266, 210
    sc = (W - L - 120) / 7300
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='26' font-size='15' font-weight='600' fill='{INK}'>Chip time per 12-frame chunk (chip-ms), LingBot 480</text>"]
    for i, (name, d, v, o) in enumerate(rows):
        y = 60 + i * 64
        s.append(f"<text x='{L - 10}' y='{y + 22}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        x = L
        for val, col, lab in ((d, BLUE, "DiT"), (v, ORANGE, "VAE"), (o, "#c9c7c0", "waiting")):
            s.append(f"<rect x='{x:.1f}' y='{y}' width='{max(val * sc - 2, 2):.1f}' height='34' rx='4' fill='{col}'><title>{lab}: {val} chip-ms</title></rect>")
            if val * sc > 80:
                s.append(f"<text x='{x + 6:.1f}' y='{y + 22}' font-size='11' fill='{'#ffffff' if col != '#c9c7c0' else INK}'>{lab} {val}</text>")
            else:
                s.append(f"<text x='{x:.1f}' y='{y + (48 if lab == 'VAE' else 60)}' font-size='10' fill='{INK2}'>{lab} {val}</text>")
            x += val * sc
        s.append(f"<text x='{x + 14:.1f}' y='{y + 22}' font-size='11' fill='{INK}'>= {d + v + o}</text>")
    s.append(f"<text x='{L}' y='{H - 40}' font-size='11' fill='{INK2}'>The split uses 42% less chip time per chunk at 96% of the frame rate. Most of it is the VAE:</text>")
    s.append(f"<text x='{L}' y='{H - 24}' font-size='11' fill='{INK2}'>2032 -> 324 chip-ms, because it no longer occupies all 16 chips. The kernels are unchanged.</text>")
    s.append(f"<text x='{L}' y='{H - 8}' font-size='11' fill='{INK2}'>The effect of disaggregation; Figure bounds shows the cause.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "chiptime.svg"), "w").write(fig_chiptime())
    print("chiptime.svg")


def fig_latency2():
    """Two panels: reaction time (press-to-photon median) and frames per second, same three setups."""
    rows = [("co-located 4 chips", 1114, 15.5, BLUE), ("split 4 + 1 chips", 1221, 17.5, ORANGE), ("split 4 + 4 chips", 1114, 17.6, ORANGE)]
    W, H = 820, 300
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>Same reaction time, more frames per second (LingBot 480)</text>"]
    def panel(x0, title, key, vmax, unit):
        s.append(f"<text x='{x0}' y='62' font-size='13' font-weight='600' fill='{INK}'>{title}</text>")
        for i, r in enumerate(rows):
            y = 80 + i * 56
            v = r[key]
            w = 230 * v / vmax
            s.append(f"<text x='{x0}' y='{y + 14}' font-size='11' fill='{INK}'>{r[0]}</text>")
            s.append(f"<rect x='{x0}' y='{y + 20}' width='{w:.1f}' height='22' rx='4' fill='{r[3]}'><title>{r[0]}: {v} {unit}</title></rect>")
            s.append(f"<text x='{x0 + w + 8:.1f}' y='{y + 36}' font-size='12' fill='{INK}'>{v} {unit}</text>")
    panel(24, "Time for a key press to show up (median)", 1, 1300, "ms")
    panel(430, "Frames per second", 2, 20, "fps")
    s.append(f"<text x='24' y='{H - 30}' font-size='11' fill='{INK2}'>Like a car wash with two workers: a car comes out more often (more fps), but each car still</text>")
    s.append(f"<text x='24' y='{H - 14}' font-size='11' fill='{INK2}'>takes as long to get through (same reaction time). A 1-chip VAE is slow enough to add 107 ms; 4 VAE chips do not.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    pass   # latency.svg dropped: carwash.svg carries the latency point; the numbers are a table in the draft


def fig_carwash():
    """The car-wash picture with measured LingBot numbers (4-chip DiT, 480): chunk out every 774 ms co-located
    (15.5 fps) vs every 682 ms split 4 + 4 (17.6 fps); each press still takes ~1.11 s to show."""
    W, H, L = 860, 300, 170
    sc = (W - L - 30) / 6200
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>Overlapping the stages: more chunks per minute, same time per key press (LingBot 480, 4-chip DiT)</text>"]
    def row(y, name, period, col, perminute):
        s.append(f"<text x='{L - 10}' y='{y + 18}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        t, k = 0, 0
        while t + period <= 6200:
            s.append(f"<rect x='{L + t * sc:.1f}' y='{y}' width='{period * sc - 3:.1f}' height='28' rx='4' fill='{col}'><title>chunk {k + 1} out at {t + period:.0f} ms</title></rect>")
            s.append(f"<text x='{L + t * sc + 6:.1f}' y='{y + 18}' font-size='10' fill='#ffffff'>{k + 1}</text>")
            t += period; k += 1
        s.append(f"<text x='{L}' y='{y + 46}' font-size='11' fill='{INK2}'>{k} chunks in 6.2 s: a chunk (12 frames) every {period} ms = {perminute} frames per minute</text>")
    row(60, "co-located", 774, BLUE, 930)
    row(140, "split", 682, ORANGE, 1056)
    s.append(f"<text x='24' y='{H - 44}' font-size='12' fill='{INK}'>Each key press still takes about 1.1 s to show up in both (1114 ms median).</text>")
    s.append(f"<text x='24' y='{H - 26}' font-size='11' fill='{INK2}'>Co-located: each chunk is denoised, then decoded, on the same chips. Split: the VAE decodes chunk n while the DiT denoises chunk n + 1.</text>")
    s.append(f"<text x='24' y='{H - 10}' font-size='11' fill='{INK2}'>Split = 4 DiT chips + 4 VAE chips (17.6 fps); co-located = 4 chips (15.5 fps).</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    pass   # pipeline.svg removed (unclear); timeline.svg shows the overlap


def fig_latency3():
    """Reaction time (press-to-photon median): give the VAE enough chips, or reactions get ~0.1 s slower."""
    rows = [("co-located, 4 chips", 1114, BLUE, "baseline (15.5 fps)"),
            ("split, 4 DiT + 1 VAE chip", 1221, ORANGE, "VAE too small: 107 ms slower (17.5 fps)"),
            ("split, 4 DiT + 4 VAE chips", 1114, ORANGE, "VAE sized right: same as co-located (17.6 fps)")]
    W, H, L = 820, 260, 210
    sc = 380 / 1300
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>Give the VAE enough chips, or reactions get 0.1 s slower</text>",
         f"<text x='24' y='48' font-size='12' fill='{INK2}'>Time from a key press until the frame shows it (median of 100 presses, LingBot 480)</text>"]
    for i, (name, v, col, note) in enumerate(rows):
        y = 70 + i * 52
        s.append(f"<text x='{L - 10}' y='{y + 18}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        s.append(f"<rect x='{L}' y='{y}' width='{v * sc:.1f}' height='28' rx='4' fill='{col}'><title>{name}: {v} ms</title></rect>")
        s.append(f"<text x='{L + v * sc + 8:.1f}' y='{y + 13}' font-size='12' fill='{INK}'>{v} ms</text>")
        s.append(f"<text x='{L + v * sc + 8:.1f}' y='{y + 27}' font-size='11' fill='{INK2}'>{note}</text>")
    s.append(f"<text x='24' y='{H - 14}' font-size='11' fill='{INK2}'>Splitting never makes a press show up sooner; sized right, it keeps the reaction time and adds frames per second.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "latency.svg"), "w").write(fig_latency3())
    print("latency.svg")


def fig_architecture2():
    """Two decoupled role pools (as in LLM prefill/decode disaggregation figures): a scheduler, a DiT pool and a VAE pool
    that each scale out on their own, a many-to-many latent fabric between them, EKS/DRA underneath."""
    W, H = 980, 560
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
         f"<path d='M0,0 L10,5 L0,10 z' fill='{INK2}'/></marker></defs>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='30' font-size='16' font-weight='600' fill='{INK}'>Role-disaggregated serving: two pools that scale independently</text>"]
    def box(x, y, w, h, t, sub, col, fill="#ffffff"):
        s.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{col}' stroke-width='2'/>")
        s.append(f"<text x='{x + w / 2}' y='{y + h / 2 - 2}' font-size='12' font-weight='600' text-anchor='middle' fill='{INK}'>{t}</text>")
        if sub:
            s.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 14}' font-size='10' text-anchor='middle' fill='{INK2}'>{sub}</text>")
    def arrow(x1, y1, x2, y2, col=INK2):
        s.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='1.6' marker-end='url(#a)'/>")
    # clients and scheduler
    box(400, 48, 180, 44, "Clients (sessions)", "actions in, frames out", INK2)
    box(330, 116, 320, 50, "Scheduler", "places each session: a DiT replica + the least-loaded VAE", INK2, "#f3f2ee")
    arrow(490, 92, 490, 116)
    # DiT pool
    s.append(f"<rect x='40' y='196' width='380' height='250' rx='12' fill='none' stroke='{BLUE}' stroke-width='2' stroke-dasharray='6 4'/>")
    s.append(f"<text x='56' y='218' font-size='13' font-weight='600' fill='{INK}'>DiT pool (denoise): R_d replicas</text>")
    for i, x in enumerate((60, 180, 300)):
        lab = ["DiT 1", "DiT 2", "DiT R_d"][i]
        box(x, 240, 100, 64, lab, "n_d chips", BLUE)
    s.append(f"<text x='290' y='278' font-size='16' text-anchor='middle' fill='{INK2}'>...</text>")
    s.append(f"<text x='56' y='376' font-size='11' fill='{INK2}'>each replica: n_d devices + paired EFAs (DRA claim),</text>")
    s.append(f"<text x='56' y='392' font-size='11' fill='{INK2}'>TP x CP, per-stream KV cache</text>")
    s.append(f"<text x='56' y='420' font-size='12' font-weight='600' fill='{BLUE}'>scale out: add DiT replicas (more streams)</text>")
    # VAE pool
    s.append(f"<rect x='560' y='196' width='380' height='250' rx='12' fill='none' stroke='{ORANGE}' stroke-width='2' stroke-dasharray='6 4'/>")
    s.append(f"<text x='576' y='218' font-size='13' font-weight='600' fill='{INK}'>VAE pool (decode): R_v replicas</text>")
    for i, x in enumerate((580, 700, 820)):
        lab = ["VAE 1", "VAE 2", "VAE R_v"][i]
        box(x, 240, 100, 64, lab, "n_v chips", ORANGE)
    s.append(f"<text x='810' y='278' font-size='16' text-anchor='middle' fill='{INK2}'>...</text>")
    s.append(f"<text x='576' y='376' font-size='11' fill='{INK2}'>each replica: K sessions, a causal cache each,</text>")
    s.append(f"<text x='576' y='392' font-size='11' fill='{INK2}'>serves ~ T_d / T_v DiT streams</text>")
    s.append(f"<text x='576' y='420' font-size='12' font-weight='600' fill='{ORANGE}'>scale out: add VAE replicas (keep the ratio)</text>")
    # many-to-many latent fabric: a bus under both pools, every replica attached
    bus = 336
    s.append(f"<line x1='110' y1='{bus}' x2='870' y2='{bus}' stroke='{INK2}' stroke-width='4'/>")
    for x in (110, 230, 350):
        s.append(f"<line x1='{x}' y1='304' x2='{x}' y2='{bus}' stroke='{BLUE}' stroke-width='2'/>")
    for x in (630, 750, 870):
        s.append(f"<line x1='{x}' y1='{bus}' x2='{x}' y2='304' stroke='{ORANGE}' stroke-width='2' marker-end='url(#a)'/>")
    s.append(f"<text x='490' y='{bus - 22}' font-size='12' font-weight='600' text-anchor='middle' fill='{INK}'>latent fabric</text>")
    s.append(f"<text x='490' y='{bus - 8}' font-size='11' text-anchor='middle' fill='{INK}'>any DiT to any VAE</text>")
    s.append(f"<text x='490' y='{bus + 18}' font-size='10' text-anchor='middle' fill='{INK2}'>NIXL over EFA</text>")
    s.append(f"<text x='490' y='{bus + 32}' font-size='10' text-anchor='middle' fill='{INK2}'>or NeuronLink</text>")
    s.append(f"<text x='490' y='{bus + 46}' font-size='10' text-anchor='middle' fill='{INK2}'>1.1 MB, 0.17 ms</text>")
    # scheduler to pools
    arrow(400, 166, 230, 196); arrow(580, 166, 750, 196)
    # frames back
    s.append(f"<path d='M 940 260 C 975 200, 975 80, 580 70' fill='none' stroke='{INK2}' stroke-width='1.4' stroke-dasharray='4 3' marker-end='url(#a)'/>")
    s.append(f"<text x='920' y='120' font-size='10' fill='{INK2}' text-anchor='end'>frames back</text>")
    # platform layer
    box(40, 470, 900, 40, "Amazon EKS + Neuron DRA: device-and-EFA claims per role; pod discovery, least-loaded placement, failover", "", INK2, "#f3f2ee")
    s.append(f"<text x='40' y='536' font-size='11' fill='{INK2}'>Sizing: n_d, n_v from each stage's scaling curve; ratio R_v = ceil(R_d * T_v / T_d). Co-located instead: every chip runs both stages, T_d(N) + T_v(N) per chunk.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "architecture.svg"), "w").write(fig_architecture2())
    print("architecture.svg redrawn")


def fig_architecture3():
    """The serving flow as it runs (deck S11e, serve.py, vae_serve.py): browser -> lbw-web (session manager) -> any Ready
    DiT pod (the session is bound to it) -> that DiT picks any Ready VAE with an open slot -> frames back via lbw-web."""
    W, H = 1000, 600
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
         f"<path d='M0,0 L10,5 L0,10 z' fill='{INK2}'/></marker>"
         f"<marker id='c' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='4' markerHeight='4' orient='auto'>"
         f"<path d='M0,0 L10,5 L0,10 z' fill='{ORANGE}'/></marker></defs>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='30' font-size='16' font-weight='600' fill='{INK}'>How a session is served: two pools that scale independently</text>"]
    def box(x, y, w, h, t, sub, col, fill="#ffffff", fs=12):
        s.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{col}' stroke-width='2'/>")
        s.append(f"<text x='{x + w / 2}' y='{y + (h / 2 if not sub else h / 2 - 4)}' font-size='{fs}' font-weight='600' text-anchor='middle' fill='{INK}'>{t}</text>")
        if sub:
            s.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 12}' font-size='10' text-anchor='middle' fill='{INK2}'>{sub}</text>")
    def arrow(x1, y1, x2, y2, col=INK2, w=1.8, dash=""):
        s.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='{w}' stroke-dasharray='{dash}' marker-end='url(#a)'/>")
    def lab(x, y, t, col=INK2, fs=10, anchor="middle"):
        s.append(f"<text x='{x}' y='{y}' font-size='{fs}' text-anchor='{anchor}' fill='{col}'>{t}</text>")
    # browser and session manager
    box(30, 230, 120, 60, "Client", "actions in, frames out", INK2)
    box(190, 220, 150, 80, "Session manager", "opens and tracks sessions", INK2, "#f3f2ee")
    arrow(150, 252, 190, 252); arrow(190, 270, 150, 270)
    lab(170, 244, "1"); lab(170, 286, "4")
    # DiT pool
    s.append(f"<rect x='380' y='60' width='250' height='420' rx='12' fill='none' stroke='{BLUE}' stroke-width='2' stroke-dasharray='6 4'/>")
    s.append(f"<text x='394' y='82' font-size='13' font-weight='600' fill='{INK}'>DiT pool (DiT + T5)</text>")
    lab(394, 98, "any Ready pod takes a new session", INK2, 10, "start")
    for i, y in enumerate((112, 202, 292)):
        box(400, y, 210, 70, ["DiT pod 1", "DiT pod 2", "DiT pod R_d"][i], "one session at a time · n_d chips", BLUE)
    lab(505, 382, "...", INK2, 16)
    lab(394, 410, "scale out: add DiT pods", BLUE, 12, "start")
    lab(394, 428, "= more concurrent sessions", BLUE, 11, "start")
    arrow(340, 260, 380, 260)
    lab(360, 248, "2", INK2, 11)
    lab(358, 284, "bound", BLUE, 9); lab(358, 295, "(KV", BLUE, 9); lab(358, 306, "cache)", BLUE, 9)
    # VAE pool with slots
    s.append(f"<rect x='690' y='60' width='280' height='420' rx='12' fill='none' stroke='{ORANGE}' stroke-width='2' stroke-dasharray='6 4'/>")
    s.append(f"<text x='704' y='82' font-size='13' font-weight='600' fill='{INK}'>VAE pool</text>")
    lab(704, 98, "any Ready pod with an open slot", INK2, 10, "start")
    for i, y in enumerate((112, 222)):
        box(710, y, 240, 90, ["VAE pod 1", "VAE pod R_v"][i], "", ORANGE)
        for k in range(4):
            col = ORANGE if (i == 0 and k < 3) else "#c9c7c0"
            s.append(f"<rect x='{730 + k * 52}' y='{y + 52}' width='44' height='24' rx='4' fill='{col}'/>")
            s.append(f"<text x='{752 + k * 52}' y='{y + 68}' font-size='10' text-anchor='middle' fill='#ffffff'>slot {k + 1}</text>")
    lab(830, 330, "...", INK2, 16)
    lab(704, 360, "K open slots per VAE pod: it serves", INK2, 11, "start")
    lab(704, 376, "up to K sessions (~ T_d / T_v keep up)", INK2, 11, "start")
    lab(704, 410, "scale out: add VAE pods", ORANGE, 12, "start")
    lab(704, 428, "= more open slots", ORANGE, 11, "start")
    # DiT -> VAE streams (any to any)
    # one pool-to-pool arrow, with a circular arrow around it: any VAE, least loaded (no fixed pairing)
    arrow(630, 260, 690, 260, ORANGE, 2.4)
    s.append(f"<path d='M 668 244 A 17 17 0 1 0 676 266' fill='none' stroke='{ORANGE}' stroke-width='2' marker-end='url(#c)'/>")
    lab(660, 232, "3", INK2, 11)
    lab(660, 296, "any VAE", ORANGE, 9); lab(660, 307, "least-loaded", ORANGE, 9); lab(660, 318, "failover ~3 s", ORANGE, 9)
    # frames back
    s.append(f"<path d='M 830 480 L 830 500 L 265 500 L 265 300' fill='none' stroke='{INK2}' stroke-width='1.6' stroke-dasharray='5 4' marker-end='url(#a)'/>")
    lab(560, 516, "4 frames back through the session manager to the client", INK2, 11)
    s.append(f"<text x='24' y='{H - 44}' font-size='11' fill='{INK2}'>1 the client opens a session. 2 any Ready DiT pod takes it; the session is bound to that pod (it holds the KV cache).</text>")
    s.append(f"<text x='24' y='{H - 28}' font-size='11' fill='{INK2}'>3 the DiT streams its latents to any VAE pod with a free slot (least loaded); if that VAE fails, another takes over in ~3 s. 4 frames back.</text>")
    s.append(f"<text x='24' y='{H - 12}' font-size='11' fill='{INK2}'>Each role scales on its own: DiT pods for more sessions, VAE pods for more open slots. Kubernetes DRA gives each pod its Neuron devices + EFAs.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "architecture.svg"), "w").write(fig_architecture3())
    print("architecture.svg (flow)")


def fig_bounds3():
    """Measured time split three ways (tensor-engine profile of the top graph of each stage):
    compute at full steps (FLOPs / peak), compute in partly-full steps (tensor busy - that), compute engine waiting."""
    rows = [("DiT attention block", 1.84, 4.37 - 1.84, 6.38 - 4.37, "waiting on data movement (memory / IO)"),
            ("VAE convolution", 2.53, 5.02 - 2.53, 5.71 - 5.02, "waiting on normalization and tile-edge exchange")]
    W, H, L = 800, 300, 160
    sc = (W - L - 90) / 6.5
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='26' font-size='15' font-weight='600' fill='{INK}'>What each stage spends its time on (ms per call, 4-chip, 480x768)</text>"]
    for i, (name, full, partial, wait, why) in enumerate(rows):
        y = 56 + i * 76
        s.append(f"<text x='{L - 10}' y='{y + 22}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        x = L
        for v, col, lab in ((full, BLUE, f"full steps {full:.2f}"), (partial, AQUA, f"partly-full steps {partial:.2f}"), (wait, ORANGE, f"waiting {wait:.2f}")):
            s.append(f"<rect x='{x:.1f}' y='{y}' width='{v * sc - 2:.1f}' height='32' rx='4' fill='{col}'><title>{name} {lab} ms</title></rect>")
            txt = lab if v * sc > 80 else f"{v:.2f}"
            s.append(f"<text x='{x + 5:.1f}' y='{y + 21}' font-size='10' fill='{INK if col == AQUA else '#ffffff'}'>{txt}</text>")
            x += v * sc
        s.append(f"<text x='{x + 6:.1f}' y='{y + 21}' font-size='11' fill='{INK}'>= {full + partial + wait:.2f} ms</text>")
        s.append(f"<text x='{L}' y='{y + 50}' font-size='11' fill='{INK2}'>orange: {why}</text>")
    s.append(f"<text x='24' y='{H - 60}' font-size='11' fill='{INK2}'>Blue + aqua: the math engine computing (blue = the time it would need with every step full).</text>")
    s.append(f"<text x='24' y='{H - 44}' font-size='11' fill='{INK2}'>Orange: the math engine waiting. The DiT waits on data movement (2.0 ms); the VAE hardly waits (0.7 ms).</text>")
    s.append(f"<text x='24' y='{H - 28}' font-size='11' fill='{INK2}'>Different bottlenecks, so the stages scale differently with chips (the cause).</text>")
    s.append(f"<text x='24' y='{H - 12}' font-size='11' fill='{INK2}'>Disaggregation does not change these kernels; Figure chiptime shows what it saves.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds3())
    print("bounds.svg (three parts)")


def fig_bounds4():
    """Two parts per stage: the math engine computing vs waiting (tensor-engine busy vs idle, top graph per stage)."""
    rows = [("DiT", 4.37, 6.38 - 4.37, "I/O a third of the time"),
            ("VAE", 5.02, 5.71 - 5.02, "almost no I/O")]
    W, H, L = 760, 230, 80
    sc = (W - L - 190) / 6.5
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>The DiT is I/O-heavy; the VAE is compute-bound</text>",
         f"<text x='24' y='48' font-size='12' fill='{INK2}'>Time per call of each stage's main program (ms, 4 chips, 480x768)</text>"]
    for i, (name, comp, wait, note) in enumerate(rows):
        y = 70 + i * 60
        s.append(f"<text x='{L - 12}' y='{y + 22}' font-size='14' font-weight='600' text-anchor='end' fill='{INK}'>{name}</text>")
        s.append(f"<rect x='{L}' y='{y}' width='{comp * sc - 2:.1f}' height='34' rx='4' fill='{BLUE}'/>")
        s.append(f"<text x='{L + 8}' y='{y + 22}' font-size='12' fill='#ffffff'>compute {comp:.1f}</text>")
        s.append(f"<rect x='{L + comp * sc:.1f}' y='{y}' width='{wait * sc - 2:.1f}' height='34' rx='4' fill='{ORANGE}'/>")
        s.append(f"<text x='{L + comp * sc + 6:.1f}' y='{y + 22}' font-size='12' fill='#ffffff'>{'I/O ' if wait * sc > 60 else ''}{wait:.1f}</text>")
        s.append(f"<text x='{L + (comp + wait) * sc + 8:.1f}' y='{y + 22}' font-size='12' fill='{INK}'>{note}</text>")
    s.append(f"<text x='24' y='{H - 26}' font-size='11' fill='{INK2}'>Blue: compute. Orange: I/O (moving data between the chip's memories and to the other chips).</text>")
    s.append(f"<text x='24' y='{H - 10}' font-size='11' fill='{INK2}'>Slow for different reasons, so the two stages need different numbers of chips.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds4())
    print("bounds.svg (computing vs waiting)")


def fig_bounds5():
    """Typical usage of each stage, side by side (all measured): matmul engine busy, data movement busy, device memory per
    rank, speed-up with chips. The point: the DiT is a big, chip-hungry, talkative model; the VAE is small and compute-dense."""
    W, H = 900, 330
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>Two stages, two usage profiles (LingBot, measured)</text>"]
    panels = [("Matmul engine busy", "% of time", 100, (68, 88), "{:.0f}%"),
              ("Data movement busy", "% of time, overlaps compute", 100, (82, 51), "{:.0f}%"),
              ("Device memory per rank", "GiB (DiT: weights + KV, 720p)", 20, (8.4 + 8.65, 3.0), "{:.1f}"),
              ("Speed-up with more chips", "DiT 2 -> 16 chips, VAE 1 -> 8", 4, (3.2, 2.1), "{:.1f}x")]
    pw, x0 = 205, 24
    for k, (title, unit, vmax, (d, v), fmt) in enumerate(panels):
        x = x0 + k * (pw + 12)
        s.append(f"<text x='{x}' y='64' font-size='12' font-weight='600' fill='{INK}'>{title}</text>")
        s.append(f"<text x='{x}' y='80' font-size='10' fill='{INK2}'>{unit}</text>")
        base = 270
        for j, (val, col, name) in enumerate(((d, BLUE, "DiT"), (v, ORANGE, "VAE"))):
            bh = 160 * min(val, vmax) / vmax
            bx = x + 30 + j * 80
            s.append(f"<rect x='{bx}' y='{base - bh:.1f}' width='56' height='{bh:.1f}' rx='4' fill='{col}'/>")
            s.append(f"<text x='{bx + 28}' y='{base - bh - 6:.1f}' font-size='12' text-anchor='middle' fill='{INK}'>{fmt.format(val)}</text>")
            s.append(f"<text x='{bx + 28}' y='{base + 16}' font-size='11' text-anchor='middle' fill='{INK}'>{name}</text>")
        s.append(f"<line x1='{x + 20}' x2='{x + pw - 10}' y1='{base}' y2='{base}' stroke='{INK2}'/>")
    s.append(f"<text x='24' y='{H - 18}' font-size='11' fill='{INK2}'>The DiT is big, spreads over several chips and talks between them; the VAE is small and compute-dense, fits one chip with room to share.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds5())
    print("bounds.svg (usage profiles)")


def fig_bounds6():
    """Usage profiles of the two stages from ONE split run (LingBot 720p, 16-chip DiT + 2-chip VAE, 2026-11-22 system +
    device profiles, hottest graph of each stage): matmul busy, data movement busy, all engines idle, memory, speed-up."""
    W, H = 820, 330
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>Two stages, two usage profiles (LingBot 720p, 16-chip DiT + 2-chip VAE, measured)</text>"]
    panels = [("Matmul engine busy", "% of time", 100, (57, 94), "{:.0f}%"),
              ("Data movement busy", "% of time, overlaps compute", 100, (80, 41), "{:.0f}%"),
              ("All engines idle", "% of time (waiting)", 100, (19, 2), "{:.0f}%"),
              ("Device memory per rank", "GiB (DiT: weights + KV)", 20, (8.4 + 8.65, 3.0), "{:.1f}")]
    pw, x0 = 182, 20
    for k, (title, unit, vmax, (d, v), fmt) in enumerate(panels):
        x = x0 + k * (pw + 12)
        s.append(f"<text x='{x}' y='64' font-size='12' font-weight='600' fill='{INK}'>{title}</text>")
        s.append(f"<text x='{x}' y='80' font-size='10' fill='{INK2}'>{unit}</text>")
        base = 270
        for j, (val, col, name) in enumerate(((d, BLUE, "DiT"), (v, ORANGE, "VAE"))):
            bh = 160 * min(val, vmax) / vmax
            bx = x + 24 + j * 72
            s.append(f"<rect x='{bx}' y='{base - bh:.1f}' width='52' height='{max(bh, 2):.1f}' rx='4' fill='{col}'><title>{name} {title}: {fmt.format(val)}</title></rect>")
            s.append(f"<text x='{bx + 26}' y='{base - bh - 6:.1f}' font-size='12' text-anchor='middle' fill='{INK}'>{fmt.format(val)}</text>")
            s.append(f"<text x='{bx + 26}' y='{base + 16}' font-size='11' text-anchor='middle' fill='{INK}'>{name}</text>")
        s.append(f"<line x1='{x + 14}' x2='{x + pw - 10}' y1='{base}' y2='{base}' stroke='{INK2}'/>")
    s.append(f"<text x='24' y='{H - 18}' font-size='11' fill='{INK2}'>The DiT moves data and waits (idle 19%); the VAE keeps its matmul engine busy (94%). Different needs, so they get different chips.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds6())
    print("bounds.svg (usage profiles, 720p split run)")


def fig_bounds7():
    """Per-engine busy time of each stage's hottest graph, from one split run (LingBot 720p, 16-chip DiT + 2-chip VAE,
    2026-11-22 device profiles, engine_overlap.py): which engine bounds each stage."""
    rows = [("Tensor (matmul)", 57, 94), ("Scalar (exp, activations)", 54, 49), ("Vector (max, sums, casts)", 35, 25),
            ("GpSimd (DMA setup, misc)", 24, 19), ("DMA (data movement)", 80, 41), ("all engines idle", 19, 2)]
    W, H, L = 860, 420, 210
    sc = 4.2
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>What bounds each stage: % of time each engine is busy</text>",
         f"<text x='24' y='46' font-size='12' fill='{INK2}'>LingBot 720p split run, hottest graph of each stage (DiT: two transformer blocks; VAE: a fused conv)</text>"]
    for i, (name, d, v) in enumerate(rows):
        y = 70 + i * 50
        if name == "all engines idle":
            s.append(f"<line x1='24' x2='{W - 24}' y1='{y - 8}' y2='{y - 8}' stroke='{GRID}'/>")
        s.append(f"<text x='{L - 10}' y='{y + 17}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        for j, (val, col, who) in enumerate(((d, BLUE, "DiT"), (v, ORANGE, "VAE"))):
            yy = y + j * 15
            s.append(f"<rect x='{L}' y='{yy}' width='{max(val * sc, 2):.1f}' height='13' rx='3' fill='{col}'><title>{who} {name}: {val}%</title></rect>")
            s.append(f"<text x='{L + val * sc + 6:.1f}' y='{yy + 11}' font-size='11' fill='{INK}'>{who} {val}%</text>")
    s.append(f"<line x1='{L + 100 * sc}' x2='{L + 100 * sc}' y1='62' y2='{70 + 6 * 50 - 14}' stroke='{INK2}' stroke-dasharray='3 3'/>")
    s.append(f"<text x='{L + 100 * sc - 4}' y='60' font-size='10' text-anchor='end' fill='{INK2}'>100%</text>")
    s.append(f"<text x='24' y='{H - 30}' font-size='12' fill='{INK}'>VAE: compute-bound. The matmul engine is busy 94% of the time; nothing waits.</text>")
    s.append(f"<text x='24' y='{H - 12}' font-size='12' fill='{INK}'>DiT: no engine is saturated. DMA is the closest (80%), the matmul engine 57%, and everything waits 19%: data movement, not math.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds7())
    print("bounds.svg (per engine)")


def fig_bounds8():
    """Each stage's wall time split three ways (sums to 100%): the matmul engine busy (computing), the matmul engine
    idle while other engines work (moving data, softmax, casts), and every engine idle (waiting). From engine_overlap.py
    of each stage's hottest graph, 720p split run (DiT: Tensor 57.2%, all idle 18.8%; VAE: Tensor 94.0%, all idle 1.8%)."""
    rows = [("DiT (16 chips)", 57.2, 24.0, 18.8), ("VAE (2 chips)", 94.0, 4.2, 1.8)]
    W, H, L = 860, 250, 130
    sc = (W - L - 160) / 100
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='24' y='28' font-size='16' font-weight='600' fill='{INK}'>What each stage's chips are doing, % of the time (sums to 100%)</text>",
         f"<text x='24' y='46' font-size='12' fill='{INK2}'>LingBot 720p split run, hottest graph of each stage</text>"]
    parts = (("computing (matmul busy)", BLUE), ("moving data / softmax, matmul idle", AQUA), ("waiting (all idle)", "#c9c7c0"))
    for i, (name, *vals) in enumerate(rows):
        y = 70 + i * 62
        s.append(f"<text x='{L - 10}' y='{y + 22}' font-size='12' text-anchor='end' fill='{INK}'>{name}</text>")
        x = L
        for (lab, col), v in zip(parts, vals):
            s.append(f"<rect x='{x:.1f}' y='{y}' width='{max(v * sc - 2, 2):.1f}' height='34' rx='4' fill='{col}'><title>{name} {lab}: {v:.0f}%</title></rect>")
            if v * sc > 40:
                s.append(f"<text x='{x + 6:.1f}' y='{y + 22}' font-size='12' fill='{'#ffffff' if col == BLUE else INK}'>{v:.0f}%</text>")
            else:
                s.append(f"<text x='{x + (0 if lab.startswith('moving') else 26):.1f}' y='{y + 48}' font-size='11' fill='{INK2}'>{v:.0f}%</text>")
            x += v * sc
    lx = L
    for lab, col in parts:
        s.append(f"<rect x='{lx}' y='{H - 62}' width='12' height='10' rx='2' fill='{col}'/>")
        s.append(f"<text x='{lx + 16}' y='{H - 53}' font-size='11' fill='{INK}'>{lab}</text>")
        lx += 16 + 7 * len(lab) + 20
    s.append(f"<text x='24' y='{H - 26}' font-size='12' fill='{INK}'>VAE: compute-bound, computing 94% of the time.</text>")
    s.append(f"<text x='24' y='{H - 10}' font-size='12' fill='{INK}'>DiT: computing only 57%; 24% moving data with the matmul engine idle, 19% waiting.</text>")
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds8())
    print("bounds.svg (sums to 100%)")


if __name__ == "__main__":
    # bounds.svg: per-engine busy (engines run in parallel, so the shares overlap); bounds_100.svg: the same time split
    # three ways, summing to 100%
    open(os.path.join(OUT, "bounds.svg"), "w").write(fig_bounds7())
    open(os.path.join(OUT, "bounds_100.svg"), "w").write(fig_bounds8())
    print("bounds.svg (per engine) + bounds_100.svg")
