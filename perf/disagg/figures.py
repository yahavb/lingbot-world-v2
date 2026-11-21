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
    ("Co-located, 480x768", BLUE, "", [(4, 14.33), (8, 20.03), (16, 23.37)]),
    ("Split: DiT + 1-chip VAE, 480x768", ORANGE, "", [(3, 10.07), (5, 17.54), (9, 25.76), (17, 32.14)]),
    ("Co-located, 720p", BLUE, "6 4", [(16, 9.96)]),
    ("Split: DiT + 1-chip VAE, 720p", ORANGE, "6 4", [(9, 8.46), (17, 12.68)]),
]


def fps_chart():
    W, H, L, R, T, B = 760, 460, 64, 150, 48, 56
    xmax, ymax = 18, 35
    X = lambda c: L + (W - L - R) * c / xmax
    Y = lambda f: H - B - (H - T - B) * f / ymax
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='{SURF}'/>",
         f"<text x='{L}' y='26' font-size='16' font-weight='600' fill='{INK}'>LingBot-World v2: frame rate vs chips, one stream</text>"]
    for f in range(0, ymax + 1, 5):
        s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(f)}' y2='{Y(f)}' stroke='{GRID}' stroke-width='1'/>")
        s.append(f"<text x='{L - 8}' y='{Y(f) + 4}' font-size='11' text-anchor='end' fill='{INK2}'>{f}</text>")
    for c in (2, 4, 6, 8, 10, 12, 14, 16, 18):
        s.append(f"<text x='{X(c)}' y='{H - B + 18}' font-size='11' text-anchor='middle' fill='{INK2}'>{c}</text>")
    s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(0)}' y2='{Y(0)}' stroke='{INK2}' stroke-width='1'/>")
    s.append(f"<text x='{(L + W - R) / 2}' y='{H - 14}' font-size='12' text-anchor='middle' fill='{INK2}'>total chips (Trainium3)</text>")
    s.append(f"<text x='16' y='{(T + H - B) / 2}' font-size='12' fill='{INK2}' transform='rotate(-90 16 {(T + H - B) / 2})' text-anchor='middle'>frames per second</text>")
    for name, col, dash, pts in SERIES:
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(c):.1f},{Y(f):.1f}" for i, (c, f) in enumerate(pts))
            s.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='2' stroke-dasharray='{dash}' stroke-linejoin='round'/>")
        for c, f in pts:
            s.append(f"<circle cx='{X(c):.1f}' cy='{Y(f):.1f}' r='4.5' fill='{col if not dash else SURF}' stroke='{col}' "
                     f"stroke-width='2'><title>{name}: {c} chips, {f} fps</title></circle>")
    # direct labels at the line ends (text in ink, the mark carries the color)
    s.append(f"<text x='{X(17) + 10}' y='{Y(32.14) + 4}' font-size='12' fill='{INK}'>split 32.1</text>")
    s.append(f"<text x='{X(16) + 10}' y='{Y(23.37) + 4}' font-size='12' fill='{INK}'>co-located 23.4</text>")
    s.append(f"<text x='{X(17) + 10}' y='{Y(12.68) + 4}' font-size='12' fill='{INK}'>split 720p 12.7</text>")
    s.append(f"<text x='{X(16) + 10}' y='{Y(9.96) + 14}' font-size='12' fill='{INK}'>co-located 720p 10.0</text>")
    # co-located 8 chips at 720p: no point (does not fit)
    s.append(f"<text x='{X(8)}' y='{Y(2.2)}' font-size='11' text-anchor='middle' fill='{INK2}'>720p co-located, 8 chips:</text>")
    s.append(f"<text x='{X(8)}' y='{Y(0.6)}' font-size='11' text-anchor='middle' fill='{INK2}'>does not fit in device memory</text>")
    # legend (always present for >= 2 series)
    lx, ly = L + 14, T + 14
    for i, (name, col, dash, _) in enumerate(SERIES):
        y = ly + i * 22
        s.append(f"<line x1='{lx}' x2='{lx + 22}' y1='{y}' y2='{y}' stroke='{col}' stroke-width='2' stroke-dasharray='{dash}'/>")
        s.append(f"<circle cx='{lx + 11}' cy='{y}' r='4' fill='{col if not dash else SURF}' stroke='{col}' stroke-width='2'/>")
        s.append(f"<text x='{lx + 30}' y='{y + 4}' font-size='11' fill='{INK}'>{name}</text>")
    s.append(f"<text x='{lx}' y='{ly + 4 * 22 + 14}' font-size='10' fill='{INK2}'>Split = DiT on n chips + VAE on 1 chip</text>")
    s.append(f"<text x='{lx}' y='{ly + 4 * 22 + 28}' font-size='10' fill='{INK2}'>(total n + 1). Same flags both sides.</text>")
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
