"""model_check.py: the stage-composition cost model against every measured LingBot 480 cell (docs/disagg/RESULTS.md).
Inputs: per-stage measurements only (DiT alone, VAE pod, inline VAE). Monolith = 12000 / (P_d(N) + T_v_inline(N));
split = 12000 / max(P_d(n_d), T_v_pod(n_v)); streams per VAE = P_d / T_v_pod(1); plus a held-out scaling fit."""
fps_null = {2: 10.07, 4: 17.49, 8: 25.84, 16: 31.97}            # p1_dit*_480 (DiT alone)
Pd = {n: 12000 / f for n, f in fps_null.items()}
Tv_inline = {4: 151.3, 8: 132.9, 16: 142.5}                      # p6_monofix* [PHASE] VAE (synced)
Tv_pod = {1: 324, 2: 213, 4: 173, 8: 153}                        # VAE pods, median decode per chunk
mono = {4: 14.33, 8: 20.03, 16: 23.37}                           # p6_monofix4/8/16u
split = {(2, 1): 10.07, (2, 2): 10.06, (4, 1): 17.54, (4, 2): 17.52, (4, 4): 17.47, (8, 1): 25.76, (8, 2): 25.95,
         (8, 4): 25.48, (8, 8): 25.60, (16, 1): 32.14, (16, 2): 32.21, (16, 4): 32.17}   # p3_split_*_480
for n, f in mono.items():
    p = 12000 / (Pd[n] + Tv_inline[n]); print(f"monolith {n:2d}: predicted {p:5.2f} measured {f:5.2f} ({100*(p-f)/f:+.1f}%)")
for (d, v), f in split.items():
    p = 12000 / max(Pd[d], Tv_pod[v]); print(f"split {d:2d}+{v}: predicted {p:5.2f} measured {f:5.2f} ({100*(p-f)/f:+.1f}%)")
print("streams per VAE:", {d: round(Pd[d] / Tv_pod[1], 2) for d in Pd})
xs, ys = [1/2, 1/4, 1/8], [Pd[2], Pd[4], Pd[8]]
mx, my = sum(xs) / 3, sum(ys) / 3
a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs); b = my - a * mx
print(f"held-out: P_d(n) = {a:.0f}/n + {b:.0f} (fit 2, 4, 8) -> 16: {a/16 + b:.0f} ms vs {Pd[16]:.0f} measured")


def fit3(n, y):
    """T(n) = a/n + b + c*log2(n) through 3 points (n doubling)."""
    import math
    (n1, n2, n3), (y1, y2, y3) = n, y
    A = [[1/n1, 1, math.log2(n1)], [1/n2, 1, math.log2(n2)], [1/n3, 1, math.log2(n3)]]
    # Cramer
    det = lambda m: (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                     + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    D = det(A); Y = [y1, y2, y3]
    sol = []
    for k in range(3):
        M = [row[:] for row in A]
        for r in range(3):
            M[r][k] = Y[r]
        sol.append(det(M) / D)
    return lambda m: sol[0]/m + sol[1] + sol[2]*math.log2(m), sol


import math
f, s = fit3((2, 4, 8), (Pd[2], Pd[4], Pd[8]))
print(f"held-out with a communication term: P_d = {s[0]:.0f}/n + {s[1]:.0f} + {s[2]:.0f} log2 n (fit 2, 4, 8) -> 16: "
      f"{f(16):.0f} ms vs {Pd[16]:.0f} measured ({100*(f(16)-Pd[16])/Pd[16]:+.1f}%)")
g, s2 = fit3((1, 2, 4), (Tv_pod[1], Tv_pod[2], Tv_pod[4]))
print(f"VAE pod: T_v = {s2[0]:.0f}/n + {s2[1]:.0f} + {s2[2]:.0f} log2 n (fit 1, 2, 4) -> 8: {g(8):.0f} ms vs {Tv_pod[8]} measured "
      f"({100*(g(8)-Tv_pod[8])/Tv_pod[8]:+.1f}%)")
