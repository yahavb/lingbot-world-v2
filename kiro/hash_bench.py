"""Cost of the encode-cache key on the job's conditioning image (host CPU only, no device).
    python3 kiro/hash_bench.py [image]"""
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "examples/03/image.jpg"
img = Image.open(path).convert("RGB")
print("image", img.size, img.mode, "blocks", img.im.isblock())


def t(name, fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    print(f"{name:28s} min {min(ts):7.2f} ms  med {sorted(ts)[len(ts) // 2]:7.2f} ms")
    return r


buf = t("tobytes", lambda: img.tobytes())
mv = memoryview(buf)
t("crc32", lambda: zlib.crc32(mv))
t("adler32", lambda: zlib.adler32(mv))
t("key (tobytes+crc+adler)", lambda: (lambda b: (zlib.crc32(b), zlib.adler32(b)))(memoryview(img.tobytes())))
for nt in (4, 8, 16):
    pool = ThreadPoolExecutor(nt)

    def par(b=mv, nt=nt, pool=pool):
        step = -(-b.nbytes // nt)
        parts = [b[i:i + step] for i in range(0, b.nbytes, step)]
        return tuple(pool.map(zlib.crc32, parts)) + tuple(pool.map(zlib.adler32, parts))
    t(f"crc+adler {nt} threads", par)
    pool.shutdown()

try:
    import pyarrow as pa

    def arrow_buf():
        a = pa.array(img)
        return a.type, len(a), [None if b is None else b.size for b in a.buffers()]
    print("arrow", t("pa.array(img)", arrow_buf))
    a = pa.array(img)
    vb = [b for b in a.buffers() if b is not None][-1]
    amv = memoryview(vb)
    print("arrow buffer bytes", amv.nbytes, "vs tobytes", mv.nbytes)
    t("arrow crc32", lambda: zlib.crc32(amv))
    # pixel check: the arrow values hold RGBX, compare RGB to tobytes
    arr = np.frombuffer(amv, np.uint8)
    print("arrow len/4", arr.size // 4, "pixels", img.size[0] * img.size[1])
    rgb = arr.reshape(-1, 4)[:, :3].reshape(-1)
    print("RGB equal tobytes", np.array_equal(rgb, np.frombuffer(mv, np.uint8)),
          "X byte values", np.unique(arr.reshape(-1, 4)[:, 3])[:5])
except Exception as e:  # noqa: BLE001
    print("arrow failed:", type(e).__name__, e)
