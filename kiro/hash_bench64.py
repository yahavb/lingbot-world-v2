"""Encode-cache key cost with N processes hashing at once, as the 64 ranks do inside GEN.
    python3 kiro/hash_bench64.py [nproc]
Host CPU only. Compares the q5 key (tobytes + crc32 + adler32) with wan.image2video.encode_key
at several LINGBOT_KEY_THREADS, on the image as loaded and on single_block_image(image)."""
import multiprocessing as mp
import os
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
VARIANTS = [("q5 key", 0, False), ("t4 multi-block", 4, False), ("t1 single-block", 1, True),
            ("t2 single-block", 2, True), ("t4 single-block", 4, True), ("t8 single-block", 8, True)]


def worker(i, bar, q):
    from PIL import Image
    sys.path.insert(0, ".")
    from wan.image2video import encode_key, single_block_image
    img = Image.open("examples/03/image.jpg").convert("RGB")
    blk = single_block_image(img)
    res = {}
    for name, nt, single in VARIANTS:
        os.environ["LINGBOT_KEY_THREADS"] = str(nt)
        im = blk if single else img
        ts = []
        for _ in range(4):
            bar.wait()
            t0 = time.perf_counter()
            k = encode_key(im, 81, 480, 768)
            ts.append((time.perf_counter() - t0) * 1e3)
        res[name] = (min(ts[1:]), max(ts[1:]), k if i == 0 else None)
    q.put((i, res))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    bar = mp.Barrier(N)
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(i, bar, q)) for i in range(N)]
    for p in ps:
        p.start()
    out = [q.get() for _ in ps]
    for p in ps:
        p.join()
    for name, _, _ in VARIANTS:
        mins = [r[name][0] for _, r in out]
        maxs = [r[name][1] for _, r in out]
        key = [r[name][2] for _, r in out if r[name][2] is not None][0]
        ks = str(key)
        print(f"{name:16s} {N} procs: fastest proc {min(mins):6.1f} ms, slowest proc {max(mins):6.1f} ms"
              f" (worst rep {max(maxs):6.1f} ms)  key {ks[:90]}{'...' if len(ks) > 90 else ''}")
