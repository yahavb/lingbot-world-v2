"""Chunked KV cache: N independent per-chunk buffers instead of one windowed buffer.

Ported from the reference implementation `reference kv_cache.py` + `reference setup.py`. Two comments in
that code are the whole reason this file exists, and both are about the Neuron
*runtime*, not Dynamo:

    kv_cache.py:100
      "Independent base tensors allow direct must-alias output binding. Using offset
       views of a larger parent requires output epilogue/copy handling."

    setup.py:225
      "Compiling a block-stack per cache write destination to avoid an expensive IO
       rebind when reusing a single NEFF"

The flat cache this replaces is a single `[B, window, N, D]` buffer written at an
integer offset that advances every chunk. Reusing one compiled NEFF while that offset
moves makes the runtime rebind the NEFF's I/O on essentially every call.

That cost is invisible to a single-forward replay -- one destination, bound once, never
rebound -- and it dominates a real denoise loop, where the destination moves every
chunk. It is why `profile_dit_from_captured.py` measured compiled at 3x FASTER per
forward while e2e GEN measured compiled at 1.8x SLOWER than eager, and why nothing
aimed at the graph count moved the number: prewarming all positions, pinning the flat
buffers with mark_static_address, and turning the offsets into plain ints all still
reuse ONE NEFF across every destination.

Here a chunk is a whole tensor written at its own index 0, so no integer offset ever
enters the graph.
"""

import torch


def num_chunks(local_attn_size, chunk_size):
    """the reference implementation setup.py:123 -- `n_chunks = kv.window_size // kv.chunk_size`.

    Both are in latent frames. Returns 0 when the cache is global
    (`local_attn_size == -1`), which has no chunk structure to speak of.
    """
    if local_attn_size is None or local_attn_size < 0:
        return 0
    return max(1, int(local_attn_size) // max(1, int(chunk_size)))


def chunk_routing(chunk_index, n_chunks):
    """`(destination chunk, leading chunks)` for the write about to happen.

    the reference implementation `kv_cache.py:165`. Chunk 0 is the pinned sink -- written once and never a
    destination again; the rest round-robin. Its docstring publishes a table for
    window_size=12 / chunk_size=3 (n_chunks=4), which this reproduces exactly:

        i=0   (0, ())          1 chunk in play
        i=1   (1, (0,))
        i=2   (2, (0, 1))
        i=3   (3, (0, 1, 2))
        i=4   (1, (0, 2, 3))   steady state: always n_chunks-1 leading, so the
        i=5   (2, (0, 1, 3))   traced shape is identical whichever slot rotates out

    the reference implementation derives `dest` from a `_start_slot()` helper in its SDK, which is
    not vendored in their repo; this reproduces the published table arithmetically
    instead. Verified against all six documented rows.

    The order is `(leading..., destination)` with the destination LAST. Attention is
    order-invariant here and RoPE was applied before anything was cached, so the order
    matters only for tracing the same tuple layout every call -- but the destination's
    position is load-bearing, because that is the entry the write goes to.
    """
    i = int(chunk_index)
    if n_chunks <= 1:
        return 0, ()
    dest = 0 if i == 0 else 1 + (i - 1) % (n_chunks - 1)
    written = min(i, n_chunks)
    leading = tuple(j for j in range(written) if j != dest)
    return dest, leading


def allocate_self_kv_chunks(num_layers, batch, chunk_tokens, heads, head_dim,
                            n_chunks, dtype, device, n_major=False):
    """Per block, `n_chunks` INDEPENDENT K and V buffers.

    Layout is `[B, chunk_tokens, N, D]`, or `(N, chunk_tokens, D)` when `n_major`.

    Independent base tensors, not views of one parent -- see the module docstring.
    Every buffer is address-pinned, which is also what makes Dynamo specialise a
    distinct graph per destination ordering: the tuple shape is constant at steady
    state, so tensor identity is the only thing distinguishing one routing from
    another.
    """
    # PER-LAYER layout. n_major may be a bool (every layer the same, the old behaviour)
    # or a per-layer sequence, because hybrid per-layer attention gives a hot layer the
    # exact-attention path, which reads [B, S, N, D], while its cold neighbours feed the
    # const-max kernel and need N-major. Allocating one layout for the whole stack would
    # hand one of the two the wrong shape.
    if isinstance(n_major, (bool, int)):
        n_major_by_layer = [bool(n_major)] * num_layers
    else:
        n_major_by_layer = [bool(f) for f in n_major]
        if len(n_major_by_layer) != num_layers:
            raise ValueError(
                f"n_major has {len(n_major_by_layer)} entries for {num_layers} layers")

    cache = []
    for layer in range(num_layers):
        n_major = n_major_by_layer[layer]
        k_chunks, v_chunks = [], []
        # N-MAJOR, 3-D, for the const-max AG kernel: the reference implementation's preallocate_chunked logs
        # "chunked, N-major". The NKI kernel reads `n_heads, _, d = k_shard.shape` and
        # allocates its gather buffer as (n_heads, kv_chunk, d), so a 4-D [B, S, N, d]
        # chunk leaves the batch dim unconsumed and the kernel refuses to compile:
        #   NDSlice.stream: requires all batch dims to be consumed first
        #   (got 1 unconsumed)
        # batch is 1 on this path anyway -- ag_const_max_attention squeezes it.
        #
        # K IS PRE-TRANSPOSED to (N, d, chunk_tokens) for the const-max AG path, matching
        # the updated kernel's cache layout: "K: [N, d, tokens_per_chunk] (pre-transposed for
        # MM1), V: [N, tokens_per_chunk, d]". The rewritten kernel loads a leading chunk
        # straight into MM1 without a DMA transpose, and reads that shape.
        #
        # THE LAYOUT AND THE KERNEL MOVE TOGETHER -- this note previously recorded the
        # transpose as REVERTED because, against the OLD kernel, it rendered uniform frames
        # (mean_entropy 0.0, mean_frame_diff 0.0). That pairing was the problem, not the
        # transpose: kernels/ now holds the updated kernel, which requires it. Reverting one
        # without the other is what breaks, in either direction. If frames go uniform again,
        # check that kernels/attention_const_max_allgather.py is still the updated version
        # before touching this shape.
        #
        # Only the N-major (const-max) branch changes. The exact-attention branch keeps
        # [B, S, N, D] because attention_cte reads that layout, and hybrid per-layer
        # attention means both exist in the same stack.
        k_shape = ((heads, head_dim, chunk_tokens) if n_major
                   else (batch, chunk_tokens, heads, head_dim))
        v_shape = ((heads, chunk_tokens, head_dim) if n_major
                   else (batch, chunk_tokens, heads, head_dim))
        for _ in range(n_chunks):
            k = torch.zeros(*k_shape, dtype=dtype, device=device)
            v = torch.zeros(*v_shape, dtype=dtype, device=device)
            for t in (k, v):
                try:
                    torch._dynamo.mark_static_address(t)
                except Exception:
                    pass
            k_chunks.append(k)
            v_chunks.append(v)
        cache.append({"k": k_chunks, "v": v_chunks})
    return cache


def ordered_chunks(block_cache, destination, leading):
    """`(leading..., destination)` K/V tuples for one block. the reference implementation kv_cache.py:135."""
    order = (*leading, destination)
    return (tuple(block_cache["k"][i] for i in order),
            tuple(block_cache["v"][i] for i in order))


def ordered_chunks_by_block(cache, destination, leading):
    """The same, for every block in model order. the reference implementation kv_cache.py:153."""
    pairs = tuple(ordered_chunks(bc, destination, leading) for bc in cache)
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def chunk_bytes(cache):
    """Total bytes held by a chunked cache, for the HBM banner."""
    total = 0
    for bc in cache:
        for key in ("k", "v"):
            for t in bc[key]:
                total += t.numel() * t.element_size()
    return total
