# Autoregressive video diffusion with torch.compile: a rolling KV cache for live world models on AWS Trainium

The question is no longer whether a video diffusion model compiles, but whether an
autoregressive one does. World models such as LingBot-World generate video chunk by chunk, and
each chunk attends a KV cache of earlier chunks, a fixed sink plus a rolling window, so the
scene stays consistent and the stream can run without end. In this talk, we present our work
serving a 14B autoregressive world model as a live, steerable stream on AWS Trainium with
torch.compile and torch.distributed DeviceMesh.

A cache that moves every chunk is hard for a compiled graph. Our first version wrote one flat
buffer at an advancing offset, and the compiled model ran 1.8x slower than eager because the
runtime rebound the graph's inputs on almost every call. We replaced it with a chunked cache:
each chunk is its own tensor, and the choice of which slot to evict is made outside the graph,
so the traced shape never changes. Attention runs on each cached chunk in place, and the
partial results are merged exactly using their row max and sum, instead of copying into a
zero-padded buffer that also skewed the softmax. Tensor and context parallel collectives come
from a DeviceMesh, so they stay inside the compiled graph.

On half of one Trainium3 instance, the stream runs at 17.3 frames per second with about 1.07
seconds from a key press to new frames, and end-to-end generation is 1.93x faster than the
original code. Attendees will leave with a pattern for any autoregressive diffusion model:
fixed-shape cache slots, eviction outside the graph, and per-chunk attention with an exact
merge. We close with a short live demo.
