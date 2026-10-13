"""Static-shape compilation helper for Neuron.

Wraps a module or function as a single Neuron-compiled graph instead of
executing individual ops eagerly. This is most useful when profiling shows
that per-op dispatch and host/device transfer overhead dominate execution
time, rather than the compute itself. Tensor parallelism can amplify this
effect by introducing additional collectives and copies while reducing the
size of individual matmuls.

Compiling at the level of a repeated block is typically the right
granularity: identical blocks share a single compiled graph, while dynamic
parts of the pipeline (e.g. tokenization and per-sequence truncation) remain
outside the compiled region.

`fullgraph=True` traces the entire decorated module as a single graph,
including collective ops such as `all_reduce`. Always validate newly
compiled paths on real hardware—successful compilation does not guarantee
behavior identical to the eager implementation.
"""

import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 128


def compile(mod_or_fn):
    return torch.compile(mod_or_fn, backend="neuron", dynamic=False, fullgraph=True)
