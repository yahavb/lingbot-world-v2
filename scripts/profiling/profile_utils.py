"""Profiling utilities for torch.profiler + NeuronProfiler on Neuron.

Provides:
- setup_env(): set Neuron profiling env vars (call before import torch)
- create_profiler(): build NeuronConfig + NeuronProfiler
- profile_call(): warm up, then run one profiled steady-state call
- print_summary(): curated CPU table with key metrics
- copy_neffs(): colocate NEFFs with the NTFF for Explorer ingestion
- print_explorer_path(): print the path to upload to Neuron Explorer
"""
import os
import glob
import re
import shutil
import subprocess

# Instance-type prefix -> neuronx-cc platform target. Ordered longest-first
# so 'trn3-dev1...' can't match a shorter prefix by accident.
_PLATFORM_PREFIXES = ('trn3', 'trn2', 'trn1', 'inf2', 'inf1')


def detect_platform(default='trn2'):
    """Return the neuronx-cc platform target for the instance we're on.

    Reads `neuron-ls`'s `instance-type:` line rather than assuming a
    target. Hardcoding this is a correctness trap: compiling with
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 on a trn3 box produces a
    trn2-targeted graph that still *runs* on trn3, so the profile looks
    valid while measuring the wrong hardware (and silently forgoes
    NeuronCore-v4). Falls back to `default` if neuron-ls is unavailable.
    """
    try:
        out = subprocess.run(
            ['neuron-ls'], capture_output=True, text=True, timeout=15).stdout
        m = re.search(r'^instance-type:\s*(\S+)', out, re.M)
        if m:
            instance_type = m.group(1).lower()
            for platform in _PLATFORM_PREFIXES:
                if instance_type.startswith(platform):
                    return platform
    except (OSError, subprocess.SubprocessError):
        pass
    return default


def setup_env(output_dir, visible_core=None, platform=None):
    """Set Neuron profiling env vars. Must be called BEFORE import torch.

    Leave `visible_core` None in a multi-rank launch: restricting
    visibility per-rank collides with `torch.neuron.set_device(local_rank)`
    picking the original core index. Use `capture_core` on
    `create_profiler`/`profile_call` instead.

    `platform` pins the neuronx-cc target ('trn2', 'trn3', ...). Leave it
    None to detect from the instance (see `detect_platform`); an existing
    NEURON_PLATFORM_TARGET_OVERRIDE in the environment always wins, so a
    caller can still force a cross-target compile deliberately.
    """
    if 'NEURON_PLATFORM_TARGET_OVERRIDE' not in os.environ:
        os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = (
            platform or detect_platform())
    if visible_core is not None:
        os.environ['NEURON_RT_VISIBLE_CORES'] = str(visible_core)
    os.environ['NEURON_RT_INSPECT_ENABLE'] = '1'
    os.environ['NEURON_RT_INSPECT_DEVICE_PROFILE'] = '1'
    os.environ['NEURON_RT_INSPECT_OUTPUT_DIR'] = output_dir
    os.environ['XLA_IR_DEBUG'] = '1'
    os.environ['XLA_HLO_DEBUG'] = '1'
    os.environ['NEURON_FRAMEWORK_DEBUG'] = '1'
    # Print the resolved target -- a profile captured against the wrong
    # platform is the kind of mistake that's invisible in the numbers.
    print(f"[profile_utils] platform target: "
          f"{os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']}", flush=True)


def create_profiler(output_dir, capture_core='0'):
    """Create NeuronConfig and NeuronProfiler. Call AFTER import torch.

    Pass the rank's `local_rank` as `capture_core` in a multi-rank launch
    so each rank captures its own core instead of everyone capturing core 0.
    """
    try:
        from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler
    except ImportError as exc:
        raise RuntimeError(
            "torch_neuronx.profiling is not available -- --profile requires "
            "it (see scripts/profiling/profile_utils.py)."
        ) from exc

    nc = NeuronConfig(
        modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
        profile_output_dir=output_dir,
        capture_enabled_for_nc=str(capture_core),
    )
    exporter = NeuronProfiler(nc)
    return nc, exporter


def profile_call(fn, output_dir, label="", warmup=1, capture_core='0'):
    """Warm up `fn` `warmup` times above/outside the profiler, then run it
    once more inside a torch.profiler capture to measure steady state. All
    calls run under `torch.no_grad()` (mixing grad-enabled and no_grad calls
    forces Dynamo to recompile, which would corrupt the measurement).
    Prints the summary table and the Explorer upload path.

    Pass `warmup=0` if the caller already warmed up `fn` itself (e.g. to
    keep other torchrun ranks in lockstep).

    `capture_core`: see `create_profiler`. Returns the last call's result.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity

    for _ in range(warmup):
        with torch.no_grad():
            fn()
        torch.neuron.synchronize()

    nc, exporter = create_profiler(output_dir, capture_core=capture_core)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        experimental_config=nc,
        on_trace_ready=exporter.export_trace,
        record_shapes=False,  # adds CPU overhead
    ) as prof:
        with torch.no_grad():
            result = fn()
        torch.neuron.synchronize()

    print_summary(prof, label)
    copy_neffs(output_dir)
    print_explorer_path(output_dir)
    return result


def print_summary(prof, label=""):
    """Print curated profiling summary: key metrics + top entries by category."""
    ka = prof.key_averages()
    total_us = sum(e.cpu_time_total for e in ka)
    if total_us == 0:
        print("No profiling data captured.")
        return

    # Classify entries
    sleep_us = 0
    alloc_count = 0
    alloc_us = 0
    copy_count = 0
    copy_us = 0
    other_entries = []

    for e in ka:
        key = e.key.lower()
        if 'sleep' in key:
            sleep_us = max(sleep_us, e.cpu_time_total)
        elif 'alloc' in key:
            alloc_count += e.count
            alloc_us += e.cpu_time_total
        elif 'copy' in key:
            copy_count += e.count
            copy_us += e.cpu_time_total
        else:
            other_entries.append(e)

    other_entries.sort(key=lambda e: e.cpu_time_total, reverse=True)

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    sleep_pct = sleep_us / total_us * 100
    print(f"\n  Total: {total_us/1000:.0f}ms | Sleep: {sleep_pct:.0f}%"
          f" | Allocs: {alloc_count} ({alloc_us/1000:.1f}ms)"
          f" | Copies: {copy_count} ({copy_us/1000:.1f}ms)")

    # Table -- always show sleep + alloc + copy + top 4 others = 7 rows max
    print(f"\n  {'Entry':<40} {'CPU Total':>10} {'Calls':>7} {'% Total':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*7} {'-'*8}")

    print(f"  {'RuntimeWorker::sleep':<40} {sleep_us/1000:>8.1f}ms {'—':>7} {sleep_pct:>7.1f}%")

    if alloc_count > 0:
        alloc_pct = alloc_us / total_us * 100
        print(f"  {'neuron::alloc::lazy (total)':<40} {alloc_us/1000:>8.1f}ms {alloc_count:>7} {alloc_pct:>7.1f}%")

    if copy_count > 0:
        copy_pct = copy_us / total_us * 100
        print(f"  {'aten::copy_ + neuron::copy (total)':<40} {copy_us/1000:>8.1f}ms {copy_count:>7} {copy_pct:>7.1f}%")

    shown = 0
    for e in other_entries:
        if shown >= 4:
            break
        name = e.key[:40]
        pct = e.cpu_time_total / total_us * 100
        if pct < 0.5:
            break
        print(f"  {name:<40} {e.cpu_time_total/1000:>8.1f}ms {e.count:>7} {pct:>7.1f}%")
        shown += 1

    print()


def copy_neffs(output_dir):
    """Colocate NEFFs with the NTFF so Explorer can resolve every UUID it
    references. `NEURON_RT_INSPECT_DEVICE_PROFILE` splits the warmup call
    and the profiled call into separate device-profile sessions even
    though they run the same compiled graph, so most NEFFs only get
    written to the warmup session's directory -- copy them over to the
    session that actually holds the NTFF. Only looks at sibling session
    dirs under this same output_dir (same process, same run), never the
    shared /tmp/neff_cache, so it can't pull in NEFFs from another rank.
    """
    ntff_dirs = []
    neff_dirs = []
    for ts_dir in glob.glob(f'{output_dir}/i-*_pid_*/*'):
        if not os.path.isdir(ts_dir):
            continue
        files = os.listdir(ts_dir)
        if any(f.endswith('.ntff') for f in files):
            ntff_dirs.append(ts_dir)
        if any(f.endswith('.neff') for f in files):
            neff_dirs.append(ts_dir)

    for ntff_dir in ntff_dirs:
        have = {f for f in os.listdir(ntff_dir) if f.endswith('.neff')}
        for neff_dir in neff_dirs:
            if neff_dir == ntff_dir:
                continue
            for f in os.listdir(neff_dir):
                if f.endswith('.neff') and f not in have:
                    shutil.copy2(os.path.join(neff_dir, f), ntff_dir)
                    have.add(f)


def print_explorer_path(output_dir):
    """Print the path to upload to Explorer."""
    dirs = glob.glob(f'{output_dir}/i-*_pid_*')
    if dirs:
        print(f"  Explorer upload: {dirs[0]}")
