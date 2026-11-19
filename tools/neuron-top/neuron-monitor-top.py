#!/usr/bin/env python3

# Copyright 2021 - 2022, Amazon.com, Inc. or its affiliates. All Rights Reserved

import argparse
import bisect
import curses
import glob
import json
import math
import os
import re
import select
import signal
import sys #debug only
import threading
import time
import queue

from functools import total_ordering

NEURON_TOP_VERSION="${VERSION}"
# Show each line of error message in own one-line label
DRIVER_ERR_MSG=["Error: Unable to read device information from the driver",
                "For troubleshooting guide:",
                "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-troubleshoot.html#neuron-driver-installation-fails"]

class Const:
    SCREEN_MARGIN = 1
    MAX_MODEL_NAME_LEN = 40
    MEM_USAGE_FIELD_NAMES_MODEL = [['tensors', 'constants', 'dma_buffers', 'application_memory'],
                                   ['tensors', 'constants', 'model_code', 'runtime_memory']]
    MEM_USAGE_FIELD_NAMES = [MEM_USAGE_FIELD_NAMES_MODEL[0],
                             MEM_USAGE_FIELD_NAMES_MODEL[1] + ['model_shared_scratchpad']]
    MEM_USAGE_FIELD_TITLES = {
        'dma_buffers': 'DMA Buffers',
        'application_memory': 'App. Memory',
        'model_shared_scratchpad': 'Model Scratchpad'
    }
    METRIC_GROUP_NAMES = ('neuroncore_counters', 'neuron_runtime_vcpu_usage', 'memory_used')
    SYSTEM_METRIC_GROUP_NAMES = ('memory_info', 'vcpu_usage')
    NEURON_NODES_REGEX = re.compile(r'.*neuron([0-9]+)$')

    @classmethod
    def field_name_to_field_title(cls, field_name):
        if field_name in cls.MEM_USAGE_FIELD_TITLES:
            return cls.MEM_USAGE_FIELD_TITLES[field_name]
        names = field_name.split('_')
        return ' '.join([name.capitalize() for name in names])


class NeffContext:
    """ Finds the NEFF file behind each loaded model (by the UUID in the NEFF header) and reads the
        context the compiler stored in it: HLO ops, the Python source lines they came from, and the
        input count. Parsing runs on a background thread so large NEFFs never stall the UI.
        Cache locations come from the environment of neuron-top and of every process using a
        Neuron device; extra directories can be given in NEURON_TOP_NEFF_DIRS (colon separated).
    """
    HEADER_SIZE = 1024
    UUID_START, UUID_END = 204, 220
    # Default NEFF locations used by torch-neuronx and neuronx-cc. Never '/' or '~': walking those is slow.
    DEFAULT_DIRS = ('/tmp/neff_cache', '/var/tmp/neuron-compile-cache', '/tmp/neuron-compile-cache')
    RESCAN_INTERVAL_S = 10
    MISSING = 'missing'
    # A NEFF can appear after neuron-top first looks (the app writes it, or starts later):
    # re-check models without a NEFF file every MISSING_RETRY_S, at most MISSING_RETRIES times
    MISSING_RETRY_S = 30
    MISSING_RETRIES = 10
    # reading a known NEFF can fail while it is still being written: retry it, a few times
    READ_RETRIES = 5
    # Ops that appear everywhere and say little about what a graph does
    GENERIC_OPS = {'reshape', 'convert', 'broadcast', 'copy', 'transpose', 'bitcast', 'parameter',
                   'constant', 'tuple', 'get-tuple-element'}

    # Environment variables that point at NEFF caches, read from neuron-top's own environment and
    # from every local process that has a Neuron device open (the cache is set where the app runs)
    CACHE_ENV_VARS = ('NEURON_COMPILE_CACHE_URL', 'TORCH_NEURONX_NEFF_CACHE_DIR',
                      'NEURON_LIBTORCH_CACHE_ROOT', 'VLLM_CACHE_ROOT')

    def __init__(self):
        self.extra_dirs = [d for d in os.environ.get('NEURON_TOP_NEFF_DIRS', '').split(':') if d]
        self.dirs = self._search_dirs()
        self.paths = {}       # uuid -> NEFF file path
        self.contexts = {}    # uuid -> context dict, or MISSING
        self.queued = set()
        self.first_seen = {}  # uuid -> time it was first requested
        self.missing = {}     # uuid -> (time marked MISSING, times marked MISSING)
        self.read_failures = {}  # uuid -> failed reads of a known NEFF file
        self.last_scan = 0
        self.lock = threading.Lock()
        self.work = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def get(self, uuid):
        """ Returns the context dict, MISSING if no NEFF file has this UUID, or None while pending. """
        with self.lock:
            ctx = self.contexts.get(uuid)
            retry = False
            if ctx == self.MISSING and uuid not in self.queued:
                since, count = self.missing.get(uuid, (0, 0))
                retry = count < self.MISSING_RETRIES and time.time() - since >= self.MISSING_RETRY_S
                if retry:
                    self.first_seen[uuid] = time.time()   # need a scan newer than this retry
            if (ctx is None or retry) and uuid not in self.queued:
                self.first_seen.setdefault(uuid, time.time())
                self.queued.add(uuid)
                self.work.put(uuid)
        return ctx

    def _worker(self):
        while True:
            uuid = self.work.get()
            ctx = None  # None = not decided yet, ask again on a later refresh
            try:
                if uuid not in self.paths and time.time() - self.last_scan > self.RESCAN_INTERVAL_S:
                    self._scan()
                path = self.paths.get(uuid)
                if path:
                    try:
                        ctx = self._read(path)
                    except Exception:  # e.g. still being written: stay pending, try again later
                        failures = self.read_failures.get(uuid, 0) + 1
                        self.read_failures[uuid] = failures
                        ctx = self.MISSING if failures >= self.READ_RETRIES else None
                elif self.last_scan >= self.first_seen.get(uuid, 0):
                    # a scan that started after this model appeared did not find its NEFF
                    ctx = self.MISSING
            except Exception:  # scan failure: fall back to the plain name, keep the thread alive
                ctx = self.MISSING
            with self.lock:
                if ctx is not None:
                    self.contexts[uuid] = ctx
                    if ctx == self.MISSING:
                        self.missing[uuid] = (time.time(), self.missing.get(uuid, (0, 0))[1] + 1)
                self.queued.discard(uuid)

    @classmethod
    def _dirs_from_env(cls, env):
        """ NEFF directories implied by one environment (vllm-neuron / libtorch-neuronx layout:
            <root>/neuron/native_neff_cache, with root = NEURON_LIBTORCH_CACHE_ROOT or VLLM_CACHE_ROOT). """
        dirs = []
        for var in ('NEURON_COMPILE_CACHE_URL', 'TORCH_NEURONX_NEFF_CACHE_DIR'):
            value = env.get(var, '')
            if value and '://' not in value:
                dirs.append(value)
        for var in ('NEURON_LIBTORCH_CACHE_ROOT', 'VLLM_CACHE_ROOT'):
            root = env.get(var, '')
            if root and '://' not in root:
                dirs.append(os.path.join(root, 'neuron'))
        home = env.get('HOME', '')
        if home:
            dirs.append(os.path.join(home, '.cache', 'neuron_libtorch'))
        return dirs

    @classmethod
    def _neuron_process_envs(cls):
        """ Environments of local processes that have a /dev/neuron* device open. """
        envs = []
        for proc in glob.glob('/proc/[0-9]*'):
            try:
                if not any(os.readlink(fd).startswith('/dev/neuron') for fd in glob.glob(proc + '/fd/*')):
                    continue
                with open(proc + '/environ', 'rb') as f:
                    raw = f.read().split(b'\0')
            except OSError:
                continue
            env = {}
            for item in raw:
                key, sep, value = item.partition(b'=')
                if sep and key.decode('utf-8', 'replace') in cls.CACHE_ENV_VARS + ('HOME', ):
                    env[key.decode('utf-8', 'replace')] = value.decode('utf-8', 'replace')
            if env not in envs:
                envs.append(env)
        return envs

    def _search_dirs(self):
        dirs = list(self.DEFAULT_DIRS) + self.extra_dirs + self._dirs_from_env(os.environ)
        for env in self._neuron_process_envs():
            dirs += self._dirs_from_env(env)
        dirs += glob.glob('/tmp/neuron_compile_wdir_*')
        return [d for d in dict.fromkeys(dirs) if d and d not in ('/', os.path.expanduser('~'))]

    def _scan(self):
        self.last_scan = time.time()
        self.dirs = self._search_dirs()   # apps may start after neuron-top
        for d in self.dirs:
            for path in glob.glob(os.path.join(d, '**', '*.neff'), recursive=True):
                try:
                    with open(path, 'rb') as f:
                        hdr = f.read(self.HEADER_SIZE)
                except OSError:
                    continue
                if len(hdr) == self.HEADER_SIZE:
                    self.paths.setdefault(hdr[self.UUID_START:self.UUID_END].hex(), path)

    def _read(self, path):
        import collections
        import tarfile
        ops, sources, inputs, input_bytes = collections.Counter(), collections.Counter(), 0, 0
        with open(path, 'rb') as f:
            f.seek(self.HEADER_SIZE)
            # streaming mode: one pass, never loads the whole NEFF into memory
            with tarfile.open(fileobj=f, mode='r|gz') as tf:
                for m in tf:
                    name = os.path.basename(m.name)
                    if not m.isfile():
                        continue
                    if name == 'neff.json':
                        sig = json.load(tf.extractfile(m)).get('metadata', {}).get('signatures', {})
                        sig_inputs = sig.get('default', {}).get('inputs', {}).values()
                        inputs = len(sig_inputs)
                        input_bytes = sum(self.tensor_bytes(v) for v in sig_inputs)
                    elif name.startswith('debug_info_hlo'):
                        for op in self._hlo_ops(tf.extractfile(m).read()):
                            kind = op['instr'].split('=', 1)[1].strip().split('(', 1)[0]
                            if kind in ('parameter', 'constant', 'tuple', 'get-tuple-element'):
                                continue
                            ops[kind] += 1
                            src = op.get('source_file')
                            if src and not src.rstrip(':').endswith('.mlir'):
                                sources[(src.rstrip(':'), op.get('source_line', '?'))] += 1
        return {'ops': ops, 'sources': sources, 'inputs': inputs, 'input_bytes': input_bytes}

    DTYPE_BYTES = {'float64': 8, 'int64': 8, 'uint64': 8, 'float32': 4, 'int32': 4, 'uint32': 4,
                   'bfloat16': 2, 'float16': 2, 'int16': 2, 'uint16': 2}

    @classmethod
    def tensor_bytes(cls, tensor):
        count = 1
        for dim in tensor.get('shape', []):
            count *= dim
        return count * cls.DTYPE_BYTES.get(tensor.get('dtype'), 1)

    @classmethod
    def label(cls, ctx, name):
        """ Row label for a model, e.g. 'modeling_x.py:129 · sdpa_attention · 158 inputs' or 'op: slice'. """
        if ctx is None:
            return name
        if ctx == cls.MISSING:
            return '%s · loaded from memory, no NEFF file found' % (name, )
        if ctx['sources']:
            (top_file, top_line), _ = ctx['sources'].most_common(1)[0]
            parts = ['%s:%s' % (os.path.basename(top_file), top_line)]
            for (src_file, _), _ in ctx['sources'].most_common():
                stem = os.path.splitext(os.path.basename(src_file))[0]
                if src_file != top_file and stem not in parts and len(parts) < 3:
                    parts.append(stem)
            if ctx['inputs']:
                parts.append('%d inputs' % (ctx['inputs'], ))
            return ' · '.join(parts)
        if ctx['ops']:
            specific = [op for op, _ in ctx['ops'].most_common() if op not in cls.GENERIC_OPS]
            chosen = (specific or [op for op, _ in ctx['ops'].most_common()])[:3]
            return 'op: ' + ', '.join(chosen)
        return name

    @staticmethod
    def has_user_source(ctx):
        return isinstance(ctx, dict) and bool(ctx['sources'])

    # --- minimal schema-less protobuf reader; the debug_info files are protobuf ---
    @staticmethod
    def _varint(b, i):
        r = s = 0
        while True:
            c = b[i]
            i += 1
            r |= (c & 0x7f) << s
            s += 7
            if c < 0x80:
                return r, i

    @classmethod
    def _pb_fields(cls, b):
        """ Returns [(field_no, value)], or None if b does not parse as a protobuf message. """
        out, i = [], 0
        try:
            while i < len(b):
                key, i = cls._varint(b, i)
                fn, wt = key >> 3, key & 7
                if fn == 0:
                    return None
                if wt == 0:
                    v, i = cls._varint(b, i)
                elif wt == 1:
                    v, i = b[i:i + 8], i + 8
                elif wt == 2:
                    n, i = cls._varint(b, i)
                    v, i = b[i:i + n], i + n
                elif wt == 5:
                    v, i = b[i:i + 4], i + 4
                else:
                    return None
                if i > len(b):
                    return None
                out.append((fn, v))
        except IndexError:
            return None
        return out

    @staticmethod
    def _text(v):
        if isinstance(v, bytes):
            try:
                return v.decode('utf-8')
            except UnicodeDecodeError:
                return None
        return None

    @classmethod
    def _scalar(cls, v):
        if isinstance(v, int):
            return v
        sub = cls._pb_fields(v) if isinstance(v, bytes) else None
        if sub:
            for _, x in sub:
                s = cls._scalar(x)
                if s not in (None, ''):
                    return s
        return cls._text(v)

    @classmethod
    def _hlo_ops(cls, b):
        """ Yields {instr, op_type, source_file, source_line} for each HLO instruction. """
        fields = cls._pb_fields(b)
        if not fields:
            return
        attrs, instr = {}, None
        for _, v in fields:
            t = cls._text(v)
            if t is not None and '=' in t and t.lstrip().startswith('%'):
                instr = t.strip()
            sub = cls._pb_fields(v) if isinstance(v, bytes) else None
            if sub and len(sub) == 2 and cls._text(sub[0][1]) in ('op_type', 'source_file', 'source_line'):
                attrs[cls._text(sub[0][1])] = cls._scalar(sub[1][1])
            elif sub:
                yield from cls._hlo_ops(v)
        if instr:
            yield dict(instr=instr, **attrs)


class OpAttribution:
    """ Splits a NEFF's instruction memory (Model Code) by HLO op and Python source line.
        Every engine instruction is fixed width and listed in order in the asm debug info, so its
        bytes can be followed down the compiler's debug info chain:
            asm instruction -> backend (BIR) -> penguin -> HLO op
        Compiler-made penguin steps (tile loads, copies) have no HLO op; they take the op of the
        step that consumes their output, found by following data-flow edges backwards.
        The result is a share of instruction memory, not of compute time. Host CPU only.
    """
    ENGINES = ('PE', 'Activation', 'Pool', 'DVE', 'SP')
    TOP_ROWS = 12
    # NEFF members needed: per-subgraph asm/backend debug info, and the shared upper levels
    MEMBER_RE = re.compile(r'^(?:\./)?(sg[0-9]+|debug_info)/(debug_info_[A-Za-z0-9_.]+)$')
    BIN_RE = re.compile(r'^(?:\./)?(sg[0-9]+)/(PE|Activation|Pool|DVE|SP)[0-9]+\.bin$')

    @staticmethod
    def _varint(b, i):
        r = s = 0
        while True:
            c = b[i]
            i += 1
            r |= (c & 0x7f) << s
            s += 7
            if c < 0x80:
                return r, i

    @classmethod
    def _fields(cls, b):
        i, n = 0, len(b)
        while i < n:
            k, i = cls._varint(b, i)
            wt = k & 7
            if wt == 0:
                v, i = cls._varint(b, i)
            elif wt == 2:
                ln, i = cls._varint(b, i)
                v, i = b[i:i + ln], i + ln
            elif wt == 1:
                v, i = None, i + 8
            elif wt == 5:
                v, i = None, i + 4
            else:
                raise ValueError('unexpected protobuf wire type %d' % (wt, ))
            yield k >> 3, wt, v

    @classmethod
    def _ids(cls, v):
        if isinstance(v, int):
            return [v]
        out, i = [], 0
        while i < len(v):
            x, i = cls._varint(v, i)
            out.append(x)
        return out

    @classmethod
    def _instructions(cls, data, want=(3,)):
        """ Yields (id, name, {field: ids}) for each instruction of an IrDebugInfo message.
            IrDebugInfo: 2 = parent debug info filename, 3 = instructions.
            Instruction: 1 = id, 2 = name, 3 = parent ids, 4 = data-flow predecessors, 7 = attrs. """
        for fn, wt, v in cls._fields(data):
            if fn != 3 or wt != 2:
                continue
            iid, name, refs, attrs = None, None, {}, None
            for ifn, iwt, iv in cls._fields(v):
                if ifn == 1 and iwt == 0:
                    iid = iv
                elif ifn == 2 and iwt == 2:
                    name = iv
                elif ifn in want and ifn != 7 and iv is not None:
                    try:
                        refs.setdefault(ifn, []).extend(cls._ids(iv))
                    except IndexError:
                        pass
                elif ifn == 7 and iwt == 2 and 7 in want:
                    key = val = None
                    for afn, awt, av in cls._fields(iv):
                        if afn == 1 and awt == 2:
                            key = av.decode('utf-8', 'replace')
                        elif afn == 2 and awt == 2:
                            val = av.decode('utf-8', 'replace')
                    if key:
                        attrs = attrs or {}
                        attrs[key] = val
            if attrs:
                refs['attrs'] = attrs
            yield iid, name, refs

    @classmethod
    def _parent_file(cls, data):
        for fn, wt, v in cls._fields(data):
            if fn == 2 and wt == 2:
                return v.decode('utf-8', 'replace')
        return None

    @classmethod
    def compute(cls, neff_path):
        import collections
        import shutil
        import tarfile
        import tempfile
        start = time.time()
        tmp = tempfile.mkdtemp(prefix='neuron-top-ops-')
        bin_bytes = collections.Counter()   # (sg, engine) -> instruction bytes
        try:
            with open(neff_path, 'rb') as f:
                f.seek(NeffContext.HEADER_SIZE)
                with tarfile.open(fileobj=f, mode='r|gz') as tf:
                    for m in tf:
                        if not m.isfile():
                            continue
                        b = cls.BIN_RE.match(m.name)
                        if b:
                            bin_bytes[(b.group(1), b.group(2))] += m.size
                            continue
                        d = cls.MEMBER_RE.match(m.name)
                        if d and 'DMA' not in d.group(2):
                            os.makedirs(os.path.join(tmp, d.group(1)), exist_ok=True)
                            with open(os.path.join(tmp, d.group(1), d.group(2)), 'wb') as out:
                                shutil.copyfileobj(tf.extractfile(m), out)
            return cls._attribute(tmp, bin_bytes, start)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @classmethod
    def _read(cls, path):
        data = b''
        for part in sorted(glob.glob(path + '*')):   # protos over 2GB are split as <name>_<n>
            with open(part, 'rb') as f:
                data += f.read()
        return data

    @classmethod
    def _attribute(cls, root, bin_bytes, start):
        import collections
        upper = {}

        def upper_level(fname, sg_idx, want):
            path = os.path.join(root, 'debug_info', fname)
            if os.path.exists('%s_sg%06d' % (path, sg_idx)):   # written per subgraph; ids collide across them
                path = '%s_sg%06d' % (path, sg_idx)
            if path not in upper:
                data = cls._read(path)
                upper[path] = (cls._parent_file(data),
                               dict((i, (n, r)) for i, n, r in cls._instructions(data, want)))
            return upper[path]

        by_pair, total, mapped = collections.Counter(), 0, 0
        for sg in sorted(glob.glob(os.path.join(root, 'sg[0-9]*'))):
            sg_name = os.path.basename(sg)
            sg_idx = int(sg_name[2:])
            bir_cache, inherited = {}, {}
            for eng in cls.ENGINES:
                asm_path = os.path.join(sg, 'debug_info_asm_%s.dbg' % (eng, ))
                nbytes = bin_bytes.get((sg_name, eng), 0)
                if not nbytes or not glob.glob(asm_path + '*'):
                    continue
                asm_data = cls._read(asm_path)
                asm = [r.get(3, []) for _, _, r in cls._instructions(asm_data)]
                if not asm:
                    continue
                width = float(nbytes) / len(asm)
                total += nbytes
                bir_file = cls._parent_file(asm_data)
                if bir_file not in bir_cache:
                    data = cls._read(os.path.join(sg, bir_file))
                    bir_cache[bir_file] = (cls._parent_file(data),
                                           dict((i, r.get(3, [])) for i, _, r in cls._instructions(data)))
                pen_file, bir = bir_cache[bir_file]
                hlo_file, pen = upper_level(pen_file, sg_idx, (3, 4))
                _, hlo = upper_level(hlo_file, sg_idx, (3, 7)) if hlo_file else (None, {})
                if pen_file not in inherited:
                    inherited[pen_file] = cls._inherit(pen)
                pen_hlo = inherited[pen_file]
                for asm_parents in asm:
                    hlo_ids = []
                    for b_id in asm_parents:
                        for p_id in bir.get(b_id, ()):
                            hlo_ids.extend(pen_hlo.get(p_id, ()))
                    if not hlo_ids:
                        continue
                    share = width / len(hlo_ids)
                    mapped += width
                    for h in hlo_ids:
                        name, refs = hlo.get(h, (None, {}))
                        by_pair[cls._hlo_key(name, refs.get('attrs'))] += share
        rows = [[op, src, int(b)] for (op, src), b in by_pair.most_common(cls.TOP_ROWS)]
        return {'total': total, 'mapped': int(mapped), 'rows': rows, 'seconds': round(time.time() - start, 1)}

    @staticmethod
    def _inherit(pen):
        """ penguin id -> HLO ids, extended backwards along data-flow edges to steps with no HLO op """
        hlo_of = dict((pid, r[3]) for pid, (_, r) in pen.items() if r.get(3))
        frontier = list(hlo_of)
        while frontier:
            nxt = []
            for pid in frontier:
                for pred in pen[pid][1].get(4, ()):
                    if pred in pen and pred not in hlo_of:
                        hlo_of[pred] = hlo_of[pid]
                        nxt.append(pred)
            frontier = nxt
        return hlo_of

    @staticmethod
    def _hlo_key(name, attrs):
        op = '?'
        if name:
            text = name.decode('utf-8', 'replace')
            if '=' in text:
                op = text.split('=', 1)[1].strip().split('(', 1)[0]
        src = ''
        if attrs and attrs.get('source_file'):
            path = attrs['source_file'].rstrip(':')
            if not path.endswith('.mlir'):
                src = '%s:%s' % (os.path.basename(path), attrs.get('source_line', '?'))
        return op, src


class OpAttributionRunner:
    """ Runs OpAttribution in a separate, low-priority process, one NEFF at a time, so neuron-top
        stays small and responsive (a large NEFF takes tens of seconds and about 1GB of host RAM).
    """
    TIMEOUT_S = 600
    # a failed run is retried at most ERROR_RETRIES times, ERROR_RETRY_S apart, then stays failed
    ERROR_RETRIES = 2
    ERROR_RETRY_S = 60
    def __init__(self):
        self.results = {}   # uuid -> ('running', None) | ('done', dict) | ('error', message)
        self.failures = {}  # uuid -> (time of last failure, number of failures)
        self.lock = threading.Lock()
        self.work = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def get(self, uuid):
        with self.lock:
            return self.results.get(uuid)

    def request(self, uuid, path):
        with self.lock:
            if uuid in self.results:
                if self.results[uuid][0] != 'error':
                    return
                last, count = self.failures.get(uuid, (0, 0))
                if count > self.ERROR_RETRIES or time.time() - last < self.ERROR_RETRY_S:
                    return
            self.results[uuid] = ('running', None)
        self.work.put((uuid, path))

    def _worker(self):
        import subprocess
        while True:
            uuid, path = self.work.get()
            try:
                cmd = [sys.executable, os.path.abspath(__file__), '--op-attribution', path]
                if os.path.exists('/usr/bin/nice'):
                    cmd = ['/usr/bin/nice', '-n', '10'] + cmd
                # a stuck child must not block every later request; TimeoutExpired kills it
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      timeout=self.TIMEOUT_S)
                if proc.returncode == 0:
                    result = ('done', json.loads(proc.stdout.decode()))
                else:
                    err = proc.stderr.decode(errors='replace').strip().splitlines()
                    result = ('error', err[-1] if err else 'exit code %d' % (proc.returncode, ))
            except Exception as e:  # keep the worker alive whatever happens
                result = ('error', str(e))
            with self.lock:
                self.results[uuid] = result
                if result[0] == 'error':
                    self.failures[uuid] = (time.time(), self.failures.get(uuid, (0, 0))[1] + 1)


class EfaMonitor:
    """ EFA traffic per device, read from Linux, sampled once a second on a background thread.
        Counters: sysfs ports/1/hw_counters when the driver exposes them, else the RDMA netlink
        statistics (`rdma -j statistic show link`). Also finds which local processes have each EFA
        open (/dev/infiniband/uverbsN) and which Neuron devices sit on the same PCIe switch.
    """
    IB_ROOT = '/sys/class/infiniband'
    COUNTERS = ('tx_bytes', 'rx_bytes', 'rdma_write_bytes', 'rdma_read_bytes', 'retrans_pkts', 'rx_drops')
    PERIOD_S = 1.0
    # Finding which processes have an EFA open reads every fd in /proc; with many large processes
    # (e.g. 64 vLLM ranks) that takes tens of seconds, so it runs on its own thread, less often.
    USERS_PERIOD_S = 15.0

    def __init__(self):
        self.lock = threading.Lock()
        self.devices = self._discover()     # name -> {'pci', 'switch', 'rate_gbps', 'state', 'claimed'}
        self.rows = {}                      # name -> rates and users
        self.source = None                  # 'sysfs' | 'rdma netlink' | None
        self.error = ''
        self.neuron_by_switch = {}
        self.users = {}                     # name -> number of local processes with it open
        if self.devices:
            threading.Thread(target=self._worker, daemon=True).start()
            threading.Thread(target=self._users_worker, daemon=True).start()

    @classmethod
    def _discover(cls):
        devices = {}
        for path in sorted(glob.glob(os.path.join(cls.IB_ROOT, '*'))):
            name = os.path.basename(path)
            driver = os.path.basename(os.path.realpath(os.path.join(path, 'device', 'driver')))
            if driver != 'efa' and not name.startswith(('rdmap', 'efa')):
                continue
            pci = os.path.realpath(os.path.join(path, 'device'))
            rate, state = 0.0, ''
            try:
                with open(os.path.join(path, 'ports', '1', 'rate')) as f:
                    rate = float(f.read().split()[0])     # "200 Gb/sec (4X HDR)"
                with open(os.path.join(path, 'ports', '1', 'state')) as f:
                    state = f.read().split(':')[-1].strip()
            except (OSError, ValueError, IndexError):
                pass
            devices[name] = {'pci': pci, 'switch': os.path.dirname(os.path.dirname(pci)),
                             'rate_gbps': rate, 'state': state, 'claimed': False}
        # A pod only gets /dev/infiniband/uverbsN for the EFAs in its resource claim, while /sys shows
        # every EFA on the host. On a host (no container) every EFA has its device node.
        for verbs in glob.glob('/sys/class/infiniband_verbs/uverbs*'):
            names_dir = os.path.join(verbs, 'device', 'infiniband')
            names = os.listdir(names_dir) if os.path.isdir(names_dir) else []
            if names and names[0] in devices and os.path.exists('/dev/infiniband/' + os.path.basename(verbs)):
                devices[names[0]]['claimed'] = True
        return devices

    def _map_neuron_devices(self):
        """ PCIe switch -> Neuron device indices, from neuron-ls (BDF) and /sys/bus/pci. """
        import subprocess
        try:
            out = subprocess.run(['neuron-ls', '--json-output'], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, timeout=30).stdout
            for dev in json.loads(out.decode() or '[]'):
                bdf = dev.get('bdf')
                if bdf is None:
                    continue
                pci = os.path.realpath('/sys/bus/pci/devices/' + bdf)
                self.neuron_by_switch.setdefault(os.path.dirname(os.path.dirname(pci)), []).append(dev['neuron_device'])
        except Exception:  # neuron-ls missing or not parseable: just no pairing column
            pass

    def _read_sysfs(self):
        values = {}
        for name in self.devices:
            base = os.path.join(self.IB_ROOT, name, 'ports', '1', 'hw_counters')
            if not os.path.isdir(base):
                return None
            entry = {}
            for counter in self.COUNTERS:
                try:
                    with open(os.path.join(base, counter)) as f:
                        entry[counter] = int(f.read())
                except (OSError, ValueError):
                    pass
            values[name] = entry
        return values

    def _read_netlink(self):
        import subprocess
        out = subprocess.run(['rdma', '-j', 'statistic', 'show', 'link'], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=5).stdout
        values = {}
        for entry in json.loads(out.decode() or '[]'):
            name = entry.get('ifname')
            if name in self.devices and entry.get('port', 1) == 1:
                values[name] = dict((c, int(entry[c])) for c in self.COUNTERS if c in entry)
        return values

    def _read_counters(self):
        if self.source != 'rdma netlink':
            values = self._read_sysfs()
            if values is not None:
                self.source = 'sysfs'
                return values
        self.source = 'rdma netlink'
        return self._read_netlink()

    def _read_users(self):
        """ EFA name -> number of local processes that have it open """
        verbs_to_dev = {}
        for verbs in glob.glob('/sys/class/infiniband_verbs/uverbs*'):
            names = os.listdir(os.path.join(verbs, 'device', 'infiniband')) if \
                os.path.isdir(os.path.join(verbs, 'device', 'infiniband')) else []
            if names:
                verbs_to_dev['/dev/infiniband/' + os.path.basename(verbs)] = names[0]
        users = {}
        for proc in glob.glob('/proc/[0-9]*'):
            seen = set()
            for fd in glob.glob(proc + '/fd/*'):
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target in verbs_to_dev:
                    seen.add(verbs_to_dev[target])
            for name in seen:
                users[name] = users.get(name, 0) + 1
        return users

    def _users_worker(self):
        self._map_neuron_devices()
        while True:
            try:
                users = self._read_users()
                with self.lock:
                    self.users = users
            except Exception:  # /proc races (processes exiting): try again next round
                pass
            time.sleep(self.USERS_PERIOD_S)

    def _worker(self):
        prev, prev_t = None, 0.0
        while True:
            try:
                now = time.time()
                values = self._read_counters()
                with self.lock:
                    users = dict(self.users)
                rows = {}
                if prev is not None and now > prev_t:
                    dt = now - prev_t
                    for name, entry in values.items():
                        old = prev.get(name, {})
                        rate = dict((c, max(0, entry[c] - old.get(c, entry[c])) / dt) for c in entry)
                        rate['users'] = users.get(name, 0)
                        rows[name] = rate
                prev, prev_t = values, now
                with self.lock:
                    self.rows, self.error = rows, ''
            except Exception as e:  # e.g. no sysfs counters and no `rdma` tool: show why, keep trying
                with self.lock:
                    self.error = 'EFA counters unavailable (%s)' % (e.__class__.__name__, )
                    self.source = None
            time.sleep(self.PERIOD_S)

    def snapshot(self):
        with self.lock:
            return dict(self.rows), self.source, self.error


class Utils:
    """ General use static functions
    """
    @staticmethod
    def human_readable_size(size, unit, div, right_align=True, fixed_width=True, precision=1):
        suffix = ('', 'K', 'M', 'G', 'T', 'P', 'E')
        remaining = float(size)
        format_spec = f'%.{precision}f%s'
        digit_count = len(str(div - 1))
        for suf in suffix:
            if remaining < div:
                if fixed_width:
                    if len(suf) == 0:
                        digit_count += 1
                    if right_align:
                        format_spec = f'%{digit_count + precision + 1}.{precision}f%s'
                    else:
                        format_spec = f'%-{digit_count + precision + 1}.{precision}f%s'
                return format_spec % (remaining, suf + unit)
            remaining = remaining / div
        return 'NaN'

    @staticmethod
    def human_readable_byte_size(size, right_align=True, fixed_width=True):
        return Utils.human_readable_size(size, 'B', 1024, right_align, fixed_width)

    @staticmethod
    def format_percentage(perc):
        return '  100%' if perc > 99.995 else '%5.2f%%' % (perc,)

    @staticmethod
    def format_flops(flops):
        tflops = flops * 1.0e-12
        if tflops < 0.001:
            return '       0TF'
        return '%8.3fTF' % (tflops)

    @staticmethod
    def prop_getter(arr, idx):
        return arr[idx] if idx < len(arr) else arr[0]

    @staticmethod
    def get_neuron_device_indices():
        indices = []
        file_matches = glob.glob('/dev/neuron*')
        for neuron_node in file_matches:
            result = re.search(Const.NEURON_NODES_REGEX, neuron_node)
            if result is None:
                continue
            indices.append(int(result.group(1)))
        return sorted(indices)

    @staticmethod
    def shorten_text(text, max_length, cut_left=True):
        if max_length <= 0:
            return ''
        overflow = len(text) - max_length
        if overflow <= 0:
            return text
        return text[overflow:] if cut_left else text[:-overflow]


class ScreenManager:
    """ Class managing the screen, handling color management and providing support for common draw operations
        (such as drawing a utilization bar).
    """
    def __init__(self):
        self.surface = curses.initscr()
        self.surface.nodelay(True)
        self.surface.keypad(True)
        self.surface_size = self.surface.getmaxyx()
        self.available_color_attr = {}
        curses.start_color()
        curses.curs_set(0)
        curses.noecho()
        curses.nonl()
        try:
            curses.cbreak()
        except:
            pass
        self.update_screen_size()

    def __del__(self):
        curses.endwin()

    @staticmethod
    def dim(color_attr):
        return color_attr | curses.A_DIM

    @staticmethod
    def bright(color_attr):
        return color_attr | curses.A_BOLD

    @staticmethod
    def make_color_key(color, bg_color):
        return (color << 8) | bg_color

    def safe_addstr(self, row, col, text, attrs=None):
        if attrs is None:
            attrs = self.get_default_color_attr()
        try:
            self.surface.addstr(row, col, text, attrs)
        except:
            pass

    def get_color_attr(self, color, bg_color=curses.COLOR_BLACK):
        color_key = ScreenManager.make_color_key(color, bg_color)
        if color_key in self.available_color_attr:
            return curses.color_pair(self.available_color_attr[color_key])
        nb_entries = len(self.available_color_attr)
        if nb_entries >= curses.COLORS:
            return self.get_default_color_attr()
        new_index = nb_entries + 1
        curses.init_pair(new_index, color, bg_color)
        self.available_color_attr[color_key] = new_index
        return curses.color_pair(new_index)

    def get_default_color_attr(self):
        return curses.color_pair(0)

    def update_screen_size(self):
        surface_size = self.surface.getmaxyx()
        just_resized = self.surface_size != surface_size
        self.surface_size = surface_size
        return just_resized

    def refresh(self):
        self.surface.refresh()

    def get_width(self):
        return self.surface_size[1]

    def get_height(self):
        return self.surface_size[0]

    def get_size(self):
        return self.surface_size[1], self.surface_size[0]

    def draw_ratio_bar(self, pos_x, pos_y, title_text, value_text, width, ratios, clrs, highlight=False):
        """ Draws a bar which has several sections, each displayed in a different color

        The format of the bar is title_text ||||||||||||value_text and has a length of width (including the texts),
        the color of each section is provided by the clrs list and the length of each section is provided by ratios.
        Parameters
        ----------
        pos_x : int
            x position where the bar will be drawn
        pos_y : int
            y position where the bar will be drawn
        title_text: string
            text displayed on the left side of the bar
        value_text: string
            text displayed on the right side of the bar
        width: int
            total length of the bar (including the texts, the actual length of the bar will be adjusted based on
            this parameter and the length of title_text and value_text)
        ratios: list
            a list of values between 0 and 1 representing the length of each section
        clrs: list
            a list of colors to draw each section, has to have the same number of elements as ratios
        highlight: boolean
            if true, the bar and the texts will have a different background color (default False)
        """
        total_text_width = len(value_text)
        title_text_width = len(title_text)
        if title_text_width > 0:
            total_text_width += title_text_width + 1
        bar_width = width - total_text_width
        if bar_width <= 0:
            return pos_x
        self.safe_addstr(pos_y, pos_x, ' ' * width, self.get_default_color_attr())
        if title_text_width > 0:
            self.safe_addstr(pos_y, pos_x, title_text + ' ', self.get_default_color_attr())
            pos_x += title_text_width + 1
        prev_remainder = 0.0
        remaining_width = bar_width
        for idx, ratio in enumerate(ratios):
            used_width = ratio * bar_width
            used_width -= prev_remainder
            if used_width >= 0.0:
                prev_remainder = 0.0
            else:
                prev_remainder = -used_width
                used_width = 0.0
            remainder = used_width - math.floor(used_width)
            used_width = int(used_width)
            if used_width == 0:
                continue
            if remainder > 0:
                prev_remainder = (1.0 - remainder)
            remainder_width = 0
            if remainder > 0.25:
                remainder_width = 1
            else:
                prev_remainder = 0
            color = self.get_color_attr(clrs[idx])
            if highlight:
                color = self.get_color_attr(clrs[idx], curses.COLOR_BLUE)
            for part in ((used_width, ScreenManager.bright(color)),
                         (remainder_width, ScreenManager.dim(color))):
                if part[0] == 0:
                    continue
                self.safe_addstr(pos_y, pos_x, '|' * part[0], part[1])
                pos_x += part[0]
                remaining_width -= part[0]
        fill_color = self.get_default_color_attr()
        if highlight:
            fill_color = self.get_color_attr(curses.COLOR_WHITE, curses.COLOR_BLUE)
        if remaining_width > 0:
            self.safe_addstr(pos_y, pos_x, '|' * remaining_width, ScreenManager.dim(fill_color))
            pos_x += remaining_width
        self.safe_addstr(pos_y, pos_x, value_text, self.get_default_color_attr())
        return pos_x + len(value_text)

    def clear_section(self, pos_y, height):
        scr_w, scr_h = self.get_size()
        for crt_y in range(pos_y, pos_y + height):
            if crt_y < 0 or crt_y >= scr_h:
                continue
            self.safe_addstr(crt_y, 0, ' ' * scr_w, self.get_default_color_attr())

    def clear_screen(self):
        self.surface.clear()
        self.refresh()


class Widget:
    """ Base class for all screen widgets.
    """
    REFRESH_MODE_NONE = 0
    REFRESH_MODE_DATA = 1
    REFRESH_MODE_FULL = (1 << 2)

    def __init__(self, screen, pos_y, height):
        self.screen = screen
        self.refresh_mode = Widget.REFRESH_MODE_FULL
        self.pos_y = pos_y
        self.height = height
        self.visible = True

    def check_bounds(self):
        height = self.screen.get_height()
        return self.pos_y >= 0 and self.pos_y < height

    def refresh(self):
        pass

    def recalculate_layout(self):
        pass

    def clear(self):
        self.screen.clear_section(self.pos_y, self.height)

    def set_pos_y(self, new_y):
        if self.pos_y != new_y:
            prev_pos_y = self.pos_y
            self.pos_y = new_y
            self.on_pos_y_changed(prev_pos_y)

    def set_height(self, new_height):
        if self.height != new_height:
            prev_height = self.height
            self.height = new_height
            self.on_height_changed(prev_height)

    def set_visible(self, visible):
        if self.visible != visible:
            prev_visible = self.visible
            self.visible = visible
            self.on_visible_changed(prev_visible)

    def on_pos_y_changed(self, prev_y):
        self.screen.clear_section(prev_y, self.height)
        self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def on_height_changed(self, prev_height):
        self.recalculate_layout()
        self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def on_visible_changed(self, prev_visible):
        if self.visible is False:
            self.screen.clear_section(self.pos_y, self.height)
        else:
            self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def on_data_updated(self):
        self.refresh_mode |= Widget.REFRESH_MODE_DATA

    def on_screen_resize(self):
        self.recalculate_layout()
        self.refresh_mode |= Widget.REFRESH_MODE_FULL

    def refresh_requested(self):
        return self.refresh_mode != 0

    def on_post_refresh(self):
        self.refresh_mode = 0

    def is_refresh_mode_requested(self, mode):
        return (self.refresh_mode & mode) != 0

    def request_refresh_mode(self, mode):
        self.refresh_mode |= mode

    def on_key(self, key):
        pass


class LabelWidget(Widget):
    """ Simple widget that displays one or more labels on a single line.
    """
    ALIGN_LEFT = 0
    ALIGN_RIGHT = 1
    ALIGN_CENTER = 2

    def __init__(self, screen, pos_y, margins, texts, aligns, colors, bg_color):
        Widget.__init__(self, screen, pos_y, 1)
        self.colors = colors
        self.margins = margins
        self.bg_color = bg_color
        self.set_texts(texts, aligns)

    def set_texts(self, texts, aligns):
        self.texts = texts
        self.shortened_texts = [''] * len(self.texts)
        self.start_x = 0
        self.width = 0
        self.aligns = aligns
        self.text_pos_x = [0] * len(texts)
        self.recalculate_layout()

    def set_colors(self, colors, bg_color):
        self.colors = colors
        self.bg_color = bg_color
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def recalculate_layout(self):
        width = self.screen.get_width()
        col_count = len(self.texts)
        if col_count == 0:
            return
        text_area_width = width - sum(self.margins) - Const.SCREEN_MARGIN * 2
        base_col_width = text_area_width // col_count
        col_width_remaining = text_area_width % col_count
        if base_col_width <= 0:
            return
        self.start_x = Const.SCREEN_MARGIN + self.margins[0]
        self.width = text_area_width
        crt_x = self.start_x
        for idx, label in enumerate(self.texts):
            col_width = base_col_width
            if col_width_remaining > 0:
                col_width += 1
                col_width_remaining -= 1
            short_text = Utils.shorten_text(label, col_width - 1, False)
            # hack to keep the > < selector intact
            if len(short_text) > 1 and short_text[0] == '>' and short_text[-1] != '<':
                short_text = short_text[:-1] + '<'
            self.shortened_texts[idx] = short_text
            text_width = len(short_text)
            pos_x = crt_x
            align = Utils.prop_getter(self.aligns, idx)
            if align == LabelWidget.ALIGN_RIGHT:
                pos_x = crt_x + col_width - text_width
            elif align == LabelWidget.ALIGN_CENTER:
                pos_x = crt_x + (col_width - text_width) // 2
            self.text_pos_x[idx] = pos_x
            crt_x += col_width
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def refresh(self):
        if not self.check_bounds():
            return
        if not self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
            return
        color_attr = self.screen.get_color_attr(Utils.prop_getter(self.colors, 0), self.bg_color)
        self.screen.safe_addstr(self.pos_y, self.start_x, ' ' * self.width, color_attr)
        for idx, label in enumerate(self.shortened_texts):
            if not label:
                continue
            color_attr = self.screen.get_color_attr(Utils.prop_getter(self.colors, idx), self.bg_color)
            self.screen.safe_addstr(self.pos_y, self.text_pos_x[idx], label, color_attr)


class NCUsageWidget(Widget):
    """ NeuronCore utilization widget, shows 4 utilization bars representing the NeuronCore utilization for a
        Neuron device.
    """
    NC_USAGE_MODE_PERCENT = 0
    NC_USAGE_MODE_FLOPS = 1

    def __init__(self, screen, pos_y, nd_index, nc_per_nd):
        Widget.__init__(self, screen, pos_y, 1)
        self.nd_index = nd_index
        self.nc_per_nd = nc_per_nd
        self.utilization = [0.0] * nc_per_nd
        self.flops = [0.0] * nc_per_nd
        self.highlight = [False] * nc_per_nd
        self.bar_updated = [True] * nc_per_nd
        self.bar_widths = [0] * nc_per_nd
        self.bar_pos_x = [0] * nc_per_nd
        self.bar_visible = [False] * nc_per_nd
        self.mode = NCUsageWidget.NC_USAGE_MODE_PERCENT
        self.recalculate_layout()

    def update_data(self, utilization, flops):
        data_updated = False
        for idx, util in enumerate(utilization):
            if self.utilization[idx] != util:
                self.bar_updated[idx] = True
                self.utilization[idx] = util
                data_updated = True
        for idx, util in enumerate(flops):
            if self.flops[idx] != flops:
                self.bar_updated[idx] = True
                self.flops[idx] = util
                data_updated = True
        if data_updated:
            self.on_data_updated()

    def set_highlight(self, index, show=True):
        if index == -1:
            index = 0
            end_idx = self.nc_per_nd
        else:
            end_idx = index + 1
        highlight_updated = False
        for idx in range(index, end_idx):
            if show != self.highlight[idx]:
                self.highlight[idx] = show
                self.bar_updated[idx] = True
                highlight_updated = True
        if highlight_updated:
            self.on_data_updated()

    def switch_mode(self):
        if self.mode == NCUsageWidget.NC_USAGE_MODE_FLOPS:
            self.mode = NCUsageWidget.NC_USAGE_MODE_PERCENT
        else:
            self.mode = NCUsageWidget.NC_USAGE_MODE_FLOPS
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def recalculate_layout(self):
        width = self.screen.get_width() - (Const.SCREEN_MARGIN * 2)
        col_width = width - Const.SCREEN_MARGIN - self.nc_per_nd + 1
        remaining_width = col_width % self.nc_per_nd
        col_width //= self.nc_per_nd
        crt_x = Const.SCREEN_MARGIN
        for idx, _ in enumerate(self.utilization):
            adj_col_width = col_width
            if remaining_width > 0:
                adj_col_width += 1
                remaining_width -= 1
            self.bar_pos_x[idx] = crt_x
            self.bar_widths[idx] = col_width
            crt_x += col_width + 1

    def refresh(self):
        """ NDxx ########[99.99%] ########[99.99%] ########[99.99%] ########[99.99%]
        """
        if not self.check_bounds():
            return
        nd_text = 'ND%-2d' % (self.nd_index,)
        for idx, util in enumerate(self.utilization):
            if self.bar_updated[idx] or self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
                percent_text = ''
                if self.mode == NCUsageWidget.NC_USAGE_MODE_PERCENT:
                    percent_text = '[%s]' % (Utils.format_percentage(util))
                else:
                    percent_text = '[%s]' % (Utils.format_flops(self.flops[idx]))
                self.screen.draw_ratio_bar(self.bar_pos_x[idx], self.pos_y, nd_text, percent_text, self.bar_widths[idx],
                                           (util / 100.0, ), (curses.COLOR_GREEN, ), self.highlight[idx])
                self.bar_updated[idx] = False
            nd_text = ''

class RuntimeMemoryUsageWidget(Widget):
    """ Runtime memory utilization widget, shows an utilization bar indicating the host memory usage, and displays
        the Neuron device memory usage.
    """
    SPACING = 4
    def __init__(self, screen, pos_y, fields, field_names, title, title_width, extra_cols=0):
        Widget.__init__(self, screen, pos_y, 1)
        self.cur_usage_values = dict([(field, 0) for field in fields])
        self.data_updated = dict([(field, False) for field in fields])
        self.fields = fields
        self.field_names = field_names
        self.col_padding = 0
        self.extra_cols = extra_cols
        self.title = title
        self.title_width = title_width + RuntimeMemoryUsageWidget.SPACING
        self.recalculate_layout()

    def update_data(self, usage_values):
        diff_found = False
        for name, value in usage_values.items():
            if name not in self.cur_usage_values:
                continue
            if value != self.cur_usage_values[name]:
                self.cur_usage_values[name] = value
                self.data_updated[name] = True
                diff_found = True
        if diff_found:
            self.on_data_updated()

    def recalculate_layout(self):
        width = self.screen.get_width() - (Const.SCREEN_MARGIN * 2) - RuntimeMemoryUsageWidget.SPACING
        width -= self.title_width
        self.col_width = max(0, (width // (len(self.cur_usage_values) + self.extra_cols) - RuntimeMemoryUsageWidget.SPACING))

    def refresh(self):
        """ Title:     Total: xx.xxGB/xx.xxGB     Field1: xx.xxGB
        """
        if not self.check_bounds():
            return

        if self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
            self.clear()

        current_x = Const.SCREEN_MARGIN
        self.screen.safe_addstr(self.pos_y, current_x, self.title)
        current_x += self.title_width + RuntimeMemoryUsageWidget.SPACING

        for idx, field in enumerate(self.fields):
            pos_x = current_x
            current_x += self.col_width + RuntimeMemoryUsageWidget.SPACING
            if not self.data_updated[field] and not self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
                continue

            field_name = self.field_names[idx]
            cur_usage_value = self.cur_usage_values[field]
            bytes_current = Utils.human_readable_byte_size(cur_usage_value, False, False)
            string_part_one = '%s: ' % (field_name)
            string_part_two = '%s' % (bytes_current)
            full_string = '%s%s' % (string_part_one, string_part_two)
            full_string_len = len(full_string)

            if full_string_len > self.col_width:
                continue

            remaining = self.col_width - full_string_len
            self.screen.safe_addstr(self.pos_y, pos_x, string_part_one, self.screen.get_default_color_attr())
            self.screen.safe_addstr(self.pos_y, pos_x + len(string_part_one), string_part_two,
                                    self.screen.get_color_attr(curses.COLOR_GREEN))
            if remaining > 0:
                self.screen.safe_addstr(self.pos_y, pos_x + full_string_len, ' ' * remaining,
                                        self.screen.get_default_color_attr())
            self.data_updated[field] = False


class VCPUUsageWidget(Widget):
    """ vCPU utilization widget, shows an utilization bar indicating the total system average vCPU usage and one
        displaying the percent of vCPU used exclusively by the Neuron Runtime.
    """
    SPACING = 4
    def __init__(self, screen, pos_y):
        Widget.__init__(self, screen, pos_y, 2)
        self.system_cpu_usage = {
            'user': 0,
            'system': 0,
            'needs_refresh': True
        }
        self.neuron_runtime_cpu_usage = {
            'user': 0,
            'system': 0,
            'needs_refresh': True
        }
        self.recalculate_layout()

    def recalculate_layout(self):
        self.col_width = self.screen.get_width() - (Const.SCREEN_MARGIN * 2)
        self.bar_pos_x = Const.SCREEN_MARGIN

    def update_data(self, system_cpu_usage, neuron_runtime_cpu_usage):
        found_updates = False
        for entry in ((self.system_cpu_usage, system_cpu_usage),
                      (self.neuron_runtime_cpu_usage, neuron_runtime_cpu_usage)):
            if entry[1] is None:
                continue
            for field in entry[1]:
                if entry[0][field] != entry[1][field]:
                    entry[0][field] = entry[1][field]
                    entry[0]['needs_refresh'] = True
                    found_updates = True
        if found_updates:
            self.on_data_updated()

    def refresh(self):
        """ System vCPU Usage  ########[99.99%,99.99%]
            Runtime vCPU Usage ########[99.99%,99.99%]
        """
        if not self.check_bounds():
            return
        full_refresh = self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL)
        if full_refresh:
            self.clear()

        title_1 = 'System vCPU Usage '
        title_2 = 'Runtime vCPU Usage'

        for idx, entry in enumerate(((self.system_cpu_usage, title_1),
                                     (self.neuron_runtime_cpu_usage, title_2))):
            if entry[0]['needs_refresh'] or full_refresh:
                percent_text = '[%s,%s]' % (Utils.format_percentage(entry[0]['user']),
                                            Utils.format_percentage(entry[0]['system']))
                self.screen.draw_ratio_bar(self.bar_pos_x, self.pos_y + idx, entry[1], percent_text, self.col_width,
                                           (entry[0]['user'] / 100.0, entry[0]['system'] / 100.0),
                                           (curses.COLOR_GREEN, curses.COLOR_RED))
                entry[0]['needs_refresh'] = False


class ModelMemoryUsageListWidget(Widget):
    """ Model tree view widget, displays the currently loaded models in a tree-like structure which responds
        to key pressed (up/down/left/right/x). It also indicates the amount of host memory and Neuron Device
        memory used by each.
    """
    ITEM_TYPE_ND = 0
    ITEM_TYPE_NC = 1
    ITEM_TYPE_MD = 2
    ITEM_TYPE_USAGE = 3

    SELECT_NEXT = 0
    SELECT_PREV = 1

    INDENT_SIZE = 4

    # Models below this size with no user source line are grouped into one row
    SMALL_GRAPH_BYTES = 1024 * 1024
    # Identifier of the group row; sorts after every real model ID and is not printed
    GROUP_ID = 1 << 62
    # Identifier of the 'Not owned by a NEFF' row; sorts after the helper-graph group
    UNOWNED_ID = GROUP_ID + 1
    # Identifier of a model's 'Model Code by op' row; sorts after its usage rows
    OPS_ID = 1000
    UNOWNED_TITLES = {
        'model_shared_scratchpad': 'Scratchpad (reused by every NEFF)',
        'constants': 'Constants (not tied to a NEFF)',
        'model_code': 'Model Code (not tied to a NEFF)',
        'runtime_memory': 'Runtime internal',
    }

    @total_ordering
    class UsageListItem:
        """ Single item of a ModelMemoryUsageListWidget
        """
        def __init__(self, parent, identifier, text, item_type):
            self.selected = False
            self.expanded = False
            self.is_active = False
            self.needs_refresh = True
            self.dimmed = False
            self.text = text
            self.byte_sizes = []
            self.parent = parent
            self.identifier = identifier
            self.item_type = item_type
            self.index = 0
            self.screen_pos_y = None
            self.children = []

        def set_data(self, byte_sizes, is_dimmed):
            needs_refresh = False
            if self.byte_sizes != byte_sizes:
                self.byte_sizes = byte_sizes
                needs_refresh = True
            if self.dimmed != is_dimmed:
                self.dimmed = is_dimmed
                needs_refresh = True
            if needs_refresh:
                # redraw this line, otherwise a shorter new value leaves old characters behind
                self.needs_refresh = True
            self.is_active = True

        def refresh(self, screen, pos_x, pos_y, base_pos_y, values_x, max_x, max_y, full_refresh):
            """
            [+] ND0                                      1221MB              16MB
                [+] NC0                                   100MB               8MB
                    [+] myModel               11223344     50MB               4MB
                        tensors                             1MB
            """
            needs_refresh = self.needs_refresh or full_refresh
            self.needs_refresh = False
            if needs_refresh and pos_y >= base_pos_y and pos_y <= max_y:
                screen.clear_section(pos_y, 1)
                color_attr = screen.get_default_color_attr()
                if self.dimmed:
                    color_attr = ScreenManager.dim(color_attr)
                text = ''
                if self.children:
                    if self.expanded:
                        text = '[-] '
                    else:
                        text = '[+] '
                max_len = values_x[0] - pos_x - 1 - len(text)
                text += Utils.shorten_text(self.text, max_len, True)
                text = Utils.shorten_text(text, values_x[0] - pos_x - 1, False)
                if text:
                    screen.safe_addstr(pos_y, pos_x, text, color_attr)
                for idx, value in enumerate([self.identifier] + self.byte_sizes):
                    if idx == 0 and (self.item_type is not ModelMemoryUsageListWidget.ITEM_TYPE_MD or
                                     self.identifier >= ModelMemoryUsageListWidget.GROUP_ID):
                        continue
                    text_x = values_x[idx]
                    if idx == 0:
                        text = '%d' % (self.identifier, )
                    else:
                        text_x += 1
                        text = Utils.human_readable_byte_size(value, False, False)
                    screen.safe_addstr(pos_y, text_x, text, color_attr)
                    if text_x + len(text) >= max_x:
                        break
                if self.selected:
                    color_attr = screen.get_color_attr(curses.COLOR_YELLOW, curses.COLOR_RED)
                    screen.surface.chgat(pos_y, pos_x, max_x - pos_x - 1, color_attr)
            self.screen_pos_y = pos_y
            pos_y += 1
            if self.expanded:
                for item in self.children:
                    pos_y = item.refresh(screen, pos_x + ModelMemoryUsageListWidget.INDENT_SIZE, pos_y, base_pos_y,
                                         values_x, max_x, max_y, full_refresh)
            return pos_y

        def __lt__(self, other):
            if self.item_type == other.item_type:
                return self.identifier < other.identifier
            return self.item_type < other.item_type

        def __eq__(self, other):
            return self.item_type == other.item_type and self.identifier == other.identifier

        def __hash__(self):
            return id(self)

    def __init__(self, screen, pos_y, bottom_margin, nd_indices, nc_count):
        Widget.__init__(self, screen, pos_y, screen.get_height() - bottom_margin - pos_y)
        self.children = []
        self.column_names = ['Model ID', 'Device Memory', 'Host Memory']
        self.bottom_margin = bottom_margin
        self.selected = None
        self.scroll_amount = 0
        self.base_y = None
        self.selection_changed_handler = None
        self.fully_expanded = False
        self.neff_context = NeffContext()
        self.op_runner = OpAttributionRunner()
        self.ops_shown = set()   # uuids whose 'Model Code by op' rows are open
        self._init_device_tree(nd_indices, nc_count)
        self.recalculate_layout()

    def set_selection_changed_handler(self, handler):
        self.selection_changed_handler = handler

    def _parse_tree(self, root, func):
        func(root)
        for child in root.children:
            self._parse_tree(child, func)

    def _pre_update(self):

        def pre_updater(node):
            if node is not self:
                node.is_active = False

        self._parse_tree(self, pre_updater)

    def _post_update(self):
        to_delete = {}
        selected_deleted = False

        def post_updater(node):
            nonlocal selected_deleted
            for idx, child in enumerate(node.children):
                if child.is_active is True or child.item_type not in [self.ITEM_TYPE_MD, self.ITEM_TYPE_USAGE]:
                    continue
                if node not in to_delete:
                    to_delete[node] = set()
                to_delete[node].add(idx)
                if self.selected is child:
                    selected_deleted = True

        self._parse_tree(self, post_updater)
        if to_delete:
            for deleted, deleted_indices in to_delete.items():
                deleted.children = [c for i, c in enumerate(deleted.children) if i not in deleted_indices]
                for idx, child in enumerate(deleted.children):
                    child.index = idx
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
        if not self.children:
            self._select_new_item(None)
        elif selected_deleted or self.selected is None:
            self._select_new_item(self.children[0])
        else:
            self._make_selected_item_visible()

    def add_or_get_child(self, parent, identifier, text, item_type):
        temp = ModelMemoryUsageListWidget.UsageListItem(parent, identifier, text, item_type)
        idx = 0
        if parent.children:
            idx = bisect.bisect_right(parent.children, temp)
            found_index = idx - 1
            if found_index < len(parent.children) and parent.children[found_index] == temp:
                return parent.children[found_index]
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
        parent.children.insert(idx, temp)
        for upd_idx in range(idx, len(parent.children)):
            parent.children[upd_idx].index = upd_idx
        return parent.children[idx]

    def recalculate_layout(self):
        width, height = self.screen.get_size()
        usable_width = width - Const.SCREEN_MARGIN * 2
        self.max_x = usable_width - 1
        self.max_y = height - self.bottom_margin - 1
        model_name_len = max(Const.MAX_MODEL_NAME_LEN, usable_width // 2)
        start_x = Const.SCREEN_MARGIN + model_name_len + self.INDENT_SIZE * 3
        col_count = len(self.column_names)
        col_width = (usable_width - start_x) // col_count
        self.values_pos_x = [start_x + col_width * col_idx for col_idx in range(col_count)]
        new_base_y = self.pos_y + 1
        if self.base_y != new_base_y:
            self.base_y = new_base_y
            self._refresh_items_screen_y_pos()

    def refresh(self):
        full_refresh = self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL)
        if full_refresh:
            self.screen.clear_section(self.pos_y, self.max_y - self.pos_y + 1)
            for idx, text in enumerate(self.column_names):
                if self.values_pos_x[idx] + len(text) >= self.max_x:
                    break
                self.screen.safe_addstr(self.pos_y, self.values_pos_x[idx], text)
        crt_y = self.base_y - self.scroll_amount
        for item in self.children:
            crt_y = item.refresh(self.screen, 2, crt_y, self.base_y, self.values_pos_x,
                                 self.max_x, self.max_y, full_refresh)

    def on_key(self, key):
        if key == curses.KEY_UP:
            self._select_prev()
        elif key == curses.KEY_DOWN:
            self._select_next()
        if key == curses.KEY_LEFT:
            self._collapse_and_select()
        elif key == curses.KEY_RIGHT:
            self._expand_and_select()
        elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
            self._expand_collapse_selected()
        elif key in [ord('x'), ord('X')]:
            self._toggle_expand_all()
        elif key in [ord('o'), ord('O')]:
            self._toggle_ops_for_selected()

    def on_pos_y_changed(self, prev_y):
        self.height = self.screen.get_height() - self.bottom_margin - self.pos_y
        Widget.on_pos_y_changed(self, prev_y)
        self.recalculate_layout()

    def _get_last_expanded_item(self, root):
        while True:
            if root.expanded is False or not root.children:
                return root
            root = root.children[-1]

    def _get_next_available_item(self, current):
        while True:
            if current.index < len(current.parent.children) - 1:
                return current.parent.children[current.index + 1]
            if current.parent is self:
                return self.children[0]
            current = current.parent

    def _get_next_item(self, direction):
        crt_index = self.selected.index
        if direction == ModelMemoryUsageListWidget.SELECT_PREV:
            if crt_index > 0:
                return self._get_last_expanded_item(self.selected.parent.children[crt_index - 1])
            if self.selected.parent is not self:
                return self.selected.parent
            return self._get_last_expanded_item(self.children[-1])
        if self.selected.expanded and self.selected.children:
            return self.selected.children[0]
        return self._get_next_available_item(self.selected)

    def _make_selected_item_visible(self):
        if self.selected is not None:
            if self.is_refresh_mode_requested(Widget.REFRESH_MODE_FULL):
                self._refresh_items_screen_y_pos()
            bound = (self.base_y + 1, self.screen.get_height() - self.bottom_margin - 1)
            if self.selected.screen_pos_y < bound[0]:
                self.scroll_amount = max(0, self.scroll_amount - (bound[0] - self.selected.screen_pos_y))
                self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
            elif self.selected.screen_pos_y > bound[1]:
                self.scroll_amount += (self.selected.screen_pos_y - bound[1])
                self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def _select_new_item(self, new_item):
        old_item = self.selected
        if old_item is not new_item:
            if old_item is not None:
                old_item.selected = False
                old_item.needs_refresh = True
            if new_item is not None:
                new_item.selected = True
                new_item.needs_refresh = True
            self.selected = new_item
            self.request_refresh_mode(Widget.REFRESH_MODE_DATA)
            self._make_selected_item_visible()
            if self.selection_changed_handler is not None:
                self.selection_changed_handler.on_selection_changed(old_item, new_item)

    def _select_prev(self):
        if self.selected is None:
            return
        self._select_new_item(self._get_next_item(ModelMemoryUsageListWidget.SELECT_PREV))

    def _select_next(self):
        if self.selected is None:
            return
        self._select_new_item(self._get_next_item(ModelMemoryUsageListWidget.SELECT_NEXT))

    def _collapse_and_select(self):
        if self.selected is None:
            return
        if self.selected.expanded is True:
            self.selected.expanded = False
            for child in self.selected.children:
                child.expanded = False
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
            return
        parent = self.selected.parent
        if parent is self:
            self._select_new_item(self.children[0])
        else:
            self._select_new_item(parent)

    def _expand_and_select(self):
        if self.selected is None:
            return
        if self.selected.expanded is False:
            self.selected.expanded = True
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
            return
        if self.selected.children:
            self._select_new_item(self.selected.children[0])

    def _expand_collapse_selected(self):
        if self.selected is None:
            return
        if self.selected.children:
            self.selected.expanded = not self.selected.expanded
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def get_top_parent(self, node):
        while node.parent is not self:
            node = node.parent
        return node

    def _refresh_items_screen_y_pos(self):
        crt_y = self.base_y - self.scroll_amount

        def y_updater(node):
            nonlocal crt_y
            if node is self:
                return
            node.screen_pos_y = crt_y
            if node.parent is self or node.parent.expanded is True:
                crt_y += 1

        self._parse_tree(self, y_updater)

    def _expand_collapse_all(self, expanded=True):
        updated_exp = False

        def expand_setter(node):
            nonlocal updated_exp
            if node is self:
                return
            if node.expanded is not expanded and node.children:
                node.expanded = expanded
                updated_exp = True

        self._parse_tree(self, expand_setter)
        if updated_exp is False:
            return
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)
        if expanded is False and self.selected is not None:
            new_selection = self.get_top_parent(self.selected)
            self._select_new_item(new_selection)
        else:
            self._make_selected_item_visible()

    def _toggle_expand_all(self):
        new_val = not self.fully_expanded
        self._expand_collapse_all(new_val)
        self.fully_expanded = new_val

    def _update_unowned(self, nc_item, excl, max_input_bytes):
        """ Memory on a core that no loaded NEFF owns, under one row, largest first.
            The runtime reports application tensors as one total. PyTorch passes weights into each
            NEFF as inputs, so the input size of the largest NEFF on the core estimates the weights;
            the rest is activations and outputs, plus weights of pipeline stages whose NEFFs are
            not loaded at the moment.
        """
        rows = []
        tensors = excl.get('tensors', 0)
        if max_input_bytes > 0 and tensors > 0:
            weights = min(max_input_bytes, tensors)
            rows.append(('~App tensors: inputs of loaded NEFFs (weights, KV cache)', weights))
            # also holds weights of stages whose NEFFs are not loaded right now
            rows.append(('~App tensors: rest (activations, idle weights)', tensors - weights))
        else:
            rows.append(('App tensors (weights, activations)', tensors))
        rows += [(self.UNOWNED_TITLES[field], excl.get(field, 0))
                 for field in Const.MEM_USAGE_FIELD_NAMES[1] if field != 'tensors']
        rows = sorted([r for r in rows if r[1] > 0], key=lambda r: -r[1])
        if not rows:
            return
        group_item = self.add_or_get_child(nc_item, self.UNOWNED_ID, '', self.ITEM_TYPE_MD)
        self._set_text(group_item, 'Not owned by a NEFF')
        group_item.set_data([sum(r[1] for r in rows), 0], False)
        for rank, (title, value) in enumerate(rows):
            usage_item = self.add_or_get_child(group_item, rank, title, self.ITEM_TYPE_USAGE)
            self._set_text(usage_item, title)
            usage_item.set_data([value, 0], False)
            if usage_item.needs_refresh:
                self.request_refresh_mode(Widget.REFRESH_MODE_DATA)
        if group_item.needs_refresh:
            self.request_refresh_mode(Widget.REFRESH_MODE_DATA)

    def _toggle_ops_for_selected(self):
        """ Show or hide 'Model Code by op' for the selected model; computed once per NEFF. """
        item = self.selected
        uuid = getattr(item, 'uuid', None) if item is not None else None
        if not uuid:
            return
        if uuid in self.ops_shown:
            self.ops_shown.discard(uuid)
        else:
            self.ops_shown.add(uuid)
            path = self.neff_context.paths.get(uuid)
            if path:
                self.op_runner.request(uuid, path)
            item.expanded = True
        self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def _update_ops(self, model_item, uuid, model_code):
        """ 'Model Code by op' rows under a model. Percentages are shares of the model's Model Code
            (memory), not of compute time. The engine instructions are split by op; the rest of
            Model Code (not engine instructions) gets one row of its own. """
        header = self.add_or_get_child(model_item, self.OPS_ID, '', self.ITEM_TYPE_USAGE)
        if not getattr(header, 'auto_expanded', False):
            header.expanded = True
            header.auto_expanded = True
        path = self.neff_context.paths.get(uuid)
        if path:
            # idempotent; also covers 'o' pressed before the NEFF path was found
            self.op_runner.request(uuid, path)
        state = self.op_runner.get(uuid) if path else ('error', 'no NEFF file found')
        rows = []
        if state is None or state[0] == 'running':
            title, value = 'Model Code by op: computing (large NEFFs take a minute)...', 0
        elif state[0] == 'error':
            title, value = 'Model Code by op: unavailable (%s)' % (state[1], ), 0
        else:
            result = state[1]
            instr = result['total']
            # instructions of all subgraphs vs. what the runtime reports for this model on this core
            value = max(model_code, instr)
            total = value or 1
            title = 'Model Code by op (%% of Model Code memory; %.1f%% of instructions mapped)' % (
                100.0 * result['mapped'] / (instr or 1), )
            for op, src, nbytes in result['rows']:
                text = '%s · %s' % (op, src) if src else op
                rows.append(('%s · %.1f%%' % (text, 100.0 * nbytes / total), nbytes))
            rest = value - instr
            if rest > 0:
                rows.append(('not engine instructions, not split by op · %.1f%%' % (100.0 * rest / total, ), rest))
        self._set_text(header, title)
        header.set_data([value, 0], False)
        for rank, (text, nbytes) in enumerate(rows):
            row = self.add_or_get_child(header, rank, text, self.ITEM_TYPE_USAGE)
            self._set_text(row, text)
            row.set_data([nbytes, 0], False)
        if header.needs_refresh or rows:
            self.request_refresh_mode(Widget.REFRESH_MODE_DATA)

    def _set_text(self, item, text):
        """ Labels can change after creation, e.g. when a NEFF finishes parsing. """
        if item.text != text:
            item.text = text
            item.needs_refresh = True
            # rows below may shift (e.g. the 'Not owned' rows re-sort by size): redraw everything
            self.request_refresh_mode(Widget.REFRESH_MODE_FULL)

    def _get_byte_sizes_list(self, data_obj):
        return [data_obj['device_total'], data_obj['host_total']]

    def update_data(self, list_data):
        self._pre_update()
        nd_indices = sorted(list_data.keys())
        for nd_idx in nd_indices:
            nd_obj = list_data[nd_idx]
            nd_item = self.add_or_get_child(self, nd_idx, 'ND%2d' % (nd_idx,), self.ITEM_TYPE_ND)
            nd_item.set_data(self._get_byte_sizes_list(nd_obj), False)
            nc_indices = sorted(nd_obj['children'].keys())
            for nc_idx in nc_indices:
                nc_obj = nd_obj['children'][nc_idx]
                nc_item = self.add_or_get_child(nd_item, nc_idx, 'NC%2d' % (nc_idx,), self.ITEM_TYPE_NC)
                nc_item.set_data(self._get_byte_sizes_list(nc_obj), False)
                # Small graphs with no user source line (slice, broadcast, ...) go under one group row
                small, large = [], []
                max_input_bytes = 0
                for model, model_obj in nc_obj['children'].items():
                    ctx = self.neff_context.get(model_obj['uuid']) if model_obj['uuid'] else NeffContext.MISSING
                    if isinstance(ctx, dict):
                        max_input_bytes = max(max_input_bytes, ctx['input_bytes'])
                    is_small = model_obj['device_total'] < self.SMALL_GRAPH_BYTES and \
                        not NeffContext.has_user_source(ctx)
                    (small if is_small else large).append((model, model_obj, NeffContext.label(ctx, model_obj['name'])))
                group_item = None
                if len(small) > 1:
                    group_item = self.add_or_get_child(nc_item, self.GROUP_ID, '', self.ITEM_TYPE_MD)
                    self._set_text(group_item, '%d small helper graphs' % (len(small), ))
                    group_item.set_data([sum(m[1]['device_total'] for m in small),
                                         sum(m[1]['host_total'] for m in small)],
                                        not any(m[1]['is_running'] for m in small))
                    if group_item.needs_refresh:
                        self.request_refresh_mode(Widget.REFRESH_MODE_DATA)
                grouped = set(m[0] for m in small) if group_item is not None else set()
                for model, model_obj, label in large + small:
                    parent = group_item if model in grouped else nc_item
                    model_item = self.add_or_get_child(parent, model, label, self.ITEM_TYPE_MD)
                    model_item.uuid = model_obj['uuid']
                    self._set_text(model_item, label)
                    if model_obj['uuid'] in self.ops_shown:
                        device = model_obj['neuron_device'] or {}
                        self._update_ops(model_item, model_obj['uuid'], device.get('model_code', 0))
                    model_item.set_data(self._get_byte_sizes_list(model_obj), not model_obj['is_running'])
                    if model_item.needs_refresh:
                        self.request_refresh_mode(Widget.REFRESH_MODE_DATA)
                    for idx, field_device in enumerate(Const.MEM_USAGE_FIELD_NAMES_MODEL[1]):
                        field_host = Const.MEM_USAGE_FIELD_NAMES_MODEL[0][idx]
                        field_value_host = model_obj['host'][field_host]
                        field_value_device = model_obj['neuron_device'][field_device]
                        if field_value_host == 0 and field_value_device == 0:
                            continue
                        title = Const.field_name_to_field_title(field_device) if field_value_device != 0 else Const.field_name_to_field_title(field_host)
                        usage_item = self.add_or_get_child(model_item, idx, title, self.ITEM_TYPE_USAGE)
                        usage_item.set_data([field_value_device, field_value_host], False)
                        if usage_item.needs_refresh:
                            self.request_refresh_mode(Widget.REFRESH_MODE_DATA)
                self._update_unowned(nc_item, nc_obj['neuron_device_excl'], max_input_bytes)
        self._post_update()

    def _init_device_tree(self, nd_indices, nc_count):
        for nd_idx in nd_indices:
            nd_item = self.add_or_get_child(self, nd_idx, 'ND%2d' % (nd_idx,), self.ITEM_TYPE_ND)
            for nc_idx in range(nc_count):
                self.add_or_get_child(nd_item, nc_idx, 'NC %d' % (nc_idx,), self.ITEM_TYPE_NC)


class NeuronMonitorTop:
    """ Main application class, manages the widgets, the reader thread which gets data from neuron-monitor
        and the state of the application.
    """
    STATE_INVALID = ''
    STATE_LOADING = 'state_loading'
    STATE_MAIN = 'state_main'

    UPDATE_TYPE_DATA = 0
    UPDATE_TYPE_KEYS = 1
    UPDATE_TYPE_SCR_RESIZE = 2
    UPDATE_TYPE_CHANGE_STATE = 3
    UPDATE_TYPE_QUIT = 4

    RUNTIME_TAB_COUNT = 5
    MAX_KEY_EVENTS_IN_QUEUE = 2

    def __init__(self):
        self.json_stdin = os.fdopen(os.dup(0))
        os.dup2(os.open('/dev/tty', os.O_RDONLY), 0)

        self.running = False
        self.screen = ScreenManager()
        self.screen_resized = True
        self.widgets = {}
        self.label_widgets = {}
        self.key_listeners = set()
        self.available_runtimes = {}
        self.available_pids = []
        self.system_data = None
        self.instance_info = None
        self.current_runtime_pid = None
        self.state = None
        self.groups_next_y = 1
        self.monitor_ready = False
        self.refresh_runtime_tab = False
        self.runtime_tab_offset = 0
        self.neuron_device_indices = []
        self.neuroncore_count = 0
        self.logical_neuroncore_config = 1
        self.neuroncore_vers = ''
        self.rebuild_ui = False
        self.reset_ui = False
        self.update_queue = queue.Queue()
        self.data_update_thread = None
        self.ui_update_thread = None
        self.key_increment_lock = threading.Lock()
        self.key_events_in_queue = 0
        self.screen_update_lock = threading.Lock()
        self._change_state(NeuronMonitorTop.STATE_LOADING)
        self.show_driver_err = False
        self.pod_err = ''
        self.efa = EfaMonitor()

    def run(self):
        self.running = True

        def reader_thread(monitor):
            poller = select.poll()
            poller.register(monitor.json_stdin, select.POLLIN)
            while monitor.running:
                if not poller.poll(1):
                    time.sleep(0.1)
                    continue
                line = monitor.json_stdin.readline()
                if not line:
                    continue
                monitor_data = json.loads(line)
                monitor.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_DATA, monitor_data))

        def ui_update_thread(monitor):
            while monitor.running:
                pressed = -1
                while True:
                    with monitor.screen_update_lock:
                        crt_key = monitor.screen.surface.getch()
                    if crt_key == -1:
                        break
                    pressed = crt_key
                if pressed != -1:
                    monitor._add_key_event(pressed)
                screen_resized = monitor.screen.update_screen_size()
                if screen_resized:
                    monitor.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_SCR_RESIZE, None))
                time.sleep(0.1)

        self.data_update_thread = threading.Thread(None, reader_thread, args=(self, ))
        self.data_update_thread.start()
        self.ui_update_thread = threading.Thread(None, ui_update_thread, args=(self, ))
        self.ui_update_thread.start()
        signal.signal(signal.SIGINT, lambda signal, frame: self._stop())
        self._internal_loop()


    def _add_key_event(self, pressed):
        with self.key_increment_lock:
            if self.key_events_in_queue >= NeuronMonitorTop.MAX_KEY_EVENTS_IN_QUEUE:
                return
            self.key_events_in_queue += 1
            self.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_KEYS, pressed))

    def _on_processed_key_event(self):
        with self.key_increment_lock:
            if self.key_events_in_queue > 0:
                self.key_events_in_queue -= 1

    def _call_widget_metric_group_func(self, prefix, create_widgets, runtime_data=None, system_data=None):
        for metric_group_name in Const.METRIC_GROUP_NAMES:
            if runtime_data and metric_group_name not in runtime_data:
                continue
            metric_group_data = runtime_data[metric_group_name] if runtime_data else None
            handler_name = prefix + metric_group_name
            handler_method = getattr(self, handler_name, None)
            if handler_method is None:
                continue
            if create_widgets and metric_group_name not in self.widgets:
                self.widgets[metric_group_name] = {}
            widget_group = self.widgets[metric_group_name]
            handler_method(widget_group, metric_group_data, system_data)

    def _build_ui(self):
        self._call_widget_metric_group_func('_build_ui_', True)

    def _get_neuroncore_info_title(self, mode, aggregated_values):
        metric_name = 'Utilization'
        if mode == NCUsageWidget.NC_USAGE_MODE_FLOPS:
            metric_name = 'Effective FLOPS'
        title = 'NeuronCore %s' % (metric_name)
        if self.neuroncore_vers != '':
            title = 'NeuronCore %s %s' % (self.neuroncore_vers, metric_name)
        aggregated_usage = ''
        if len(aggregated_values) != 0:
            total_usage = sum(aggregated_values)
            if mode == NCUsageWidget.NC_USAGE_MODE_PERCENT:
                aggregated_usage = '(Avg: %s)' % (Utils.format_percentage(total_usage/len(aggregated_values)))
            elif mode == NCUsageWidget.NC_USAGE_MODE_FLOPS:
                aggregated_usage = '(Per NC: %s | Total: %s)' % (Utils.format_flops(total_usage/len(aggregated_values)),
                                                            Utils.format_flops(total_usage))
        return '%s %s' % (title, aggregated_usage)

    def _build_ui_neuroncore_counters(self, widget_group, *_):
        crt_y = self.groups_next_y
        self._set_label('neuroncore_info_title', crt_y, (0, 0),
                        (self._get_neuroncore_info_title(NCUsageWidget.NC_USAGE_MODE_PERCENT, []), ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        self._set_label('neuroncore_info_header', crt_y + 1, (4, 0),
                        ['NC%d' % (nc, ) for nc in range(self.neuroncore_count)],
                        (LabelWidget.ALIGN_CENTER, ), (curses.COLOR_WHITE, ), curses.COLOR_BLACK)
        crt_y += 2
        for crt_idx in self.neuron_device_indices:
            if crt_idx not in widget_group:
                widget_group[crt_idx] = NCUsageWidget(self.screen, crt_y, crt_idx, self.neuroncore_count)
            crt_y += 1
        self.groups_next_y = crt_y + 1
        self._build_ui_efa()

    EFA_MAX_ROWS = 6
    # 'Neuron devices': the Neuron devices on the same PCIe switch as the EFA
    # '*' before an EFA name: in this pod's resource claim
    EFA_COLUMNS = ('EFA (* claimed)', 'Neuron devices', 'Link', 'TX', 'RX', 'Link used', 'Retrans/s', 'Processes')

    def _build_ui_efa(self):
        """ EFA usage: one summary line, a header, and a fixed number of rows (busiest first). """
        if not self.efa.devices:
            return
        crt_y = self.groups_next_y
        self._set_label('efa_title', crt_y, (0, 0), ('EFA Usage', ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        self._set_label('efa_header', crt_y + 1, (2, 0), self.EFA_COLUMNS,
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_WHITE, ), curses.COLOR_BLACK)
        crt_y += 2
        # in a pod, one row per EFA of its resource claim; with no claim (a host), the busiest EFA_MAX_ROWS
        n_claimed = sum(1 for d in self.efa.devices.values() if d['claimed'])
        for idx in range(min(self.EFA_MAX_ROWS, n_claimed or len(self.efa.devices))):
            self._set_label('efa_row_%d' % (idx, ), crt_y, (2, 0), ('', ), (LabelWidget.ALIGN_LEFT, ),
                            (curses.COLOR_WHITE, ), curses.COLOR_BLACK)
            crt_y += 1
        self.groups_next_y = crt_y + 1

    def _update_efa(self):
        if not self.efa.devices or 'efa_title' not in self.label_widgets:
            return
        rows, source, error = self.efa.snapshot()
        devices = self.efa.devices
        in_use = [n for n in devices if rows.get(n, {}).get('users')]
        claimed = [n for n in devices if devices[n]['claimed']]

        def total(names, key):
            return Utils.human_readable_byte_size(sum(rows.get(n, {}).get(key, 0) for n in names), False, False)
        if error:
            title = 'EFA Usage: %d devices · %s' % (len(devices), error)
        elif not rows:
            title = 'EFA Usage: %d devices · reading counters...' % (len(devices), )
        else:
            # 'claimed' = EFAs this pod can open (its resource claim); 'in use' = opened by a process
            title = 'EFA Usage: %d claimed (of %d on the node), %d in use, TX %s/s RX %s/s  (%s)' % (
                len(claimed), len(devices), len(in_use), total(claimed or devices, 'tx_bytes'),
                total(claimed or devices, 'rx_bytes'), source)
        self._update_label_text('efa_title', (title, ), (LabelWidget.ALIGN_LEFT, ))
        # claimed first, then in use, then busiest
        def busy(n):
            return rows.get(n, {}).get('tx_bytes', 0) + rows.get(n, {}).get('rx_bytes', 0)
        order = sorted(devices, key=lambda n: (-devices[n]['claimed'], -(rows.get(n, {}).get('users', 0) > 0),
                                               -busy(n), n))
        if claimed:
            order = [n for n in order if devices[n]['claimed']]
        shown = min(self.EFA_MAX_ROWS, len(order))
        for idx in range(shown):
            name = order[idx]
            dev, r = devices[name], rows.get(name, {})
            nds = sorted(self.efa.neuron_by_switch.get(dev['switch'], []))
            link_bytes = dev['rate_gbps'] * 1e9 / 8
            used = max(r.get('tx_bytes', 0), r.get('rx_bytes', 0)) / link_bytes if link_bytes else 0
            texts = (('* ' if dev['claimed'] else '  ') + name +
                     ('' if dev['state'] in ('ACTIVE', '') else ' (%s)' % (dev['state'], )),
                     ', '.join('ND %d' % (n, ) for n in nds) or '-',
                     '%d Gb/s' % (dev['rate_gbps'], ) if dev['rate_gbps'] else '-',
                     Utils.human_readable_byte_size(r.get('tx_bytes', 0), False, False) + '/s' if r else '-',
                     Utils.human_readable_byte_size(r.get('rx_bytes', 0), False, False) + '/s' if r else '-',
                     Utils.format_percentage(100.0 * used) if r else '-',
                     '%d' % (r.get('retrans_pkts', 0), ) if r else '-',
                     '%d' % (r.get('users', 0), ) if r.get('users') else '-')
            if idx == shown - 1 and len(order) > shown:
                texts = texts[:-1] + ('%s  (+%d more)' % (texts[-1], len(order) - shown), )
            self._update_label_text('efa_row_%d' % (idx, ), texts, (LabelWidget.ALIGN_LEFT, ))

    def _build_ui_memory_used(self, widget_group, *_):
        crt_y = self.groups_next_y

        self._set_label('mem_title', crt_y, (0, 0), ('Memory Usage Summary', ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        crt_y += 1

        titles = ['Host Used Memory', 'Device Used Memory']
        key_names = ['neuron_runtime_used_bytes_host', 'neuron_runtime_used_bytes_device']
        extra_cols = [1, 0]
        title_len = max([len(title) for title in titles])
        for idx, key_name in enumerate(key_names):
            if key_name not in widget_group:
                field_names = ['total'] + Const.MEM_USAGE_FIELD_NAMES[idx]
                field_titles = [Const.field_name_to_field_title(name) for name in field_names]
                widget_group[key_name] = RuntimeMemoryUsageWidget(
                    self.screen, crt_y, field_names, field_titles, titles[idx], title_len, extra_cols[idx])
            else:
                widget_group[key_name].set_pos_y(crt_y)
            crt_y += 1

        crt_y += 1

        self._set_label('models_info_title', crt_y, (0, 0), ('Memory Usage Details', ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        crt_y += 1

        key_name = 'runtime_memory_models'
        model_list_widget = None
        if key_name not in widget_group:
            model_list_widget = ModelMemoryUsageListWidget(self.screen, crt_y, 2, self.neuron_device_indices,
                                                           self.neuroncore_count)
            model_list_widget.set_selection_changed_handler(self)
            widget_group[key_name] = model_list_widget
            self._register_key_listener(model_list_widget)
        else:
            model_list_widget = widget_group[key_name]
            model_list_widget.set_pos_y(crt_y)

    def _build_ui_neuron_runtime_vcpu_usage(self, widget_group, *_):
        crt_y = self.groups_next_y

        self._set_label('vcpu_title', crt_y, (0, 0), ('vCPU Utilization', ),
                        (LabelWidget.ALIGN_LEFT, ), (curses.COLOR_BLACK, ), curses.COLOR_CYAN)
        crt_y += 1

        key_name = 'neuron_runtime_vcpu_usage'
        if key_name not in widget_group:
            widget_group[key_name] = VCPUUsageWidget(self.screen, crt_y)
        else:
            widget_group[key_name].set_pos_y(crt_y)
        self.groups_next_y = crt_y + 3

    def _set_neuroncore_util_highlight(self, nc_widgets, selected_item, show=True):
        if selected_item.item_type == ModelMemoryUsageListWidget.ITEM_TYPE_ND:
            nd_idx = selected_item.identifier
            if nd_idx in nc_widgets:
                nc_widgets[nd_idx].set_highlight(-1, show)
        else:
            while selected_item.item_type != ModelMemoryUsageListWidget.ITEM_TYPE_NC:
                selected_item = selected_item.parent
            if selected_item.item_type == ModelMemoryUsageListWidget.ITEM_TYPE_ND:
                return
            nc_idx = selected_item.identifier
            selected_item = selected_item.parent
            nd_idx = selected_item.identifier
            if nd_idx in nc_widgets:
                nc_widgets[nd_idx].set_highlight(nc_idx, show)

    def on_selection_changed(self, old_item, new_item):
        widgets = self.widgets['neuroncore_counters']
        if old_item is not None:
            self._set_neuroncore_util_highlight(widgets, old_item, False)
        if new_item is not None:
            self._set_neuroncore_util_highlight(widgets, new_item, True)

    def _change_state(self, new_state):
        if self.state is new_state:
            return
        self.state = new_state
        self.widgets = {}
        self.label_widgets = {}
        self.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_CHANGE_STATE, None))
        if new_state == NeuronMonitorTop.STATE_MAIN:
            self.refresh_runtime_tab = True
            self.rebuild_ui = True
        self.screen.clear_screen()

    def _init_aggregated_neuroncore_counters(self, destination):
        destination['error'] = ''
        destination['neuroncores_in_use'] = {}

    def _aggregate_neuroncore_counters(self, destination, source):
        if source['error'] != '':
            return
        nc_dest = destination['neuroncores_in_use']
        nc_src = source['neuroncores_in_use']
        for nc_idx in nc_src:
            nc_util_src = nc_src[nc_idx]['neuroncore_utilization']
            flops_src = float(nc_src[nc_idx]['effective_flops'])
            if nc_idx in nc_dest:
                nc_dest[nc_idx]['neuroncore_utilization'] = min(
                    100, nc_dest[nc_idx]['neuroncore_utilization'] + nc_util_src)
                nc_dest[nc_idx]['effective_flops'] += flops_src
            else:
                nc_dest[nc_idx] = {
                    'neuroncore_utilization': nc_util_src,
                    'effective_flops': flops_src
                }

    def _init_aggregated_memory_used(self, destination):
        destination['error'] = ''
        destination['neuron_runtime_used_bytes'] = {
            'host': 0,
            'neuron_device': 0,
            'usage_breakdown': {
                'host': dict([(key, 0) for key in Const.MEM_USAGE_FIELD_NAMES[0]]),
                'neuroncore_memory_usage': {}
            }
        }
        destination['loaded_models'] = []

    def _aggregate_memory_used(self, destination, source):
        if source['error'] != '':
            return
        used_src = source['neuron_runtime_used_bytes']
        used_dest = destination['neuron_runtime_used_bytes']
        for metric, data in used_src.items():
            if metric != 'usage_breakdown':
                used_dest[metric] += data
            else:
                if metric not in used_src:
                    continue
                simple_aggr_objects = ['host']
                usage_breakdown_dest = used_dest['usage_breakdown']
                for aggr_name in simple_aggr_objects:
                    bd_src = data[aggr_name]
                    bd_dest = usage_breakdown_dest[aggr_name]
                    for src_metric, src_value in bd_src.items():
                        if src_metric not in bd_dest:
                            bd_dest[src_metric] = 0
                        bd_dest[src_metric] += src_value

                if 'neuroncore_memory_usage' not in data:
                    continue

                bd_src = data['neuroncore_memory_usage']
                bd_dest = used_dest['usage_breakdown']['neuroncore_memory_usage']
                for device, device_data in bd_src.items():
                    if device not in bd_dest:
                        bd_dest[device] = {key:0 for key in Const.MEM_USAGE_FIELD_NAMES[1]}
                    for field in device_data:
                        if field not in bd_dest[device]:
                            continue
                        bd_dest[device][field] += device_data[field]

        destination['loaded_models'] += source['loaded_models'][:]

    def _init_aggregated_neuron_runtime_vcpu_usage(self, destination):
        destination['error'] = ''
        destination['vcpu_usage'] = {
            'user': 0,
            'system': 0
        }

    def _aggregate_neuron_runtime_vcpu_usage(self, destination, source):
        if source['error'] != '':
            return
        vcpu_usage_src = source['vcpu_usage']
        vcpu_usage_dest = destination['vcpu_usage']
        for metric, value in vcpu_usage_src.items():
            vcpu_usage_dest[metric] += value

    def _aggregate_current_runtimes(self, current_runtimes):
        aggregated = {}
        aggregated['report'] = {}
        aggregated['error'] = ''
        aggregated['neuron_runtime_tag'] = 'all'
        report = aggregated['report']
        for metric in Const.METRIC_GROUP_NAMES:
            if metric not in report:
                report[metric] = {}
            handler_name = '_init_aggregated_' + metric
            handler_method = getattr(self, handler_name, None)
            handler_method(report[metric])

        for _, runtime_data in current_runtimes.items():
            current_report = runtime_data['report']
            for metric, metric_data in current_report.items():
                handler_name = '_aggregate_' + metric
                handler_method = getattr(self, handler_name, None)
                handler_method(report[metric], metric_data)
        return aggregated

    def _update_available_runtimes(self, monitor_data):
        current_runtimes = {}
        if monitor_data and monitor_data['neuron_runtime_data']:
            for item in monitor_data['neuron_runtime_data']:
                if item['error'] == '':
                    current_runtimes[item['pid']] = item
        current_runtimes[0] = self._aggregate_current_runtimes(current_runtimes)
        prev_runtime_count = len(self.available_runtimes)
        to_delete = []
        for pid, _ in self.available_runtimes.items():
            if pid not in current_runtimes:
                to_delete.append(pid)
                self.refresh_runtime_tab = True
                if self.current_runtime_pid == pid:
                    self.current_runtime_pid = None
        for pid in to_delete:
            del self.available_runtimes[pid]
        for pid, runtime in current_runtimes.items():
            if pid not in self.available_runtimes:
                self.refresh_runtime_tab = True
            self.available_runtimes[pid] = runtime
        if self.current_runtime_pid is None:
            self._select_new_pid(0)
        self.available_pids = sorted(self.available_runtimes.keys())
        self.reset_ui = prev_runtime_count > 0 and not self.available_pids

    def _update_data(self, monitor_data):
        self.system_data = None
        if monitor_data:
            self.system_data = monitor_data['system_data']
            self.instance_info = monitor_data['instance_info']
            self.pod_err = ''
            if self.instance_info['error'] != '':
                if 'ultraserver' in self.instance_info['error'].lower():
                    self.pod_err = self.instance_info['error']
                self.instance_info = None
            neuron_hw_info = monitor_data['neuron_hardware_info']
            self.neuroncore_vers = neuron_hw_info['neuroncore_version']
            if 'logical_neuroncore_config' in neuron_hw_info:
                self.logical_neuroncore_config = neuron_hw_info['logical_neuroncore_config']
            else:
                self.logical_neuroncore_config = 1
            if self.neuroncore_count != neuron_hw_info['neuroncore_per_device_count']:
                self.rebuild_ui = True
                self.neuroncore_count = neuron_hw_info['neuroncore_per_device_count']
            self.neuron_device_indices = Utils.get_neuron_device_indices()
            driver_err = len(self.neuron_device_indices) == 0
            if driver_err != self.show_driver_err:
                self.rebuild_ui = True
            self.show_driver_err = driver_err
        self._update_available_runtimes(monitor_data)
        self.monitor_ready = True

    def _update_neuroncore_counters(self, widget_group, data, _):
        nd_data = {}
        for nc_idx, nc_data in data['neuroncores_in_use'].items():
            pnc_idx = int(nc_idx)

            nd_idx = pnc_idx // self.neuroncore_count
            nc_idx = pnc_idx % self.neuroncore_count
            if nd_idx not in nd_data:
                nd_data[nd_idx] = [[0.0] * self.neuroncore_count for idx in range(2)]

            nd_data[nd_idx][0][nc_idx] = nc_data['neuroncore_utilization']
            nd_data[nd_idx][1][nc_idx] = float(nc_data['effective_flops'])

        usage_mode = None
        aggregated_values = []
        for crt_idx in self.neuron_device_indices:
            if usage_mode is None:
                usage_mode = widget_group[crt_idx].mode
            util_data = [0.0] * self.neuroncore_count
            flops_data = [0.0] * self.neuroncore_count
            if crt_idx in nd_data:
                util_data = nd_data[crt_idx][0]
                flops_data = nd_data[crt_idx][1]
            widget_group[crt_idx].update_data(util_data, flops_data)

            if usage_mode == NCUsageWidget.NC_USAGE_MODE_PERCENT:
                aggregated_values.extend(util_data)
            elif usage_mode == NCUsageWidget.NC_USAGE_MODE_FLOPS:
                aggregated_values.extend(flops_data)

        self._update_label_text('neuroncore_info_title', (self._get_neuroncore_info_title(usage_mode, aggregated_values), ),
                                (LabelWidget.ALIGN_LEFT, ))

    def _reset_neuroncore_counters(self, widget_group, *_):
        for crt_idx in self.neuron_device_indices:
            widget_group[crt_idx].update_data([0.0] * self.neuroncore_count,
                                              [0.0] * self.neuroncore_count)

    def _create_neuroncore_object(self):
        return {
            'host': {key: 0 for key in Const.MEM_USAGE_FIELD_NAMES[0]},
            'neuron_device': {key: 0 for key in Const.MEM_USAGE_FIELD_NAMES[1]},
            'neuron_device_excl': {key: 0 for key in Const.MEM_USAGE_FIELD_NAMES[1]},
            'host_total': 0,
            'device_total': 0,
            'children': {}
        }

    def _initialize_neuron_device_data(self):
        return {
            nd_idx: {
                'host_total': 0,
                'device_total': 0,
                'children': {
                    nc_idx: self._create_neuroncore_object()
                    for nc_idx in range(self.neuroncore_count)
                }
            }
            for nd_idx in self.neuron_device_indices
        }

    def _repartition_memory_usage_data(self, memory_data):
        if memory_data is None:
            return {}
        self.nd_data = self._initialize_neuron_device_data()

        aggr_memory_data = memory_data['neuron_runtime_used_bytes']
        self.abs_total_updated = set()

        for nd_idx in self.neuron_device_indices:
            for nc_idx in range(self.neuroncore_count):
                #absoluet index in the json data from monitor
                abs_data_idx = nd_idx * self.neuroncore_count + nc_idx
                abs_data_idx_str = str(abs_data_idx)
                if 'usage_breakdown' not in aggr_memory_data or \
                    abs_data_idx_str not in aggr_memory_data['usage_breakdown']['neuroncore_memory_usage']:
                    continue
                if abs_data_idx_str in aggr_memory_data['usage_breakdown']['neuroncore_memory_usage']:
                    mem_bd = aggr_memory_data['usage_breakdown']['neuroncore_memory_usage'][abs_data_idx_str]
                for mem_type in Const.MEM_USAGE_FIELD_NAMES[1]:
                    if mem_type not in mem_bd:
                        continue
                    mem_value = mem_bd[mem_type]
                    nd_obj = self.nd_data[nd_idx]
                    nc_obj = nd_obj['children'][nc_idx]
                    nc_obj['neuron_device'][mem_type] += mem_value
                    nd_obj['device_total'] += mem_value
                    nc_obj['device_total'] += mem_value
                    nc_obj['neuron_device_excl'][mem_type] += mem_value
                    self.abs_total_updated.add(abs_data_idx)
        model_data = memory_data['loaded_models']
        self._repartition_model_data(model_data)
        return self.nd_data


    def _repartition_model_data(self,model_data):
        for model in model_data:
            for _, sg_data_obj in model['subgraphs'].items():
                nc_idx = sg_data_obj['neuroncore_index']
                nd_idx = sg_data_obj['neuron_device_index']
                if nd_idx not in self.nd_data:
                    continue
                nd_obj = self.nd_data[nd_idx]
                if nc_idx not in nd_obj['children']:
                    continue
                nc_obj = nd_obj['children'][nc_idx]
                sg_obj = {
                    'is_running': model['is_running'],
                    'name': model['name'],
                    'uuid': model.get('uuid', ''),
                    'host': None,
                    'neuron_device': None,
                    'host_total': sg_data_obj['memory_used_bytes']['host'],
                    'device_total': sg_data_obj['memory_used_bytes']['neuron_device']
                }
                if 'usage_breakdown' in sg_data_obj['memory_used_bytes']:
                    sg_obj['host'] = sg_data_obj['memory_used_bytes']['usage_breakdown']['host']
                    sg_obj['neuron_device'] = sg_data_obj['memory_used_bytes']['usage_breakdown']['neuron_device']
                    for mem_type in Const.MEM_USAGE_FIELD_NAMES_MODEL[0]:
                        if mem_type not in sg_obj['host']:
                            continue
                        nc_obj['host'][mem_type] += sg_obj['host'][mem_type]
                nd_obj['host_total'] += sg_obj['host_total']
                nc_obj['host_total'] += sg_obj['host_total']

                abs_idx = nd_idx * self.neuroncore_count + nc_idx
                if abs_idx not in self.abs_total_updated:
                    nd_obj['device_total'] += sg_obj['device_total']
                    nc_obj['device_total'] += sg_obj['device_total']
                elif sg_obj['neuron_device'] is not None:
                    # subtract every model's usage, so what is left is memory no listed model owns
                    for entry, value in nc_obj['neuron_device_excl'].items():
                        if entry in sg_obj['neuron_device']:
                            nc_obj['neuron_device_excl'][entry] = max(0, value - sg_obj['neuron_device'][entry])
                prev = nc_obj['children'].get(model['model_id'])
                if prev is not None:
                    # several subgraphs of one model on this core: add them up instead of keeping the last
                    self._merge_subgraph(prev, sg_obj)
                else:
                    nc_obj['children'][model['model_id']] = sg_obj

    @staticmethod
    def _merge_subgraph(dst, src):
        dst['is_running'] = dst['is_running'] or src['is_running']
        dst['host_total'] += src['host_total']
        dst['device_total'] += src['device_total']
        for key in ('host', 'neuron_device'):
            if dst[key] is None or src[key] is None:
                dst[key] = dst[key] or src[key]
                continue
            merged = dict(dst[key])
            for field, value in src[key].items():
                merged[field] = merged.get(field, 0) + value
            dst[key] = merged

    def _aggregate_device_memory_used(self, all_device_mem_used):
        aggr_mem_used = dict([(key, 0) for key in Const.MEM_USAGE_FIELD_NAMES[1]])
        for _, device_mem_used in all_device_mem_used.items():
            for entry_name, value in device_mem_used.items():
                if entry_name not in aggr_mem_used:
                    continue
                aggr_mem_used[entry_name] += value
        return aggr_mem_used

    def _update_memory_used(self, widget_group, data, system_data):
        used_bytes_obj = data['neuron_runtime_used_bytes']
        usage_breakdown = used_bytes_obj['usage_breakdown'] if 'usage_breakdown' in used_bytes_obj else None
        host_data = {}
        device_data = {}
        if usage_breakdown:
            host_data = usage_breakdown['host']
            if 'neuron_device' in usage_breakdown:
                device_data = usage_breakdown['neuron_device']
            if 'neuroncore_memory_usage' in usage_breakdown:
                aggr_usage = self._aggregate_device_memory_used(usage_breakdown['neuroncore_memory_usage'])
                for entry, value in aggr_usage.items():
                    if entry not in device_data:
                        device_data[entry] = 0
                    device_data[entry] += value
        host_data['total'] = data['neuron_runtime_used_bytes']['host']
        device_data['total'] = data['neuron_runtime_used_bytes']['neuron_device']
        widget_group['neuron_runtime_used_bytes_host'].update_data(host_data)
        widget_group['neuron_runtime_used_bytes_device'].update_data(device_data)
        key_name = 'runtime_memory_models'
        list_data = self._repartition_memory_usage_data(data)
        widget_group[key_name].update_data(list_data)

    def _reset_memory_used(self, widget_group, *_):
        reset_dict_host = dict([(key, 0) for key in ['total'] + Const.MEM_USAGE_FIELD_NAMES[0]])
        reset_dict_device = dict([(key, 0) for key in ['total'] + Const.MEM_USAGE_FIELD_NAMES[0]])
        for key_name, reset_dict in [('neuron_runtime_used_bytes_host', reset_dict_host),
                                     ('neuron_runtime_used_bytes_device', reset_dict_device)]:
            widget_group[key_name].update_data(reset_dict)
        key_name = 'runtime_memory_models'
        widget_group[key_name].update_data({})

    def _update_neuron_runtime_vcpu_usage(self, widget_group, data, system_data):
        runtime_vcpu_usage = {
            'user': data['vcpu_usage']['user'],
            'system': data['vcpu_usage']['system'],
        }

        avg_system_vcpu_usage = system_data['vcpu_usage']['average_usage']
        vcpu_usage_aggregation = {
            'user': ['user', 'nice'],
            'system': ['system', 'io_wait', 'irq', 'soft_irq']
        }
        system_vcpu_usage = {}
        for name, fields in vcpu_usage_aggregation.items():
            system_vcpu_usage[name] = sum([avg_system_vcpu_usage[x] for x in fields])

        key_name = 'neuron_runtime_vcpu_usage'
        widget_group[key_name].update_data(system_vcpu_usage, runtime_vcpu_usage)

    def _reset_neuron_runtime_vcpu_usage(self, widget_group, *_):
        key_name = 'neuron_runtime_vcpu_usage'
        widget_group[key_name].update_data(None, {'user': 0, 'system': 0})

    def _set_label(self, name, pos_y, margins, texts, aligns, colors, bg_color):
        if name not in self.label_widgets:
            self.label_widgets[name] = LabelWidget(self.screen, pos_y, margins, texts, aligns, colors, bg_color)
        else:
            self.label_widgets[name].set_pos_y(pos_y)
            self.label_widgets[name].set_texts(texts, aligns)
            self.label_widgets[name].set_colors(colors, bg_color)

    def _unset_label(self, name):
        if name in self.label_widgets:
            self.label_widgets[name].clear()
            del self.label_widgets[name]

    def _update_label_text(self, name, texts, align):
        if name in self.label_widgets:
            self.label_widgets[name].set_texts(texts, align)

    def _update_header(self):
        crt_y = self.groups_next_y
        if self.instance_info is not None:
            instance_id = self.instance_info['instance_id']
            if os.environ.get('KUBERNETES_SERVICE_HOST') and os.environ.get('HOSTNAME'):
                instance_id = os.environ['HOSTNAME']        # the pod name
            instance_type = self.instance_info['instance_type']
            header = 'neuron-top %s running on %s (%s)' % (NEURON_TOP_VERSION, instance_id, instance_type, )
            if 'ultraserver_id' in self.instance_info:
                ultraserver_id = self.instance_info['ultraserver_id']
                if ultraserver_id != '':
                    header += ', ultraserver %s' % ultraserver_id
            self._set_label('main_header', crt_y, (0, 0), (header, ),
                (LabelWidget.ALIGN_CENTER, ), (curses.COLOR_WHITE, ), curses.COLOR_BLUE)
        else:
            self._set_label('main_header', crt_y, (0, 0),
                            ('neuron-top %s' % (NEURON_TOP_VERSION, ), ), (LabelWidget.ALIGN_CENTER, ),
                            (curses.COLOR_WHITE, ), curses.COLOR_BLUE)
        self.groups_next_y += 2

    def _update_driver_error_header(self):
        crt_y = self.groups_next_y
        for i in range(len(DRIVER_ERR_MSG)):
            label_name = "driver_err_msg"+str(i)
            if self.show_driver_err:
                self._set_label(label_name, crt_y, (0, 0),
                                (DRIVER_ERR_MSG[i], ), (LabelWidget.ALIGN_CENTER, ),
                                (curses.COLOR_BLACK, ), curses.COLOR_RED)
            else:
                self._unset_label(label_name)
            crt_y += 1
        if self.show_driver_err:
            self.groups_next_y += len(DRIVER_ERR_MSG) + 1

    def _update_pod_error_header(self):
        if self.pod_err is None or self.pod_err == '':
            return
        highlight = curses.COLOR_RED
        pod_err = 'Error: ' + self.pod_err
        if 'initializing' in self.pod_err.lower():
            highlight = curses.COLOR_YELLOW
            pod_err = 'Warning: ' + self.pod_err
        self._set_label('pod_err_msg', self.groups_next_y, (0, 0),
                        (pod_err, ), (LabelWidget.ALIGN_CENTER, ), 
                        (curses.COLOR_BLACK, ), highlight)
        self.groups_next_y += 2

    def _get_current_pid_index(self):
        for idx, pid in enumerate(self.available_pids):
            if pid == self.current_runtime_pid:
                return idx
        return -1

    def _update_tab_display(self):
        if not self.refresh_runtime_tab and not self.screen_resized:
            return
        current_idx = self._get_current_pid_index()
        pids_strings = [' ' + ('[%d]:' % (idx + 1) if idx < 9 else '') +
                        self.available_runtimes[pid]['neuron_runtime_tag'] + ' '
                        for idx, pid in enumerate(self.available_pids)]
        pids_colors = [curses.COLOR_WHITE for _ in range(self.RUNTIME_TAB_COUNT)]
        start, end = 0, 0
        if len(pids_strings) < self.RUNTIME_TAB_COUNT:
            pids_strings += [''] * (self.RUNTIME_TAB_COUNT - len(pids_strings))
        if current_idx != -1:
            start, end = (self.runtime_tab_offset, self.runtime_tab_offset + self.RUNTIME_TAB_COUNT - 1)
            if current_idx < start:
                start = current_idx
                end = current_idx + self.RUNTIME_TAB_COUNT - 1
            elif current_idx > end:
                end = current_idx
                start = current_idx - self.RUNTIME_TAB_COUNT + 1
            pids_colors[current_idx - start] = curses.COLOR_GREEN
            pids_strings[current_idx] = ">" + pids_strings[current_idx][1:-1] + "<"
        self.runtime_tab_offset = start
        crt_y = self.screen.get_height() - 2
        self._set_label('runtimes', crt_y, (0, 0), tuple(['Neuron Apps'] + pids_strings[start:end + 1]),
                        (LabelWidget.ALIGN_LEFT, ), tuple([curses.COLOR_YELLOW] + pids_colors),
                        curses.COLOR_BLUE)

    def _reset(self):
        self._call_widget_metric_group_func('_reset_', False)

    def _select_new_pid(self, pid):
        if self.current_runtime_pid != pid:
            self.current_runtime_pid = pid
            self.refresh_runtime_tab = True
            self.reset_ui = True

    def _select_tab(self, tab_index):
        if tab_index < len(self.available_pids):
            self._select_new_pid(self.available_pids[tab_index])

    def _select_tab_next_prev(self, go_next):
        if not self.available_pids:
            return
        current_idx = self._get_current_pid_index()
        if current_idx == -1:
            current_idx = 0
        else:
            if go_next:
                current_idx = (current_idx + 1) % len(self.available_pids)
            else:
                current_idx = (len(self.available_pids) + (current_idx - 1)) % len(self.available_pids)
        self._select_new_pid(self.available_pids[current_idx])

    def _show_usage_label(self):
        crt_y = self.screen.get_height() - 1
        self._set_label('info', crt_y, (0, 0),
                        ('q: quit',
                         'arrows: move tree selection',
                         'enter: expand/collapse tree item',
                         'x: expand/collapse entire tree',
                         'a/d: previous/next tab',
                         'f: FLOPS on/off',
                         '1-9: select tab'), (LabelWidget.ALIGN_CENTER, ),
                        (curses.COLOR_WHITE, ), curses.COLOR_BLUE)

    def _update_footer(self):
        self._update_tab_display()
        self._show_usage_label()

    def _update_state_loading(self):
        widget_name = 'loading'
        y_pos = self.screen.get_height() // 2
        self._set_label(widget_name, y_pos, (20, 20),
                        ('LOADING', ), (LabelWidget.ALIGN_CENTER, ),
                        (curses.COLOR_WHITE, ), curses.COLOR_BLUE)
        if self.monitor_ready:
            self._change_state(NeuronMonitorTop.STATE_MAIN)

    def _update_state_main(self):
        self.groups_next_y = 1
        self._update_header()
        self._update_pod_error_header()
        self._update_driver_error_header()
        self._update_footer()
        if self.reset_ui:
            self._reset()
            self.reset_ui = False
        if self.rebuild_ui:
            self._remove_all_key_listeners()
            self._build_ui()
            self.rebuild_ui = False
        self._update_efa()
        if self.current_runtime_pid not in self.available_runtimes:
            return
        runtime_data = self.available_runtimes[self.current_runtime_pid]
        if runtime_data['error'] != '':
            return
        report = runtime_data['report']
        self._call_widget_metric_group_func('_update_', False, report, self.system_data)
        self.refresh_runtime_tab = False

    def _update_current_state(self):
        getattr(self, '_update_%s' % (self.state, ), lambda: None)()

    def _update(self):
        if not self.running:
            return
        self._update_current_state()

    def _handle_input(self, key):
        if not self._on_key(key):
            for target in self.key_listeners:
                target.on_key(key)

    def _switch_util_mode(self):
        if 'neuroncore_counters' not in self.widgets:
            return
        widgets = self.widgets['neuroncore_counters']
        for widget in widgets.values():
            widget.switch_mode()

    def _on_key(self, key):
        if key == ord('q') or key == ord('Q'):
            self._stop()
            return True
        if key > ord('0') and key <= ord('9'):
            self._select_tab(key - ord('1'))
        elif key == ord('a') or key == ord('A'):
            self._select_tab_next_prev(False)
        elif key == ord('d') or key == ord('D'):
            self._select_tab_next_prev(True)
        elif key == ord('f') or key == ord('F'):
            self._switch_util_mode()
        return False

    def _register_key_listener(self, target):
        self.key_listeners.add(target)

    def _remove_all_key_listeners(self):
        self.key_listeners = set()

    def _refresh_screen(self):

        def update_widgets_in_dict(widget_dict):
            for _, widget in widget_dict.items():
                if self.screen_resized:
                    widget.on_screen_resize()
                if widget.refresh_requested():
                    widget.refresh()
                    widget.on_post_refresh()

        if self.screen_resized:
            self.screen.clear_screen()
            if self.state is NeuronMonitorTop.STATE_MAIN:
                self._update_footer()

        for _, widget_group in self.widgets.items():
            update_widgets_in_dict(widget_group)
        update_widgets_in_dict(self.label_widgets)
        self.screen.refresh()

    def _internal_loop(self):
        while self.running:
            update_type, update_arg = self.update_queue.get()
            if update_type is NeuronMonitorTop.UPDATE_TYPE_DATA:
                self._update_data(update_arg)
            elif update_type is NeuronMonitorTop.UPDATE_TYPE_KEYS:
                self._handle_input(update_arg)
            elif update_type is NeuronMonitorTop.UPDATE_TYPE_SCR_RESIZE:
                self.screen_resized = True
            if not self.running:
                break
            self._update()
            with self.screen_update_lock:
                self._refresh_screen()
            self.screen_resized = False
            if update_type is NeuronMonitorTop.UPDATE_TYPE_KEYS:
                self._on_processed_key_event()

        self.data_update_thread.join()
        self.ui_update_thread.join()

    def _stop(self):
        self.running = False
        self.update_queue.put((NeuronMonitorTop.UPDATE_TYPE_QUIT, None))


def main():
    parser = argparse.ArgumentParser(description='neuron-top ' + NEURON_TOP_VERSION)
    parser.add_argument('--version', action='version', version='neuron-top ' + NEURON_TOP_VERSION)
    # internal: run by OpAttributionRunner in a separate process
    parser.add_argument('--op-attribution', metavar='NEFF', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.op_attribution:
        print(json.dumps(OpAttribution.compute(args.op_attribution)))
        return

    app = NeuronMonitorTop()
    app.run()


if __name__ == '__main__':
    main()