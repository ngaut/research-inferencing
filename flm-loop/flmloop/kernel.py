"""Optional native CSR x lanes kernel; the SciPy fallback computes the same recurrence.

Both paths sum each row in stored CSR order, so they agree to float rounding (the native
kernel forbids fused multiply-add; a SciPy build that fuses may differ in the last bits).
"""
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'native' / 'graph_lanes.c'
BUILD_DIR = ROOT / 'build'
MAX_LANES = 64
FLAGS = ['-O3', '-ffp-contract=off', '-shared', '-fPIC']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(force=False, compiler=None, openmp=True):
    """Compile the kernel once; a manifest pins source and binary hashes like FLM does."""
    BUILD_DIR.mkdir(exist_ok=True)
    binary = BUILD_DIR / 'graph-lanes.so'
    manifest = BUILD_DIR / 'manifest.json'
    if not force and binary.exists() and manifest.exists():
        m = json.loads(manifest.read_text())
        if m.get('source_sha256') == sha256(SOURCE) and m.get('binary_sha256') == sha256(binary):
            return binary
    cc = compiler or os.environ.get('CC', 'cc')
    attempts = ([FLAGS + ['-fopenmp']] if openmp else []) + [FLAGS]
    errors = []
    for flags in attempts:
        result = subprocess.run([cc, *flags, str(SOURCE), '-o', str(binary)], capture_output=True, text=True)
        if result.returncode == 0:
            manifest.write_text(json.dumps({'source_sha256': sha256(SOURCE), 'binary_sha256': sha256(binary),
                                            'compiler': cc, 'flags': ' '.join(flags)}, indent=2))
            return binary
        errors.append(result.stderr.strip())
    raise RuntimeError('Could not build the native kernel:\n' + '\n'.join(errors))


class LanesKernel:
    def __init__(self, binary=None):
        binary = Path(binary or BUILD_DIR / 'graph-lanes.so')
        m = json.loads(binary.with_name('manifest.json').read_text())
        if sha256(binary) != m['binary_sha256'] or sha256(SOURCE) != m['source_sha256']:
            raise ValueError('Native kernel integrity mismatch; rebuild with flmloop.kernel.build(force=True).')
        self.flags = m['flags']
        self.function = ctypes.CDLL(str(binary)).flmloop_csr_lanes
        self.function.argtypes = [ctypes.c_int32, ctypes.c_int32] + [ctypes.c_void_p] * 5
        self.function.restype = None

    def __call__(self, matrix, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        n, lanes = x.shape
        if not (1 <= lanes <= MAX_LANES) or n != matrix.shape[1]:
            raise ValueError('Kernel expects X of shape (n, 1..64).')
        if matrix.indptr.dtype != np.int32 or matrix.indices.dtype != np.int32 or matrix.data.dtype != np.float32:
            raise TypeError('Kernel expects int32 indptr/indices and float32 data.')
        indptr = np.ascontiguousarray(matrix.indptr); indices = np.ascontiguousarray(matrix.indices)
        data = np.ascontiguousarray(matrix.data)
        y = np.empty((matrix.shape[0], lanes), np.float32)
        self.function(matrix.shape[0], lanes, indptr.ctypes.data, indices.ctypes.data, data.ctypes.data,
                      x.ctypes.data, y.ctypes.data)
        return y


def load_kernel(auto_build=True):
    """Return the native kernel, or None when no compiler is available (SciPy path is used)."""
    try:
        return LanesKernel(build() if auto_build else None)
    except Exception:
        return None


def spmm(matrix, x, kernel=None):
    """W @ X for X of shape (n,) or (n, lanes), float32."""
    x = np.ascontiguousarray(x, np.float32)
    if x.ndim == 1:
        return spmm(matrix, x[:, None], kernel)[:, 0]
    if (kernel is not None and x.shape[1] <= MAX_LANES and matrix.indptr.dtype == np.int32
            and matrix.indices.dtype == np.int32 and matrix.data.dtype == np.float32):
        return kernel(matrix, x)
    return np.asarray(matrix @ x, np.float32)
