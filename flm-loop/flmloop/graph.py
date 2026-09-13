"""Connectome graphs: FLM's prepared MaleCNS arrays, toy graphs, controls, neurotransmitter sign priors.

W[post, pre] holds anatomical contact counts divided by each neuron's total incoming
contacts (FLM's normalization): rows sum to 1 wherever in-degree > 0, entries are >= 0.
Array order is kept exactly as stored so summation order matches FLM.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

# Sign prior from predicted neurotransmitters (MaleCNS body-neurotransmitters table).
# GABA and glutamate are inhibitory in the fly central brain, histamine at photoreceptor
# synapses; acetylcholine is the main excitatory transmitter. Neuromodulators keep FLM's
# unsigned (+1) default because their effect is not a fast synaptic sign.
NEUROTRANSMITTER_SIGNS = {'acetylcholine': 1.0, 'gaba': -1.0, 'glutamate': -1.0, 'histamine': -1.0,
                          'dopamine': 1.0, 'serotonin': 1.0, 'octopamine': 1.0, 'unclear': 1.0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _as_csr(matrix):
    matrix = sparse.csr_matrix(matrix)
    if matrix.data.dtype != np.float32:
        matrix.data = matrix.data.astype(np.float32)
    if matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32)
    if matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix


class Graph:
    def __init__(self, matrix, ids=None, manifest=None, name='graph'):
        self.matrix = _as_csr(matrix)
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError('Connectome matrix must be square.')
        self.n = self.matrix.shape[0]
        self.ids = np.arange(self.n, dtype=np.int64) if ids is None else np.asarray(ids, np.int64)
        if len(self.ids) != self.n:
            raise ValueError('ids length must match the matrix.')
        self.manifest = dict(manifest or {})
        self.name = name
        self._transpose = None

    @classmethod
    def from_flm_cache(cls, folder, verify=True):
        """Load FLM's prepared graph (scripts/prepare_graph.py output): ids/data/indices/indptr .npy + manifest.json."""
        folder = Path(folder)
        manifest = json.loads((folder / 'manifest.json').read_text())
        if verify:
            for name, digest in manifest['arrays'].items():
                if sha256(folder / name) != digest:
                    raise ValueError(f'Graph integrity check failed: {name}')
        ids = np.load(folder / 'ids.npy')
        if ids.dtype != np.int64 or np.any(np.diff(ids) <= 0):
            raise ValueError('Neuron IDs must be sorted, exact int64 values.')
        arrays = tuple(np.load(folder / f'{name}.npy') for name in ('data', 'indices', 'indptr'))
        matrix = sparse.csr_matrix(arrays, shape=(len(ids), len(ids)))
        if matrix.nnz != manifest['directed_edges']:
            raise ValueError('Edge count differs from manifest.')
        return cls(matrix, ids, manifest, name=manifest.get('release', 'connectome').replace(' ', '_').lower())

    @classmethod
    def from_spec(cls, spec, seed=0, verify=True):
        """'random:N[:degree]' for a seeded random graph, otherwise a prepared FLM graph folder."""
        spec = str(spec)
        if spec.startswith('random:'):
            parts = spec.split(':')
            return cls.random(int(parts[1]), int(parts[2]) if len(parts) > 2 else 12, seed=seed, name=spec.replace(':', '-'))
        return cls.from_flm_cache(spec, verify=verify)

    @classmethod
    def toy(cls):
        """FLM's three-neuron chain 0 -> 1 -> 2 with exact large integer IDs (used by its tests)."""
        return cls(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                   np.array([2 ** 53 + 1, 2 ** 53 + 3, 2 ** 53 + 5], np.int64), name='toy')

    @classmethod
    def random(cls, n=256, degree=8, seed=0, name='random'):
        """Random sparse graph with FLM's incoming normalization, for tests and small experiments."""
        rng = np.random.default_rng(seed)
        post = np.repeat(np.arange(n), degree)
        pre = rng.integers(0, n, n * degree)
        keep = pre != post
        counts = rng.integers(1, 50, keep.sum()).astype(np.float32)
        matrix = sparse.coo_matrix((counts, (post[keep], pre[keep])), shape=(n, n)).tocsr()
        sums = np.asarray(matrix.sum(axis=1)).ravel()
        matrix.data /= np.repeat(np.maximum(sums, 1), np.diff(matrix.indptr))
        return cls(matrix, name=name)

    def transpose(self):
        if self._transpose is None:
            self._transpose = _as_csr(self.matrix.T.tocsr())
        return self._transpose

    def in_degree(self):
        return np.diff(self.matrix.indptr)

    def out_degree(self):
        return np.bincount(self.matrix.indices, minlength=self.n)

    def rewired(self, seed=0):
        """Degree-sequence-preserving random control: the multiset of presynaptic partners is
        permuted across all edges, so every neuron keeps its exact in-degree, incoming weights
        and out-degree while the specific wiring is destroyed."""
        rng = np.random.default_rng(seed)
        matrix = sparse.csr_matrix((self.matrix.data.copy(), self.matrix.indices[rng.permutation(self.matrix.nnz)],
                                    self.matrix.indptr.copy()), shape=self.matrix.shape)
        return Graph(matrix, self.ids, {**self.manifest, 'control': f'rewired seed {seed}'}, name=f'{self.name}-rewired{seed}')

    def summary(self):
        indeg = self.in_degree()
        return {'name': self.name, 'neurons': int(self.n), 'directed_edges': int(self.matrix.nnz),
                'mean_in_degree': float(indeg.mean()), 'isolated_inputs': int((indeg == 0).sum()),
                'row_sum_max': float(np.abs(np.asarray(self.matrix.sum(axis=1)).ravel()).max())}


def neurotransmitter_signs(ids, feather_path, column='consensus_nt', signs=None, default=1.0):
    """Per-neuron efficacy sign from the MaleCNS neurotransmitter table (+1 excitatory, -1 inhibitory).

    Returns (signs array aligned with ids, Counter of labels among ids). Unknown bodies get `default`.
    """
    import pyarrow.feather as feather
    table = feather.read_table(str(feather_path), columns=['body', column])
    body = np.asarray(table['body'], np.int64)
    label = np.asarray(table[column].to_pylist(), dtype=object)
    order = np.argsort(body)
    body, label = body[order], label[order]
    ids = np.asarray(ids, np.int64)
    position = np.searchsorted(body, ids)
    found = (position < len(body)) & (body[np.minimum(position, len(body) - 1)] == ids)
    mapping = {**NEUROTRANSMITTER_SIGNS, **(signs or {})}
    labels = [label[p] if f and label[p] is not None else None for p, f in zip(position, found)]
    values = np.array([mapping.get(l, default) if l is not None else default for l in labels], np.float32)
    return values, Counter('unknown' if l is None else l for l in labels)


def random_signs(n, fraction_negative, seed=0):
    """Control for the neurotransmitter prior: the same share of -1 efficacies, placed at random."""
    rng = np.random.default_rng(seed)
    values = np.ones(n, np.float32)
    values[rng.permutation(n)[:int(round(fraction_negative * n))]] = -1.0
    return values
