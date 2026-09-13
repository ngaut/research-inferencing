"""flmloop — FLM's fly-connectome reservoir generalized to a looped block that learns at test time.

Numpy core (no torch needed): kernel, graph, reservoir, plasticity, probes.
Torch path (imported on demand): torchloop (differentiable loop), llm (backbone wrapper).
"""
from .graph import Graph, neurotransmitter_signs, random_signs
from .reservoir import LoopedReservoir
from .kernel import load_kernel, spmm

__all__ = ['Graph', 'neurotransmitter_signs', 'random_signs', 'LoopedReservoir', 'load_kernel', 'spmm']
