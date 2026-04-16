"""
Topological Quantum Neural Networks (TQNN)

Interactive simulators and visualization tools for TQFT-based classification.
"""

__version__ = "0.1.0"

from tqnn.processor import (
    TQNNProcessor as TQNNProcessor,
    SpinNetworkState as SpinNetworkState,
    ClassPrototype as ClassPrototype,
    SpinNetworkMode as SpinNetworkMode,
)
from tqnn.invariants import (
    jonesPolynomial as jonesPolynomial,
    bracketPolynomial as bracketPolynomial,
)
