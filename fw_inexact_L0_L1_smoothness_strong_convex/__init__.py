"""
Frank-Wolfe with composite gradient inexactness and (L0, L1)-smoothness.
"""

from .objectives import ShiftedPointProblem
from .oracles import (
    MixedInexactOracle,
    MixedInexactObjectiveWrapper,
    LowPrecisionCompressedOracle,
    ShiftedPointOracle,
)
from .algorithms import (
    ProposedMixedInexactFW,
    ClassicalMixedInexactFW,
    AdaptiveMixedInexactFW,
)
from .stopping_rules import InexactDualGapStoppingRule

__all__ = [
    "ShiftedPointProblem",
    "MixedInexactOracle",
    "MixedInexactObjectiveWrapper",
    "LowPrecisionCompressedOracle",
    "ShiftedPointOracle",
    "ProposedMixedInexactFW",
    "ClassicalMixedInexactFW",
    "AdaptiveMixedInexactFW",
    "InexactDualGapStoppingRule",
]
