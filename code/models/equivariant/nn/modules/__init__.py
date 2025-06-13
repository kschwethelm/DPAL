from .equivariant_module import EquivariantModule
from .multiple_module import MultipleModule
from .reshuffle_module import ReshuffleModule
from .restriction_module import RestrictionModule
from .disentangle_module import DisentangleModule
from .sequential_module import SequentialModule
from .identity_module import IdentityModule
from .masking_module import MaskModule

from .conv import R2Conv

from .nonlinearities import GatedNonLinearity1
from .nonlinearities import GatedNonLinearity2
from .nonlinearities import InducedGatedNonLinearity
from .nonlinearities import NormNonLinearity
from .nonlinearities import InducedNormNonLinearity
from .nonlinearities import PointwiseNonLinearity
from .nonlinearities import VectorFieldNonLinearity
from .nonlinearities import ReLU
from .nonlinearities import ELU
from .nonlinearities import Mish
from .nonlinearities import FourierPointwise
from .nonlinearities import FourierELU
from .nonlinearities import QuotientFourierPointwise
from .nonlinearities import QuotientFourierELU

from .normalization import GroupNorm, InducedNormGroupNorm, GroupStandardization
from .normalization import BatchNorm, InducedNormBatchNorm
from .normalization import IIDInstanceNorm

from .pooling import NormAvgPool
from .pooling import NormMaxPool
from .pooling import PointwiseAvgPool
from .pooling import PointwiseAvgPoolAntialiased
from .pooling import PointwiseAdaptiveAvgPool
from .pooling import PointwiseMaxPool
from .pooling import PointwiseMaxPoolAntialiased

from .invariantmaps import GroupPooling
from .invariantmaps import MaxPoolChannels
from .invariantmaps import NormPool
from .invariantmaps import InducedNormPool

from .dropout import FieldDropout
from .dropout import PointwiseDropout


__all__ = (
    [
        "EquivariantModule",
        "MultipleModule",
        "ReshuffleModule",
        "RestrictionModule",
        "DisentangleModule",
        "SequentialModule",
        "IdentityModule",
        "MaskModule",
    ]
    + ["R2Conv"]
    + [
        "GatedNonLinearity1",
        "GatedNonLinearity2",
        "InducedGatedNonLinearity",
        "NormNonLinearity",
        "InducedNormNonLinearity",
        "PointwiseNonLinearity",
        "VectorFieldNonLinearity",
        "ReLU",
        "ELU",
        "Mish",
        "FourierPointwise",
        "FourierELU",
        "QuotientFourierPointwise",
        "QuotientFourierELU",
    ]
    + [
        "GroupNorm",
        "InducedNormGroupNorm",
        "GroupStandardization",
        "BatchNorm",
        "InducedNormBatchNorm",
        "IIDInstanceNorm",
    ]
    + [
        "NormAvgPool",
        "NormMaxPool",
        "PointwiseAvgPool",
        "PointwiseAvgPoolAntialiased",
        "PointwiseAdaptiveAvgPool",
        "PointwiseMaxPool",
        "PointwiseMaxPoolAntialiased",
        "GroupPooling",
        "MaxPoolChannels",
        "NormPool",
        "InducedNormPool",
    ]
    + ["FieldDropout", "PointwiseDropout"]
)
