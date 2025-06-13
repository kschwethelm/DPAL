from typing import Tuple
from functools import partial
from torch import nn
import numpy as np

from .nn import (
    FieldType,
    EquivariantModule,
    SequentialModule,
    R2Conv,
    GroupNorm,
    BatchNorm,
    IIDInstanceNorm,
    Mish,
    FourierELU,
    GroupPooling,
    PointwiseAvgPoolAntialiased,
    PointwiseMaxPool,
    DisentangleModule,
    RestrictionModule,
)
from .group_theory import Representation

__all__ = [
    "Restriction",
    "EquivariantConv",
    "EquivariantNorm",
    "EquivariantPool",
    "EquivariantConvBlock",
]


def only_zero_freq(repr: Representation):
    for irr in repr.irreps:
        idx = list(
            filter(
                lambda x: repr.group.irreps()[x].name == f"irrep_{irr[0]},{irr[1]}",
                range(len(repr.group.irreps())),
            )
        )
        if repr.group.irreps()[idx[0]].attributes["frequency"] != 0:
            return False
    return True


class Restriction(EquivariantModule):
    def __init__(
        self, in_type: FieldType, group: str, rotation: int, restrict: str = None
    ):
        super().__init__()
        self.in_type = in_type

        if not restrict:
            self.restrict = nn.Identity()
            self.out_type = self.in_type
        else:
            layers = list()

            if restrict == "reflection":
                assert (
                    group != "cyclic"
                ), "Cyclic groups can't be restricted to reflection."

                subgroup_id = (np.pi, 1) if group == "orthogonal" else (0, 1)
            elif restrict == "halved":
                assert (
                    group != "orthogonal"
                ), "Orthogonal group can't be restricted by halve."
                assert (
                    rotation % 2 == 0
                ), f"Number of rotations ({rotation}) is not divisible by 2."

                subgroup_id = (
                    (0, rotation // 2) if group == "dihedral" else (rotation // 2)
                )
            else:  # restrict to invariant case
                subgroup_id = (None, 1) if group == "dihedral" else 1

            layers.append(RestrictionModule(self.in_type, subgroup_id))
            layers.append(DisentangleModule(layers[-1].out_type))
            self.restrict = SequentialModule(*layers)
            self.out_type = self.restrict.out_type

    def forward(self, x):
        return self.restrict(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantConv(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        frequency: int = None,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class

        # Induced
        if self.in_type.gspace.fibergroup.name == "SO(2)":
            act_func = FourierELU(
                self.in_type.gspace,
                out_channels,
                irreps=self.in_type.gspace.fibergroup.bl_irreps(frequency),
                N=16,
            )
            out_type = act_func.in_type
            self.conv = R2Conv(
                self.in_type,
                out_type,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                sigma=None,
                frequencies_cutoff=None,
            )
        # Cyclic and Dihedral Groups
        else:
            out_type = FieldType(
                self.in_type.gspace,
                [self.in_type.gspace.regular_repr] * out_channels,
            )

            self.conv = R2Conv(
                self.in_type,
                out_type,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=bias,
                sigma=None,
                frequencies_cutoff=lambda r: 3 * r,
            )

        self.out_type = self.conv.out_type

    def forward(self, x):
        return self.conv(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        num_groups: int = None,
        affine: bool = False,
    ):
        super().__init__()
        self.in_type = in_type

        if num_groups:  # group norm
            norm = GroupNorm
            param = num_groups
        elif self.in_type.gspace.fibergroup.name == "SO(2)":  # instance norm
            norm = IIDInstanceNorm
            param = affine
        else:  # batch norm
            norm = BatchNorm
            param = affine

        self.norm = norm(self.in_type, param)
        self.out_type = self.norm.out_type

    def forward(self, x):
        return self.norm(x)

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantPool(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        pool_size: int = None,
        invariant_map: bool = False,
    ):
        super().__init__()
        self.in_type = in_type
        self.map = None

        if self.in_type.gspace.fibergroup.name == "SO(2)":
            pool = partial(PointwiseAvgPoolAntialiased, sigma=0.66, stride=2)
        else:
            pool = partial(PointwiseMaxPool, kernel_size=pool_size)

        if invariant_map:
            self.map = GroupPooling(self.in_type)
            if pool_size is not None:
                self.pool = pool(self.map.out_type)
                self.out_type = self.pool.out_type
            else:
                self.pool = nn.Identity()
                self.out_type = self.map.out_type
        elif pool_size is not None:
            self.pool = pool(in_type)
            self.out_type = self.pool.out_type
        else:
            raise ValueError(
                "EquivariantPool got no values for pooling and/or invariant mapping."
            )

    def forward(self, x):
        if self.map:
            x = self.map(x)
        x = self.pool(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape


class EquivariantConvBlock(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_channels: int,
        frequency: int = None,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        num_groups: int = None,
        pool_size: int = None,
        invariant_map: bool = False,
    ):
        super().__init__()
        self.in_type = in_type  # declaration required by base class
        self.conv = EquivariantConv(
            self.in_type,
            out_channels,
            frequency=frequency,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        # Induced
        if self.in_type.gspace.fibergroup.name == "SO(2)":
            self.act_func = FourierELU(
                self.in_type.gspace,
                out_channels,
                irreps=self.in_type.gspace.fibergroup.bl_irreps(frequency),
                N=16,
            )
        # Cyclic and Dihedral Groups
        else:
            self.act_func = Mish(self.conv.out_type)

        self.norm = EquivariantNorm(
            self.act_func.out_type, num_groups=num_groups, affine=False
        )

        if pool_size is not None or invariant_map:
            self.pool = EquivariantPool(self.norm.out_type, pool_size, invariant_map)
            self.out_type = self.pool.out_type
        else:
            self.pool = nn.Identity()
            self.out_type = self.norm.out_type

    def forward(self, x):
        x = self.conv(x)
        x = self.act_func(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return input_shape
