# Source: https://github.com/hlzl/equivariant/tree/main
import warnings
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from .equivariant.nn import (
    rot2dOnR2,
    flipRot2dOnR2,
    FieldType,
    SequentialModule,
    MaskModule,
    GroupTensor,
)
from .equivariant.layers import (
    Restriction,
    EquivariantPool,
    EquivariantNorm,
    EquivariantConvBlock,
)


class EquivariantResNet9(nn.Module):
    def __init__(
        self,
        group: str = "dihedral",  # "cyclic", "dihedral", "orthogonal"
        rotation: int = 4,  # discrete number or frequency
        fix_params: bool = True,
        restrict: Union[None, str] = "halved",  # "invariant", "reflection", "halved"
        input_channels: int = 3,
        layout: Tuple[int] = (16, 32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: Tuple[int] = (8, 8, 8),
        num_classes: int = 10,
        spatial_dims: int = 32,
    ):
        super().__init__()
        assert group in ["cyclic", "dihedral", "orthogonal"]
        assert restrict in [None, "invariant", "reflection", "halved"]

        # Get group spaces for specified rotations and flips
        if group == "cyclic":
            gspace = rot2dOnR2(rotation)
        elif group == "dihedral":
            gspace = flipRot2dOnR2(rotation)
        else:  # SO(2)
            gspace = rot2dOnR2(-1)

        # Fix number of parameters for all groups
        num_channels = np.array(layout)
        fix_constant = 1.5
        if fix_params:  # values heuristically found
            for l in range(len(num_channels)):
                if group == "orthogonal":
                    num_channels[l] = int(
                        num_channels[l] * np.sqrt(fix_constant) / rotation**2
                    )
                else:
                    num_channels[l] = int(
                        num_channels[l]
                        * np.sqrt(gspace.fibergroup.order() * fix_constant)
                        / gspace.fibergroup.order()
                    )
                if num_channels[l] < 1:
                    warnings.warn(
                        f"Group order ({gspace.fibergroup.order()*2}) is larger"
                        f" than number of channels ({num_channels[l]}) defined in layout!"
                    )
                    num_channels[l] = 1
            if restrict == "halved":
                num_channels[2] = int(
                    (layout[2] * np.sqrt(gspace.fibergroup.order() / 2 * fix_constant))
                    / (gspace.fibergroup.order() / 2)
                )
                if num_channels[2] < 1:
                    warnings.warn(
                        f"Group order ({gspace.fibergroup.order()*2/2}) is larger"
                        f" than number of channels ({layout[2]}) defined in layout!"
                    )
                    num_channels[2] = 1
            elif restrict == "reflection":
                num_channels[2] = int(layout[2] * np.sqrt(2 * fix_constant) / 2)
            elif restrict == "invariant":
                num_channels[2] = layout[2]

        # Color channels are trivial fields and don't transform when input is rotated/flipped
        self.input_field_type = FieldType(
            gspace, [gspace.trivial_repr] * input_channels
        )

        if group == "orthogonal":
            # Mask input image since the corners are moved outside the grid under rotations
            self.mask = MaskModule(self.input_field_type, S=spatial_dims, margin=1)
        else:
            self.mask = None

        # "Lifting" conv from trivial to regular feature fields
        self.conv1 = EquivariantConvBlock(
            in_type=self.input_field_type,
            out_channels=num_channels[0],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[0],
        )

        self.conv2 = EquivariantConvBlock(
            in_type=self.conv1.out_type,
            out_channels=num_channels[1],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[1],
            pool_size=2,
        )

        res1 = [
            EquivariantConvBlock(
                in_type=self.conv2.out_type,
                out_channels=num_channels[1],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[1],
            )
        ]
        res1 += [
            EquivariantConvBlock(
                in_type=res1[-1].out_type,
                out_channels=num_channels[1],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[1],
            )
        ]
        self.res1 = SequentialModule(*res1)

        self.scale_norm1 = EquivariantNorm(
            self.res1.out_type, num_groups=num_groups[1], affine=False
        )

        self.conv3 = EquivariantConvBlock(
            in_type=self.scale_norm1.out_type,
            out_channels=num_channels[2],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[2],
            pool_size=2,
        )

        # Restrict last conv and res layers
        self.restrict = Restriction(self.conv3.out_type, group, rotation, restrict)

        self.conv4 = EquivariantConvBlock(
            in_type=self.restrict.out_type,
            out_channels=num_channels[2],
            frequency=rotation,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups[2],
            pool_size=2,
        )

        res2 = [
            EquivariantConvBlock(
                in_type=self.conv4.out_type,
                out_channels=num_channels[2],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[2],
            )
        ]
        res2 += [
            EquivariantConvBlock(
                in_type=res2[-1].out_type,
                out_channels=num_channels[2],
                frequency=rotation,
                kernel_size=kernel_size,
                padding=padding,
                num_groups=num_groups[2],
            )
        ]
        self.res2 = SequentialModule(*res2)

        self.scale_norm2 = EquivariantNorm(
            self.res2.out_type, num_groups=num_groups[2], affine=False
        )

        self.invariant_map = EquivariantPool(
            self.scale_norm2.out_type, invariant_map=True
        )
        # self.global_pool = Reduce("N C (H 2) (W 2) -> N C H W", "mean")
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            self.invariant_map.out_type.size * 2 * 2, num_classes
        )

    def forward(self, x, embed=False, dropout=0.0):
        if dropout>0:
            raise NotImplementedError("Dropout not implemented yet")
        # Wrap input tensor in a GroupTensor
        x = GroupTensor(x, self.input_field_type)
        if self.mask is not None:  # SO(2)
            x = self.mask(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.scale_norm1(x)
        x = self.conv3(x)
        x = self.restrict(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.scale_norm2(x)
        x = self.invariant_map(x)
        x = x.tensor  # extract tensor from GroupTensor before common Pytorch ops
        x = self.global_pool(x)
        x = self.flatten(x)
        if not embed:
            x = self.classifier(x)
            
        return x


if __name__ == "__main__":
    inp = torch.rand(16, 3, 32, 32)
    model = EquivariantResNet9()
    out = model(inp)
    print(out.shape)
