from .....equivariant.nn import GSpace, FieldType, GroupTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any, Union

import math


__all__ = ["NormAvgPool"]


class NormAvgPool(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
    ):
        r"""
        Avg-pooling based on the fields' norms. In a given window of shape :attr:`kernel_size`,
        for each group of channels belonging to the same field, the average is calculated and
        used to do a norm-based weighted average pooling on the input.
        Except :attr:`in_type`, the other parameters correspond to the ones of :class:`torch.nn.AvgPool2d`.
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a avg over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape
        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 2

        super(NormAvgPool, self).__init__()

        reps_sizes = [r.size for r in in_type.representations]
        assert reps_sizes.count(reps_sizes[0]) == len(
            reps_sizes
        ), f"Representations need to be the same size but are {set(reps_sizes)}."

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        self.rep_size = reps_sizes[0]

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.ceil_mode = ceil_mode

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""
        Run the norm-based avg-pooling on the input tensor
        Args:
            input (GroupTensor): the input feature map
        Returns:
            the resulting feature map
        """

        assert input.type == self.in_type

        b, c, hi, wi = input.tensor.shape

        # Compute norms
        n = input.tensor**2
        norms = n.view(b, -1, self.rep_size, hi, wi).sum(dim=2).sqrt()

        # Run avg-pooling on the norms-tensor and expand back to norms size
        avg_norms = F.avg_pool2d(
            norms,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=True,
            count_include_pad=False,
        ) * (self.kernel_size[0] * self.kernel_size[1])
        avg_norms_expanded = torch.repeat_interleave(
            torch.repeat_interleave(avg_norms, self.kernel_size[0], dim=2),
            self.kernel_size[1],
            dim=3,
        )

        # Fix expanded size if input size is odd
        if avg_norms_expanded.shape != norms.shape:
            avg_norms_expanded = avg_norms_expanded[
                :, :, : norms.shape[2], : norms.shape[3]
            ]

        # Divide avg by original norms for weighted avg and expand to input size
        weighted_avg_norms = torch.divide(avg_norms_expanded, norms).repeat(
            (1, self.rep_size, 1, 1)
        )

        # Multiply input by weighted avg norms before avg pooling
        output = F.avg_pool2d(
            input.tensor * weighted_avg_norms,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
        )

        # wrap the result in a GroupTensor
        return GroupTensor(output.contiguous(), self.out_type, coords=None)

    def evaluate_output_shape(
        self, input_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        # compute the output shape (see 'torch.nn.AvgPool2D')
        ho = (hi + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[
            0
        ] + 1
        wo = (wi + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[
            1
        ] + 1

        if self.ceil_mode:
            ho = math.ceil(ho)
            wo = math.ceil(wo)
        else:
            ho = math.floor(ho)
            wo = math.floor(wo)

        return b, self.out_type.size, ho, wo

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass
