from .....equivariant.nn import GSpace, FieldType, GroupTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any, Union

__all__ = ["PointwiseAdaptiveAvgPool"]


class PointwiseAdaptiveAvgPool(EquivariantModule):
    def __init__(self, in_type: FieldType, output_size: Union[int, Tuple[int, int]]):
        r"""

        Adaptive channel-wise average-pooling: each channel is treated independently.
        This module works exactly as :class:`torch.nn.AdaptiveAvgPool2D`, wrapping it in
        the :class:`~nn.EquivariantModule` interface.

        Notice that not all representations support this kind of pooling. In general, only representations which support
        pointwise non-linearities do.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the input field type
            output_size: the target output size of the image of the form H x W

        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 2

        for r in in_type.representations:
            assert (
                "pointwise" in r.supported_nonlinearities
            ), f"""Error! Representation "{r.name}" does not support pointwise non-linearities
                so it is not possible to pool each channel independently"""

        super(PointwiseAdaptiveAvgPool, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        # run the common avg-pooling
        output = F.adaptive_avg_pool2d(input.tensor, self.output_size)

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(
        self, input_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        return b, self.out_type.size, self.output_size, self.output_size

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.AdaptiveAvgPool2d` module and set to "eval" mode.

        """

        self.eval()

        return torch.nn.AdaptiveAvgPool2d(self.output_size).eval()
