from collections import defaultdict

from torch.nn import Parameter

from .....equivariant.nn import GSpace, FieldType, GroupTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any, Union

import math


__all__ = ["NormMaxPool"]


class NormMaxPool(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        ceil_mode: bool = False,
    ):
        r"""
        Max-pooling based on the fields' norms. In a given window of shape :attr:`kernel_size`, for each
        group of channels belonging to the same field, the field with the highest norm (as the length of the vector)
        is preserved.
        Except :attr:`in_type`, the other parameters correspond to the ones of :class:`torch.nn.MaxPool2d`.
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
        Args:
            in_type (FieldType): the input field type
            kernel_size: the size of the window to take a max over
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            dilation: a parameter that controls the stride of elements in the window
            ceil_mode: when ``True``, will use ceil instead of floor to compute the output shape
        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 2

        super(NormMaxPool, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

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

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.ceil_mode = ceil_mode

        # Group fields by their size and retrieve the indices of the fields
        # number of fields of each size
        self._nfields = defaultdict(int)

        # indices of the channales corresponding to fields belonging to each group
        _indices = defaultdict(list)

        position = 0
        for i, r in enumerate(self.in_type.representations):
            _indices[r.size] += list(range(position, position + r.size))
            self._nfields[r.size] += 1
            position += r.size

        self.indices = {}
        for s in list(self._nfields.keys()):
            _indices[s] = torch.LongTensor([min(_indices[s]), max(_indices[s]) + 1])

            # register the indices tensors as parameters of this module
            self.indices[s] = _indices[s].to(f"cuda:{torch.cuda.current_device()}")

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""
        Run the norm-based max-pooling on the input tensor
        Args:
            input (GroupTensor): the input feature map
        Returns:
            the resulting feature map
        """

        assert input.type == self.in_type

        b, c, hi, wi = input.tensor.shape

        # compute the output shape (see 'torch.nn.MaxPool2D')
        b, c, ho, wo = self.evaluate_output_shape(input.tensor.shape)

        # compute the squares of the values of each channel
        # n = torch.mul(input.data, input.data)
        n = input.tensor**2

        # pre-allocate the output tensor
        output = None

        # reshape the input to merge the spatial dimensions
        input = input.tensor.reshape(b, c, -1)

        # iterate through all field sizes
        for s in list(self._nfields.keys()):
            indices = self.indices[s]
            # compute the norms
            norms = (
                n[:, indices[0] : indices[1], :, :]
                .view(b, -1, s, hi, wi)
                .sum(dim=2)
                .sqrt()
            )

            # run max-pooling on the norms-tensor
            _, indx = F.max_pool2d(
                norms,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                return_indices=True,
            )

            # in order to use the pooling indices computed for the norms to retrieve the fields, they need to be
            # expanded in the inner field dimension
            indx = indx.view(b, -1, 1, ho * wo).expand(-1, -1, s, -1)

            out = torch.empty(b, (indices[1] - indices[0]), ho, wo, device=input.device)

            # retrieve the fields from the input tensor using the pooling indeces
            out = (
                input[:, indices[0] : indices[1], :]
                .view(b, -1, s, hi * wi)
                .gather(3, indx)
                .view(b, -1, ho, wo)
            )
            if output is None:
                output = out
            else:
                output = torch.cat([output, out], axis=1)

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(
        self, input_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        # compute the output shape (see 'torch.nn.MaxPool2D')
        ho = math.floor(
            (
                hi
                + 2 * self.padding[0]
                - self.dilation[0] * (self.kernel_size[0] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )
        wo = math.floor(
            (
                wi
                + 2 * self.padding[1]
                - self.dilation[1] * (self.kernel_size[1] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )

        return b, self.out_type.size, ho, wo

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        # this kind of pooling is not really equivariant so we can not test equivariance
        pass
