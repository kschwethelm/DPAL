from collections import defaultdict

from .....equivariant.nn import GSpace, FieldType, GroupTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from typing import List, Tuple, Any


__all__ = ["FieldDropout"]


def dropout_field(input: torch.Tensor, p: float, training: bool, inplace: bool):
    if training:
        shape = list(input.size())
        shape[2] = 1

        if input.device == torch.device("cpu"):
            mask = torch.FloatTensor(*shape)
        else:
            device = input.device
            mask = torch.cuda.FloatTensor(*shape, device=device)

        mask = mask.uniform_() > p
        mask = mask.to(torch.float)

        if inplace:
            input *= mask / (1.0 - p)
            return input
        else:
            return input * mask / (1.0 - p)
    else:
        return input


class FieldDropout(EquivariantModule):
    def __init__(self, in_type: FieldType, p: float = 0.5, inplace: bool = False):
        r"""

        Applies dropout to individual *fields* independently.

        Notice that, with respect to :class:`~nn.PointwiseDropout`, this module acts on a whole field instead
        of single channels.

        Args:
            in_type (FieldType): the input field type
            p (float, optional): dropout probability
            inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

        """

        assert isinstance(in_type.gspace, GSpace)
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )

        super(FieldDropout, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        self.p = p
        self.inplace = inplace

        # number of fields of each size
        self._nfields = defaultdict(int)

        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(list)
        position = 0
        for r in self.in_type.representations:
            _indices[r.size] += list(range(position, position + r.size))
            self._nfields[r.size] += 1
            position += r.size

        self.indices = {}
        for s in self.in_type.representations:
            _indices[s.size] = torch.LongTensor(
                [min(_indices[s.size]), max(_indices[s.size]) + 1]
            )

            # register the indices tensors as parameters of this module
            self.indices[s.size] = _indices[s.size].to(
                f"cuda:{torch.cuda.current_device()}"
            )

        self._order = list(self.indices.keys())

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        if not self.training:
            return input

        coords = input.coords
        input = input.tensor

        if not self.inplace:
            output = torch.empty_like(input)

        # iterate through all field sizes
        for s in self._order:
            indices = self.indices[s]
            shape = input.shape[:1] + (self._nfields[s], s) + input.shape[2:]

            out = dropout_field(
                input[:, indices[0] : indices[1], ...].view(shape),
                self.p,
                self.training,
                self.inplace,
            )
            if not self.inplace:
                shape = input.shape[:1] + (self._nfields[s] * s,) + input.shape[2:]
                output[:, indices[0] : indices[1], ...] = out.view(shape)

        if self.inplace:
            output = input

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) > 1
        assert input_shape[1] == self.in_type.size

        return input_shape

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        pass
