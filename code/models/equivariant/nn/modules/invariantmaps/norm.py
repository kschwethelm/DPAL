from .....equivariant.nn import GSpace, FieldType, GroupTensor

from .....equivariant.nn.modules.equivariant_module import EquivariantModule
from .....equivariant.nn.modules.utils import indexes_from_labels

import torch

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np

__all__ = ["NormPool"]


class NormPool(EquivariantModule):
    def __init__(self, in_type: FieldType, **kwargs):
        r"""

        Module that implements Norm Pooling.
        For each input field, an output one is built by taking the norm of that field; as a result, the output
        field transforms according to a trivial representation.

        Args:
            in_type (FieldType): the input field type

        """
        assert isinstance(in_type.gspace, GSpace)

        super(NormPool, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type

        # build the output representation substituting each input field with a trivial representation
        self.out_type = FieldType(self.space, [self.space.trivial_repr] * len(in_type))

        # indices of the channels corresponding to fields belonging to each group in the input representation
        _in_indices = defaultdict(list)
        # indices of the channels corresponding to fields belonging to each group in the output representation
        _out_indices = defaultdict(list)

        # group fields by their size and retrieve the indices of the fields
        indeces = indexes_from_labels(
            in_type, [r.size for r in in_type.representations]
        )

        self.in_indices = {}
        self.out_indices = {}
        for s, (fields, idxs) in indeces.items():
            _in_indices[s] = torch.LongTensor([min(idxs), max(idxs) + 1])
            _out_indices[s] = torch.LongTensor([min(fields), max(fields) + 1])

            # register the indices tensors as parameters of this module
            self.in_indices[s] = _in_indices[s].to(
                f"cuda:{torch.cuda.current_device()}"
            )
            self.out_indices[s] = _out_indices[s].to(
                f"cuda:{torch.cuda.current_device()}"
            )

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Apply the Norm Pooling to the input feature map.

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        output = torch.empty(
            self.evaluate_output_shape(input.shape),
            device=input.device,
            dtype=torch.float,
        )

        for s in list(self.in_indices.keys()):
            in_indices = self.in_indices[s]
            out_indices = self.out_indices[s]

            fm = input[:, in_indices[0] : in_indices[1], ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)

        # wrap the result in a GroupTensor
        return GroupTensor(fm.norm(dim=2), self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) > 1
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        c = self.in_type.size

        x = torch.randn(3, c, *[10] * self.in_type.gspace.dimensionality)

        x = GroupTensor(x, self.in_type)

        errors = []

        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))

            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())

            assert torch.allclose(
                out1.tensor, out2.tensor, atol=atol, rtol=rtol
            ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                el, errs.max(), errs.mean(), errs.var()
            )

            errors.append((el, errs.mean()))

        return errors
