from ....equivariant.nn import GSpace, FieldType, GroupTensor

import torch
import numpy as np

from .equivariant_module import EquivariantModule
from .utils import indexes_from_labels
from ....equivariant.nn import *
from ....equivariant.group_theory import disentangle

from collections import defaultdict

from typing import List, Tuple, Any

__all__ = ["DisentangleModule"]


class DisentangleModule(EquivariantModule):
    def __init__(self, in_type: FieldType):
        r"""

        Disentangles the representations in the field type of the input.

        This module only acts as a wrapper for :func:`group.disentangle`.
        In the constructor, it disentangles each representation in the input type to build the output type and
        pre-compute the change of basis matrices needed to transform each input field.

        During the forward pass, each field in the input tensor is transformed with the change of basis corresponding
        to its representation.

        Args:
            in_type (FieldType): the input field type

        """
        assert isinstance(in_type, FieldType)
        assert isinstance(in_type.gspace, GSpace)

        super(EquivariantModule, self).__init__()

        self.in_type = in_type

        disentangled_representations = {}

        _change_of_basis_matrices = {}
        self._sizes = {}

        self.change_of_basis = {}
        for r in self.in_type._unique_representations:
            self._sizes[r.name] = r.size
            cob, reprs = disentangle(r)
            disentangled_representations[r.name] = reprs
            _change_of_basis_matrices[r.name] = torch.FloatTensor(cob).cuda()
            self.change_of_basis[r.name] = _change_of_basis_matrices[r.name]

        out_reprs = []
        self._nfields = defaultdict(int)
        for r in self.in_type.representations:
            self._nfields[r.name] += 1
            out_reprs += disentangled_representations[r.name]

        self.out_type = FieldType(self.in_type.gspace, out_reprs)

        grouped_indices = indexes_from_labels(
            self.in_type, [r.name for r in self.in_type.representations]
        )

        self._order = []
        self.fiber_indices = {}
        for repr_name, (
            fields_indices,
            fiber_indices,
        ) in grouped_indices.items():
            self._order.append(repr_name)
            fiber_indices = torch.LongTensor(
                (min(fiber_indices), max(fiber_indices) + 1)
            )
            self.fiber_indices[repr_name] = fiber_indices.cuda()

    def forward(self, input: GroupTensor) -> GroupTensor:
        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor

        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        output = torch.empty_like(input)

        # for each different representation in the fiber
        for repr_name in self._order:
            fiber_indices = self.fiber_indices[repr_name]

            # retrieve the associated change of basis
            cob = self.change_of_basis[repr_name]

            # retrieve the associated fields from the input tensor
            input_fields = input[:, fiber_indices[0] : fiber_indices[1], ...]

            # reshape to align all the fields in order to exploit broadcasting
            input_fields = input_fields.view(
                b, self._nfields[repr_name], self._sizes[repr_name], *spatial_shape
            )

            # TODO: can we exploit the fact the change of basis is a permutation matrix?
            # transform all the fields with the change of basis
            transformed_fields = torch.einsum(
                "oi,bci...->bco...", (cob, input_fields)
            ).reshape(b, -1, *spatial_shape)

            # insert the transformed fields in the output tensor
            output[:, fiber_indices[0] : fiber_indices[1], ...] = transformed_fields

        return GroupTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def check_equivariance(
        self, atol: float = 1e-7, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        c = self.in_type.size

        x = torch.randn(3, c, 10, 10)

        x = GroupTensor(x, self.in_type)

        errors = []

        for el in self.out_type.testing_elements:
            print(el)

            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el)).tensor.detach().numpy()

            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())

            assert np.allclose(
                out1, out2, atol=atol, rtol=rtol
            ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                el, errs.max(), errs.mean(), errs.var()
            )

            errors.append((el, errs.mean()))

        return errors
