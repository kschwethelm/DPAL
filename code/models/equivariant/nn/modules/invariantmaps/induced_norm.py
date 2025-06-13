from .....equivariant.nn import GSpace, FieldType, GroupTensor

from .....equivariant.nn.modules.equivariant_module import EquivariantModule

import torch

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np

__all__ = ["InducedNormPool"]


class InducedNormPool(EquivariantModule):
    def __init__(self, in_type: FieldType, **kwargs):
        r"""

        Module that implements Induced Norm Pooling.
        This module requires the input fields to be associated to an induced representation from a representation
        which supports 'norm' non-linearities.

        First, for each input field, an output one is built by taking the maximum norm of all its sub-fields.

        Args:
            in_type (FieldType): the input field type

        """
        assert isinstance(in_type.gspace, GSpace)

        super(InducedNormPool, self).__init__()

        for r in in_type.representations:
            assert any(
                nl.startswith("induced_norm") for nl in r.supported_nonlinearities
            ), 'Error! Representation "{}" does not support "induced_norm" non-linearity'.format(
                r.name
            )

        self.space = in_type.gspace
        self.in_type = in_type

        # build the output representation substituting each input field with a trivial representation
        self.out_type = FieldType(self.space, [self.space.trivial_repr] * len(in_type))

        # Group fields by their size and the size of the subfields and retrieve the indices of the fields
        # indices of the channels corresponding to fields belonging to each group in the input representation
        _in_indices = defaultdict(list)
        # indices of the channels corresponding to fields belonging to each group in the output representation
        _out_indices = defaultdict(list)

        # number of fields of each size
        self._nfields = defaultdict(int)

        position = 0
        last_id = None
        self.ids = []
        for i, r in enumerate(self.in_type.representations):
            subfield_size = None
            for nl in r.supported_nonlinearities:
                if nl.startswith("induced_norm"):
                    assert subfield_size is None, (
                        "Error! The representation supports multiple "
                        "sub-fields of different sizes"
                    )
                    subfield_size = int(nl.split("_")[-1])
                    assert r.size % subfield_size == 0

            id = (r.size, subfield_size)

            if id != last_id:
                self.ids.append(id)

            last_id = id

            _in_indices[id] += list(range(position, position + r.size))
            _out_indices[id] += [i]
            self._nfields[id] += 1
            position += r.size

        self.in_indices = {}
        self.out_indices = {}
        for id in self.ids:
            _in_indices[id] = torch.LongTensor(
                [min(_in_indices[id]), max(_in_indices[id]) + 1]
            )
            _out_indices[id] = torch.LongTensor(
                [min(_out_indices[id]), max(_out_indices[id]) + 1]
            )

            # register the indices tensors as parameters of this module
            self.in_indices[id] = _in_indices[id].to(
                f"cuda:{torch.cuda.current_device()}"
            )
            self.out_indices[id] = _out_indices[id].to(
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

        for id in self.ids:
            size, subfield_size = id
            n_subfields = size // subfield_size

            in_indices = self.in_indices[id]

            fm = input[:, in_indices[0] : in_indices[1], ...]

            # split the channel dimension in 2 dimensions, separating fields
            fm, _ = (
                fm.view(b, -1, n_subfields, subfield_size, *spatial_shape)
                .norm(dim=3)
                .max(dim=2)
            )

        # wrap the result in a GroupTensor
        return GroupTensor(fm, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        c = self.in_type.size

        x = torch.randn(3, c, 10, 10)

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
