from typing import List, Tuple, Any

import numpy as np

from collections import defaultdict

from .....equivariant.nn import GSpace, FieldType, GroupTensor

from ..equivariant_module import EquivariantModule
from .gated1 import GATED_ID, GATES_ID

import torch

from torch.nn import Parameter


__all__ = ["InducedGatedNonLinearity"]


class InducedGatedNonLinearity(EquivariantModule):
    def __init__(
        self, in_type: FieldType, gates: List = None, drop_gates: bool = True, **kwargs
    ):
        r"""

        Induced Gated non-linearities.

        .. todo::
            complete documentation!

        .. note::
            Make sure all induced gate and gates have same subgroup


        Args:
            in_type (FieldType): the input field type
            gates (list, optional): list of strings specifying which field in input is a gate and which is a gated field
            drop_gates (bool, optional): if ``True`` (default), drop the trivial fields after using them to compute
                    the gates. If ``False``, the gates are stacked with the gated fields in the output

        """

        assert isinstance(in_type.gspace, GSpace)

        if gates is None:
            assert len(in_type) % 2 == 0

            g = len(in_type) // 2
            gates = [GATES_ID] * g + [GATED_ID] * g

        assert len(gates) == len(in_type)

        super(InducedGatedNonLinearity, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type

        self.drop_gates = drop_gates
        _input_indices = defaultdict(list)
        _output_indices = defaultdict(list)

        self._nfields = defaultdict(int)

        self.branching = None

        for g, r in zip(gates, in_type.representations):
            if g == GATES_ID:
                assert (
                    "induced_gate" in r.supported_nonlinearities
                ), 'Error! Representation "{}" can\'t be a "gate"'.format(r.name)
            elif g == GATED_ID:
                for nl in r.supported_nonlinearities:
                    if nl.startswith("induced_gated"):
                        break
                else:
                    raise ValueError(
                        'Error! Representation "{}" does not support "gated" non-linearity'.format(
                            r.name
                        )
                    )
            else:
                raise ValueError('Error! "{}" type not recognized'.format(g))

        ngates = len([g for g in gates if g == GATES_ID])
        ngated = len([g for g in gates if g == GATED_ID])

        assert (
            ngates == ngated
        ), "Error! Number of gates ({}) does not match the number of gated non-linearities required ({})".format(
            ngates, ngated
        )

        quotient_size = None
        for g, r in zip(gates, in_type):
            if g == GATES_ID:
                if quotient_size is None:
                    quotient_size = r.size
                else:
                    assert r.size == quotient_size

        subfield_sizes = {}
        for g, r in zip(gates, in_type):
            if g == GATED_ID:
                subfield_size = None
                for nl in r.supported_nonlinearities:
                    if nl.startswith("induced_gated"):
                        assert subfield_size is None, (
                            "Error! The representation supports multiple "
                            "sub-fields of different sizes"
                        )
                        subfield_size = int(nl.split("_")[-1])
                        assert r.size % subfield_size == 0
                        assert r.size // subfield_size == quotient_size
                        subfield_sizes[r.name] = subfield_size

        self.quotient_size = quotient_size

        if self.drop_gates:
            # only gated fields are preserved
            # therefore, the output representation is computed from the input one, removing the gates
            self.out_type = in_type.index_select(
                [i for i, g in enumerate(gates) if g == GATED_ID]
            )
        else:
            self.out_type = in_type

        in_last_position = 0
        out_last_position = 0
        last_type = None

        # group fields by type (gated or gate) and their size and retrieve the indices of the fields
        self._types = []
        for g, r in zip(gates, in_type.representations):
            if g == GATES_ID:
                type = g
            else:
                type = r.size, subfield_sizes[r.name]
                self._nfields[type] += 1

            if type != last_type:
                if not type in self._types:
                    self._types.append(type)
            last_type = type

            _input_indices[type] += list(
                range(in_last_position, in_last_position + r.size)
            )
            in_last_position += r.size

            if g != GATES_ID or not self.drop_gates:
                # since gates are discarded in output, the position on the output fiber is shifted
                # only when a gated field is met
                _output_indices[type] += list(
                    range(out_last_position, out_last_position + r.size)
                )
                out_last_position += r.size

        _input_indices = dict(_input_indices)
        # if self.drop_gates:
        _output_indices = dict(_output_indices)
        # else:
        #     self._output_indices = self._input_indices

        self.input_indices = {}
        self.output_indices = {}
        for t in self._types:
            _input_indices[t] = torch.LongTensor(
                [min(_input_indices[t]), max(_input_indices[t]) + 1]
            )
            if t != GATES_ID or not self.drop_gates:
                _output_indices[t] = torch.LongTensor(
                    [min(_output_indices[t]), max(_output_indices[t]) + 1]
                )

            # register the indices tensors as parameters of this module
            self.input_indices[t] = _input_indices[t].to(
                f"cuda:{torch.cuda.current_device()}"
            )
            if t != GATES_ID or not self.drop_gates:
                self.output_indices[t] = _output_indices[t].to(
                    f"cuda:{torch.cuda.current_device()}"
                )

        # gates need to be distinguished from gated fields
        _gates_indices = _input_indices.pop(GATES_ID)
        self.gates_indices = _gates_indices

        # build a sorted list of the fields groups, such that every time they are iterated through in the same order
        self._order = sorted(_input_indices.keys())

        # the bias for the gates
        self.bias = Parameter(
            torch.randn(1, ngates, 1, dtype=torch.float), requires_grad=True
        )

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Apply the gated non-linearity to the input feature map.

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert isinstance(input, GroupTensor)
        assert input.type == self.in_type

        # Retrieve the gates
        gates = input.tensor[:, self.gates_indices[0] : self.gates_indices[1], ...]
        coords = input.coords

        # retrieving only gated fields from the joint tensor is worthless
        input = input.tensor

        b, c = input.shape[:2]
        spatial_dims = input.shape[2:]

        # transform the gates
        gates = torch.sigmoid(
            gates.view(b, -1, self.quotient_size, *spatial_dims)
            - self.bias.view(1, -1, 1, *[1] * len(spatial_dims))
        )

        output = None
        next_gate = 0

        # for each field size
        for type in self._order:
            size, subfield_size = type

            # retrieve the needed gates
            g = gates[:, next_gate : next_gate + self._nfields[type], ...].view(
                b, -1, 1, *spatial_dims
            )

            input_indices = self.input_indices[type]
            output_indices = self.output_indices[type]

            output = torch.empty(
                [b, (output_indices[1] - output_indices[0]), *spatial_dims]
            )

            output = (
                input[:, input_indices[0] : input_indices[1], ...].view(
                    b, -1, subfield_size, *spatial_dims
                )
                * g
            ).view(b, -1, *spatial_dims)

            # shift the position on the gates fiber
            next_gate += self._nfields[type]

        if not self.drop_gates:
            # copy the gates in the output
            output = torch.cat([output, gates], dim=1)

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, coords)

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
