from collections import defaultdict
from typing import List, Tuple, Union, Any

import torch
import numpy as np

from ....equivariant.nn import GSpace, FieldType, GroupTensor
from .equivariant_module import EquivariantModule

__all__ = ["MultipleModule"]


class MultipleModule(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        labels: List[str],
        modules: List[Tuple[EquivariantModule, Union[str, List[str]]]],
    ):
        r"""

        Split the input tensor in multiple branches identified by the input ``labels`` and
        apply to each of them the corresponding module in ``modules``.

        A label is associated to each field in the input type, while ``modules`` assigns a
        module to apply to each label (or set of labels).
        ``modules`` should be a list of pairs, each containing an :class:`~nn.EquivariantModule`
        and a label (or a list of labels).

        During forward, fields are grouped by the labels and the input tensor is split accordingly.
        Then, each subtensor is passed to the corresponding module in ``modules``.

        If ``reshuffle`` is set to a positive integer, a copy of the input tensor is first built
        sorting the fields according to the value set:

        - 1: fields are sorted by their labels
        - 2: fields are sorted by their labels and, then, by their size
        - 3: fields are sorted by their labels, by their size and, then, by their type

        In this way, fields that need to be retrieved together are contiguous and it is possible
        to exploit slicing to split the tensor.
        By default, ``reshuffle = 0`` which means that no sorting is performed.

        This modules wraps a :class:`~nn.BranchingModule` followed by a :class:`~nn.MergeModule`.

        Args:
            in_type (FieldType): the input field type
            labels (list): the list of labels to group the fields
            modules (list): list of modules to apply to the labeled fields

        """

        assert isinstance(in_type.gspace, GSpace)

        super(MultipleModule, self).__init__()

        self.gspace = in_type.gspace
        self.in_type = in_type
        self._labels = set(labels)
        # For each label, build the representation of the sub-fiber on which it acts
        self._label_out_types = in_type.group_by_labels(labels)

        out_repr = []
        modules_labels = []
        for module, l in modules:
            # Check types of modules
            if isinstance(module.in_type, tuple):
                assert all(t.gspace == self.gspace for t in module.in_type)
            else:
                assert module.in_type.gspace == self.gspace
            out_repr += module.out_type.representations
            # Check if given labels comply with modules labels
            if isinstance(l, list):
                modules_labels += l
                l = l[0]
            else:
                modules_labels.append(l)
            # Check if module type and label type match
            assert module.in_type == self._label_out_types[l], (
                f"Label {l}, branch class and module ({module}) class don't match:\n "
                f"[{module.in_type}] \n [{ self._label_out_types[l]}]\n"
            )
            # Add modules as submodules
            self.add_module(f"submodule_{l}", module)

        self.out_type = FieldType(self.gspace, out_repr)

        assert (set(modules_labels) in self._labels) or (
            set(modules_labels) == self._labels
        ), "Error! Some labels assigned to the modules don't appear among the channels labels"

        assert len(in_type.representations) == len(
            labels
        ), "Error! Number of labels ({}) does not match number of fields ({})".format(
            len(labels), len(in_type.representations)
        )

        for i in range(len(modules)):
            if isinstance(modules[i][1], str):
                modules[i] = (modules[i][0], [modules[i][1]])
            else:
                assert isinstance(modules[i][1], list)
                for s in modules[i][1]:
                    assert isinstance(s, str)

        # For each label, compute:
        #   - the set of indices on the fiber of its fields and
        #   - the the indices of the fields belonging to it
        last_position = 0
        _input_indices = defaultdict(list)
        for c, l in enumerate(labels):
            # Append indices of the current field
            _input_indices[l] += list(
                range(last_position, last_position + in_type.representations[c].size)
            )
            # Move on to next fiber
            last_position += in_type.representations[c].size

        self.indices = {}
        for l in self._labels:
            # Only the first and the last indices are preserved
            _input_indices[l] = torch.LongTensor(
                [min(_input_indices[l]), max(_input_indices[l]) + 1]
            )

            # register the indices tensors as parameters of this module
            self.indices[l] = _input_indices[l]

    def _retrieve_subfiber(self, input: GroupTensor, l: str) -> GroupTensor:
        r"""

        Return a new GroupTensor containg the portion of memory of the input tensor corresponding to the fields
        the input non-linearity acts on. The method automatically deals with the continuity of these fields, using
        either indexing or slicing.

        The resulting tensor is returned wrapped in a GroupTensor with the proper representation

        Args:
            input (GroupTensor): the input tensor
            l (str): the label to consider

        Returns:
            (GroupTensor): the sub-tensor containing the fields belonging to the input label

        """
        indices = self.indices[l]
        data = input.tensor[:, indices[0] : indices[1], ...].clone()

        # wrap the result in a GroupTensor
        return GroupTensor(data, self._label_out_types[l], input.coords)

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Split the input tensor according to the labels, apply each module to the corresponding
        input sub-tensors and stack the results.

        Args:
            input (GroupTensor): the input GroupTensor

        Returns:
            the concatenation of the output of each module

        """

        assert input.type == self.in_type

        output = None

        # Create tensor for each label
        for l in self._labels:
            subfiber = self._retrieve_subfiber(input, l)
            module = getattr(self, f"submodule_{l}")
            out = module(subfiber)

            if output is None:
                output = out.tensor
            else:
                # assert out.has_same_coords(output) - TODO: Find a way to do this clean
                output = torch.cat([output, out.tensor], axis=1)

        return GroupTensor(output, self.out_type, out.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) > 1
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        input_shapes = {l: (b, repr.size, hi, wi) for l, repr in self.out_type.items()}

        output_shapes = []

        # iterate through the modules
        for i, labels in enumerate(self._labels):
            module = getattr(self, f"submodule_{i}")
            # evaluate the corresponding output shape
            os = module.evaluate_output_shape(*[input_shapes[l] for l in labels])
            output_shapes.append(list(os))

        out_shape = list(output_shapes[0])

        for os in output_shapes[1:]:
            assert out_shape[0] == os[0]
            assert out_shape[2:] == os[2:]

            out_shape[1] += os[1]

        return out_shape

    def check_equivariance(
        self,
        x: torch.Tensor = None,
        atol: float = 2e-6,
        rtol: float = 1e-5,
        full_space_action: bool = True,
    ) -> List[Tuple[Any, float]]:
        if full_space_action:
            return super(MultipleModule, self).check_equivariance(
                x=x, atol=atol, rtol=rtol
            )

        else:
            if x is None:
                c = self.in_type.size
                x = torch.randn(10, c, 9, 9)
                print(c, self.out_type.size)
                print([r.name for r in self.in_type.representations])
                print([r.name for r in self.out_type.representations])
                x = GroupTensor(x, self.in_type)

            errors = []

            for el in self.gspace.testing_elements:
                out1 = self(x).transform_fibers(el)
                out2 = self(x.transform_fibers(el))

                errs = (out1.tensor - out2.tensor).detach().numpy()
                errs = np.abs(errs).reshape(-1)
                print(el, errs.max(), errs.mean(), errs.var())

                if not torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol):
                    tmp = np.abs((out1.tensor - out2.tensor).detach().numpy())
                    tmp = tmp.reshape(
                        out1.tensor.shape[0], out1.tensor.shape[1], -1
                    ).max(axis=2)

                    np.set_printoptions(
                        precision=2, threshold=200000000, suppress=True, linewidth=500
                    )
                    print(tmp.shape)
                    print(tmp)

                assert torch.allclose(
                    out1.tensor, out2.tensor, atol=atol, rtol=rtol
                ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                    el, errs.max(), errs.mean(), errs.var()
                )

                errors.append((el, errs.mean()))

            return errors
