import numpy as np
from collections import defaultdict
from .....equivariant.nn import FieldType, GroupTensor

from .....equivariant.nn import *
from .....equivariant.nn.modules import EquivariantModule

import torch
from typing import List, Tuple, Any
from einops import rearrange, repeat

__all__ = ["GroupNorm", "InducedNormGroupNorm", "GroupStandardization"]

def closest_divisor(number, div):
    """
    Get next closest divisor to given number and divisor. Take second closest
    divisor if initial one results in number smaller than 3.

    We set the value 3 heuristically as group normalization with groups of less
    than 4 channels seem to perform worse.
    """
    all_divs = [i for i in range(1, number + 1) if number % i == 0]
    if len(all_divs) < 2:
        closest_div = all_divs[0]
    else:
        idx = np.argsort(np.abs(np.array(all_divs) - div))[:2]
        # Check next closest divisor if result is too small
        if int(number / all_divs[idx[0]]) > 3:
            return all_divs[idx[0]]
        closest_div = all_divs[idx[1]]
    # Set divisor to 1 if result is too small
    if int(number / closest_div) <= 3:
        return 1
    return closest_div


class GroupNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        num_groups: int,
        eps: float = 1e-05,
    ):
        r"""

        Group norm for generic representations of 2D data (i.e. 4D inputs) where channels
        and representations are in the same dimension.

        Works with representations supporting pointwise non-linearities,
        i.e. regular and trivial. Only channels are grouped, not representations.
        Representations need to have the same size.

        Args:
            in_type (FieldType): Input field type / representation.
            num_groups (int): Number of groups to divide the channesl into.
            eps (float, default: ``1e-5``): Added to the denominator for numerical stability.

        """

        assert isinstance(in_type.gspace, GSpace)

        super(GroupNorm, self).__init__()

        reps_sizes = [r.size for r in in_type.representations]
        assert reps_sizes.count(reps_sizes[0]) == len(
            reps_sizes
        ), f"Representations need to be the same size but are {set(reps_sizes)}."

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        self.eps = eps

        self.reps_size = reps_sizes[0]
        self.channels = len([r.size for r in in_type.representations])

        if self.channels % num_groups != 0:
            self.num_groups = closest_divisor(self.channels, num_groups)
            print(
                f"INFO: Number of channels is not divisible by number of groups"
                f" ({self.channels}/{num_groups}). Adapted number of groups to {self.num_groups}."
            )
        else:
            self.num_groups = num_groups

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        in_tensor = rearrange(
            input.tensor,
            "B (C F) H W -> B C F H W",
            C=self.channels,
        )

        output = torch.nn.functional.group_norm(
            input=in_tensor, num_groups=self.num_groups, eps=self.eps
        )

        # Merge channel and reps dimensions and wrap result in GroupTensor
        return GroupTensor(
            rearrange(output, "B C F H W -> B (C F) H W"),
            self.out_type,
            input.coords,
        )

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, x: torch.Tensor = None, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        return super(GroupNorm, self).check_equivariance(x=x, atol=atol, rtol=rtol)

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        main_str = self._get_name() + "("
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(extra_lines) + "\n"

        main_str += ")"
        return main_str

    def extra_repr(self):
        return "{in_type}, eps={eps}, affine=False".format(**self.__dict__)


class InducedNormGroupNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        num_groups: int,
        eps: float = 1e-05,
    ):
        r"""

        Group norm for generic representations of 2D data (i.e. 4D inputs).
        Channels and representations are in the same dimension.

        Works with representations supporting norm non-linearities, i.e. induced irreps.
        Calculates the norm of the representations and uses it to norm the input.

        Args:
            in_type (FieldType): Input field type / representation.
            num_groups (int): Number of groups to divide the channesl into.
            eps (float, default: ``1e-5``): Added to the denominator for numerical stability.

        """

        assert isinstance(in_type.gspace, GSpace)

        super(InducedNormGroupNorm, self).__init__()

        reps_sizes = [r.size for r in in_type.representations]
        assert reps_sizes.count(reps_sizes[0]) == len(
            reps_sizes
        ), f"Representations need to be the same size but are {set(reps_sizes)}."

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        self.eps = eps

        self.reps_size = reps_sizes[0]
        self.channels = len([r.size for r in in_type.representations])

        if self.channels % num_groups != 0:
            print(
                f"INFO: Number of channels is not divisible by number of groups"
                f" ({self.channels}/{num_groups}). Switching to LayerNorm."
            )
            self.num_groups = 1  # layer norm
        elif num_groups >= self.channels:
            print(
                f"INFO: Number of groups is bigger or equal to number of channels"
                f" ({self.channels}). Using InstanceNorm."
            )
            self.num_groups = self.channels  # instance norm
        else:
            self.num_groups = num_groups

        self._nfields = defaultdict(int)

        # indices of the channales corresponding to fields belonging to each group
        _indices = defaultdict(list)

        position = 0
        for r in self.in_type.representations:
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

            _indices[id] += list(range(position, position + r.size))
            self._nfields[id] += 1
            position += r.size

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        b, c, h, w = input.shape

        # Split channel dimensions into groups and representations
        in_tensor = rearrange(
            input.tensor,
            "B (G C R) H W -> B G C R H W",
            G=self.num_groups,
            R=self.reps_size,
        )

        # Compute norms
        norms = torch.square(in_tensor).sum(dim=3, keepdim=False).sqrt()

        # Since the mean of the fields is 0, we can compute the variance
        # as the mean of the norms squared corrected with Bessel's correction
        norms_reshaped = norms.reshape(b, self.num_groups, -1)
        correction = (
            norms_reshaped.shape[1] / (norms_reshaped.shape[1] - 1)
            if norms_reshaped.shape[1] > 1
            else 1
        )
        group_variance = (
            norms_reshaped.mean(dim=2) / (self.reps_size + self.num_groups)
        ) * correction

        # Expand to match input size
        group_variance_reshaped = torch.repeat_interleave(
            group_variance, (c // self.num_groups), dim=1
        )
        group_variance_reshaped = repeat(
            group_variance_reshaped, "B (C H W) -> B C (h H) (w W)", H=1, W=1, h=h, w=w
        )

        # Normalize
        output = torch.divide(input.tensor, (group_variance_reshaped + self.eps).sqrt())

        # Merge channel and reps dimensions and wrap result in GroupTensor
        return GroupTensor(output, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, x: torch.Tensor = None, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        return super(GroupNorm, self).check_equivariance(x=x, atol=atol, rtol=rtol)

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        main_str = self._get_name() + "("
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(extra_lines) + "\n"

        main_str += ")"
        return main_str

    def extra_repr(self):
        return "{in_type}, eps={eps}, affine=False".format(**self.__dict__)


class GroupStandardization(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        num_groups: int,
        eps: float = 1e-05,
    ):
        r"""

        Group "norm" for generic representations of 2D data (i.e. 4D inputs).

        Based on weight standardization, works with any representation.

        Args:
            in_type (FieldType): Input field type / representation.
            num_groups (int): Number of groups to divide the channesl into.
            eps (float, default: ``1e-5``): Added to the denominator for numerical stability.

        """

        assert isinstance(in_type.gspace, GSpace)

        super(GroupStandardization, self).__init__()

        reps_sizes = [r.size for r in in_type.representations]
        assert reps_sizes.count(reps_sizes[0]) == len(
            reps_sizes
        ), f"Representations need to be the same size but are {set(reps_sizes)}."

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        self.eps = eps

        self.reps_size = reps_sizes[0]
        self.channels = len([r.size for r in in_type.representations])

        # [dasd] = grouped_fields.items()

        if self.channels % num_groups != 0:
            print(
                f"INFO: Number of channels is not divisible by number of groups"
                f" ({self.channels}/{num_groups}). Switching to LayerNorm."
            )
            self.num_groups = 1  # layer norm
        elif num_groups >= self.channels:
            print(
                f"INFO: Number of groups is bigger or equal to number of channels"
                f" ({self.channels}). Using InstanceNorm."
            )
            self.num_groups = self.channels  # instance norm
        else:
            self.num_groups = num_groups

        self._nfields = defaultdict(int)

        # indices of the channales corresponding to fields belonging to each group
        _indices = defaultdict(list)

        position = 0
        for r in self.in_type.representations:
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

            _indices[id] += list(range(position, position + r.size))
            self._nfields[id] += 1
            position += r.size

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        in_tensor = rearrange(
            input.tensor,
            "B (G C) H W -> B G C H W",
            G=self.num_groups,
        )

        std, mean = torch.std_mean(
            in_tensor, dim=(2, 3, 4), unbiased=False, keepdim=True
        )
        output = (in_tensor - mean) / (std.expand_as(in_tensor - mean) + self.eps)

        output = rearrange(
            output,
            "B G C H W -> B (G C) H W",
            G=self.num_groups,
        )

        return GroupTensor(output, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, x: torch.Tensor = None, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        return super(GroupStandardization, self).check_equivariance(
            x=x, atol=atol, rtol=rtol
        )

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        main_str = self._get_name() + "("
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(extra_lines) + "\n"

        main_str += ")"
        return main_str

    def extra_repr(self):
        return "{in_type}, eps={eps}, affine=False".format(**self.__dict__)
