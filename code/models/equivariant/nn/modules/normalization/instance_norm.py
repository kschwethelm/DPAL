from collections import defaultdict

from .....equivariant.nn import *
from .....equivariant.nn import FieldType, GroupTensor
from .....equivariant.nn.modules import EquivariantModule

import torch
from torch.nn import Parameter
import numpy as np

from abc import ABC

__all__ = ["IIDInstaceNorm"]


class IIDInstanceNorm(EquivariantModule, ABC):
    r"""

    Instance normalization for generic representations for 2D data (i.e. 4D inputs).

    This instance normalization assumes that all dimensions within the same field have the same variance, i.e. that
    the covariance matrix of each field in `in_type` is a scalar multiple of the identity.
    Moreover, the mean is only computed over the trivial irreps occourring in the input representations (the input
    representation does not need to be decomposed into a direct sum of irreps since this module can deal with the
    change of basis).

    Similarly, if `affine = True`, a single scale is learnt per input field and the bias is applied only to the
    trivial irreps.

    This assumption is equivalent to the usual Batch Normalization in a Instance Convolution NN (GCNN), where
    statistics are shared over the group dimension.
    See Chapter 4.2 at `https://gabri95.github.io/Thesis/thesis.pdf <https://gabri95.github.io/Thesis/thesis.pdf>`_ .

    Args:
        in_type (FieldType): the input field type
        affine (bool, optional): if ``True``, this module has learnable affine parameters. Default: ``True``
        eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``

    """

    def __init__(
        self,
        in_type: FieldType,
        affine: bool = True,
        eps: float = 1e-05,
    ):
        assert isinstance(in_type.gspace, GSpace)

        super(IIDInstanceNorm, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type
        self.affine = affine
        self.eps = eps

        # Retrieve the indices of the channels corresponding to fields belonging to each group
        self._nfields = defaultdict(int)  # number of fields of each type
        _indices = defaultdict(list)

        ntrivials = 0
        position = 0
        for r in self.in_type.representations:
            for irr in r.irreps:
                if self.in_type.fibergroup.irrep(*irr).is_trivial():
                    ntrivials += 1

            # Only the first and last indices are kept
            _indices[r.name] = [position, position + r.size]
            self._nfields[r.name] += 1
            position += r.size

            setattr(self, f"indices_{self._escape_name(r.name)}", _indices[r.name])

        # Store the size of each field type
        self._sizes = []
        self._has_trivial = {}

        # For each different representation in the input type
        for r in self.in_type._unique_representations:
            p = 0
            trivials = []

            # Mask containing the location of the trivial irreps in the irrep decomposition of the representation
            S = np.zeros((r.size, r.size))

            # Find all trivial irreps occurring in the representation
            for irr in r.irreps:
                irr = self.in_type.fibergroup.irrep(*irr)
                if irr.is_trivial():
                    trivials.append(p)
                    S[p, p] = 1.0
                p += irr.size

            name = r.name
            self._sizes.append((name, r.size))
            self._has_trivial[name] = len(trivials) > 0

            if self._has_trivial[name]:
                # Averaging matrix which computes the expectation of a input vector, i.e. projects it in the trivial
                # subspace by masking out all non-trivial irreps
                P = r.change_of_basis @ S @ r.change_of_basis_inv
                self.register_buffer(
                    f"avg_{self._escape_name(name)}", torch.tensor(P, dtype=torch.float)
                )

                Q = torch.tensor(r.change_of_basis, dtype=torch.float)[:, trivials]
                self.register_buffer(f"change_of_basis_{self._escape_name(name)}", Q)

            if self.affine:
                # Scale all dimensions of the same field by the same weight
                weight = Parameter(
                    torch.ones((self._nfields[r.name], 1)), requires_grad=True
                )
                self.register_parameter(f"weight_{self._escape_name(name)}", weight)
                if self._has_trivial[name]:
                    # Bias is applied only to the trivial channels
                    bias = Parameter(
                        torch.zeros((self._nfields[r.name], len(trivials))),
                        requires_grad=True,
                    )
                    self.register_parameter(f"bias_{self._escape_name(name)}", bias)

    def reset_parameters(self):
        if self.affine:
            for name, _ in self._sizes:
                weight = getattr(self, f"weight_{self._escape_name(name)}")
                weight.data.fill_(1)
                if hasattr(self, f"bias_{self._escape_name(name)}"):
                    bias = getattr(self, f"bias_{self._escape_name(name)}")
                    bias.data.fill_(0)

    def _estimate_stats(self, slice, name: str):
        agg_axes = tuple(range(3, len(slice.shape)))

        if self._has_trivial[name]:
            P = getattr(self, f"avg_{self._escape_name(name)}")

            # Compute the mean
            means = torch.einsum(
                "ij,bcj...->bci...", P, slice.mean(dim=agg_axes, keepdim=True).detach()
            )
            centered = slice - means
            means = means.reshape(slice.shape[0], means.shape[1], means.shape[2])
        else:
            means = None
            centered = slice

        # Center the data and compute the variance - set to 0 if spatial dims of size (1, 1)
        # We implicitly assume the dimensions to be iid, i.e. covariance matrix is scalar multiple of identity
        if slice.shape[-2:] == (1, 1):
            vars = torch.ones(
                size=[slice.shape[0], slice.shape[1], *(1,) * (len(slice.shape) - 2)]
            )
        else:
            vars = (
                centered.var(dim=agg_axes, unbiased=True, keepdim=False)
                .mean(dim=2, keepdim=True)
                .detach()
            )

        return means, vars

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""
        Apply norm non-linearities to the input feature map

        Args:
            input (GroupTensor): the input feature map

        Returns:
            Normalized input feature map.
        """

        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b = input.shape[0]
        spatial_dims = input.shape[2:]

        output = torch.empty_like(input)

        # iterate through all field types
        for name, size in self._sizes:
            indices = getattr(self, f"indices_{self._escape_name(name)}")

            slice = input[:, indices[0] : indices[1], ...]
            slice = slice.view(b, -1, size, *spatial_dims)

            means, vars = self._estimate_stats(slice, name)

            if self._has_trivial[name]:
                # center data by subtracting the mean
                slice = slice - means.view(
                    b, means.shape[1], means.shape[2], *(1,) * len(spatial_dims)
                )

            # normalize dividing by the std and multiply by the new scale
            if self.affine:
                weight = getattr(self, f"weight_{self._escape_name(name)}")
            else:
                weight = 1.0

            # compute the scalar multipliers needed
            scales = weight / (vars + self.eps).sqrt()  # NOTE: float in batchnorm
            # scale features
            slice = slice * scales.view(
                b, scales.shape[1], scales.shape[2], *(1,) * len(spatial_dims)
            )

            # shift the features with the learnable bias
            if self.affine and self._has_trivial[name]:
                bias = getattr(self, f"bias_{self._escape_name(name)}")
                Q = getattr(self, f"change_of_basis_{self._escape_name(name)}")
                slice = slice + torch.einsum("ij,cj->ci", Q, bias).view(
                    1, bias.shape[0], Q.shape[0], *(1,) * len(spatial_dims)
                )

            output[:, indices[0] : indices[1], ...] = slice.mean(
                dim=1, keepdim=True
            ).view(b, -1, *spatial_dims)

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) > 1, input_shape
        assert input_shape[1] == self.in_type.size, input_shape

        return (input_shape[0], self.out_type.size, *input_shape[2:])

    def _escape_name(self, name: str):
        return name.replace(".", "^")

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
        return "{in_type}, eps={eps}, affine={affine}".format(**self.__dict__)

    # TODO: Test this
    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.InstanceNorm` module and set to "eval" mode.

        """
        self.eval()

        instancenorm = torch.nn.InstanceNorm2d(
            self.in_type.size,
            self.eps,
            affine=self.affine,
        )

        for name, size in self._sizes:
            start, end = getattr(self, "{}_indices".format(name))
            n = self._nfields[name]

            if self.affine:
                weight = getattr(self, "{}_weight".format(name))
                instancenorm.weight.data[start:end] = (
                    weight.data.view(n, 1).expand(n, size).reshape(-1)
                )

            if self._has_trivial[name]:
                if self.affine:
                    bias = getattr(self, "{}_bias".format(name))
                    Q = getattr(self, "{}_change_of_basis".format(name))
                    bias = torch.einsum("ij,cj->ci", Q, bias)
                    instancenorm.bias.data[start:end] = bias.data.view(n, size).reshape(
                        -1
                    )
            else:
                if self.affine:
                    instancenorm.bias.data[start:end] = 0.0

        instancenorm.eval()

        return instancenorm

    def _check_input_shape(self, shape):
        if len(shape) != 4:
            raise ValueError(
                "Error, expected a 4D tensor but a {} one was found".format(len(shape))
            )
