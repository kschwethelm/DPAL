from collections import defaultdict
from typing import List, Tuple, Any

import torch
from torch.nn import Parameter, BatchNorm3d

from ...field_type import FieldType
from ...group_tensor import GroupTensor
from ..utils import indexes_from_labels
from .....equivariant.nn import *
from .....equivariant.nn.modules import EquivariantModule


__all__ = ["BatchNorm", "InducedNormBatchNorm"]


class BatchNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        affine: bool = True,
        eps: float = 1e-05,
        momentum: float = 0.1,
        track_running_stats: bool = False,
    ):
        r"""

        Batch normalization for representations with permutation matrices.

        Statistics are computed both over the batch and the spatial dimensions and over the channels within
        the same field (which are permuted by the representation).

        Only representations supporting pointwise non-linearities are accepted as input field type.

        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                    Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
            affine (bool, optional):  if ``True``, this module has learnable affine parameters. Default: ``True``
            track_running_stats (bool, optional): when set to ``True``, the module tracks the running mean and variance;
                                                  when set to ``False``, it does not track such statistics but uses
                                                  batch statistics in both training and eval modes.
                                                  Default: ``True``

        """

        assert isinstance(in_type.gspace, GSpace)

        super(BatchNorm, self).__init__()

        for r in in_type.representations:
            assert (
                "pointwise" in r.supported_nonlinearities
            ), 'Error! Representation "{}" does not support "pointwise" non-linearity'.format(
                r.name
            )

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        # Group fields by their size and retrieve the indices of the fields
        grouped_fields = indexes_from_labels(
            self.in_type, [r.size for r in self.in_type.representations]
        )

        # number of fields of each size
        self._nfields = {}

        # indices of the channels corresponding to fields belonging to each group
        _indices = {}
        for s, (fields, indices) in grouped_fields.items():
            self._nfields[s] = len(fields)
            _indices[s] = torch.LongTensor([min(indices), max(indices) + 1])

            # register the indices tensors as parameters of this module
            self.register_buffer("indices_{}".format(s), _indices[s])

        for s in _indices.keys():
            _batchnorm = BatchNorm3d(
                self._nfields[s],
                self.eps,
                self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )
            self.add_module("batch_norm_[{}]".format(s), _batchnorm)

    def reset_running_stats(self):
        for s in list(self._nfields.keys()):
            batchnorm = getattr(self, f"batch_norm_[{s}]")
            batchnorm.reset_running_stats()

    def reset_parameters(self):
        for s in list(self._nfields.keys()):
            batchnorm = getattr(self, f"batch_norm_[{s}]")
            batchnorm.reset_parameters()

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Args:
            input (GroupTensor): the input feature map
        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        b, c, h, w = input.tensor.shape

        output = torch.empty_like(input.tensor)

        # iterate through all field sizes
        for s in list(self._nfields.keys()):
            indices = getattr(self, f"indices_{s}")
            batchnorm = getattr(self, f"batch_norm_[{s}]")

            output[:, indices[0] : indices[1], :, :] = batchnorm(
                input.tensor[:, indices[0] : indices[1], :, :].view(b, -1, s, h, w)
            ).view(b, -1, h, w)

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        # return super(BatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.BatchNorm2d` module and set to "eval" mode.
        """

        if not self.track_running_stats:
            raise ValueError(
                """
                Equivariant Batch Normalization can not be converted into conventional batch normalization when
                "track_running_stats" is False because the statistics contained in a single batch are generally
                not symmetric
            """
            )

        self.eval()

        batchnorm = torch.nn.BatchNorm2d(
            self.in_type.size,
            self.eps,
            self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )

        num_batches_tracked = None

        for s in list(self._nfields.keys()):
            start, end = getattr(self, "indices_{}".format(s))
            bn = getattr(self, "batch_norm_[{}]".format(s))

            n = self._nfields[s]

            batchnorm.running_var.data[start:end] = (
                bn.running_var.data.view(n, 1).expand(n, s).reshape(-1)
            )
            batchnorm.running_mean.data[start:end] = (
                bn.running_mean.data.view(n, 1).expand(n, s).reshape(-1)
            )
            batchnorm.num_batches_tracked.data = bn.num_batches_tracked.data

            if num_batches_tracked is None:
                num_batches_tracked = bn.num_batches_tracked.data
            else:
                assert num_batches_tracked == bn.num_batches_tracked.data

            if self.affine:
                batchnorm.weight.data[start:end] = (
                    bn.weight.data.view(n, 1).expand(n, s).reshape(-1)
                )
                batchnorm.bias.data[start:end] = (
                    bn.bias.data.view(n, 1).expand(n, s).reshape(-1)
                )

        batchnorm.eval()

        return batchnorm

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
        return "{in_type}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}".format(
            **self.__dict__
        )


class InducedNormBatchNorm(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        r"""
        Batch normalization for induced isometric representations.
        This module requires the input fields to be associated to an induced representation from an isometric
        (i.e. which preserves the norm) non-trivial representation which supports 'norm' non-linearities.

        The module assumes the mean of the vectors is always zero so no running mean is computed and no bias is added.
        This is guaranteed as long as the representations do not include a trivial representation.

        Indeed, if :math:`\rho` does not include a trivial representation, it holds:

        .. math ::

             \forall \bold{v} \in \mathbb{R}^n, \ \ \frac{1}{|G|} \sum_{g \in G} \rho(g) \bold{v} = \bold{0}
        Hence, only the standard deviation is normalized.
        The same standard deviation, however, is shared by all the sub-fields of the same induced field.

        The input representation of the fields is preserved by this operation.

        Only representations which do not contain the trivial representation are allowed.
        You can check if a representation contains the trivial representation using
        :meth:`~torch-ecnn.Representation.contains_trivial`.
        To check if a trivial irrep is present in a representation in a :class:`~torch-ecnn.nn.FieldType`, you can use::

            for r in field_type:
                if r.contains_trivial():
                    print(f"field type contains a trivial irrep")
        Args:
            in_type (FieldType): the input field type
            eps (float, optional): a value added to the denominator for numerical stability. Default: ``1e-5``
            momentum (float, optional): the value used for the ``running_mean`` and ``running_var`` computation.
                    Can be set to ``None`` for cumulative moving average (i.e. simple average). Default: ``0.1``
            affine (bool, optional): if ``True``, this module has learnable scale parameters. Default: ``True``
        """

        assert isinstance(in_type.gspace, GSpace)

        super(InducedNormBatchNorm, self).__init__()

        for r in in_type.representations:
            assert any(
                nl.startswith("induced_norm") for nl in r.supported_nonlinearities
            ), 'Error! Representation "{}" does not support "induced_norm" non-linearity'.format(
                r.name
            )
            # Norm batch-normalization assumes the fields to have mean 0. This is true as long as it doesn't contain
            # the trivial representation
            for irr in r.irreps:
                assert not in_type.fibergroup._irreps[irr].is_trivial()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        self.affine = affine

        # Group fields by their size and retrieve the indices of the fields
        # number of fields of each size
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

        for id in list(self._nfields.keys()):
            _indices[id] = torch.LongTensor([min(_indices[id]), max(_indices[id]) + 1])

            # register the indices tensors as parameters of this module
            self.register_buffer(f"{id}_indices", _indices[id])

            running_var = torch.ones(
                (1, self._nfields[id], 1, 1, 1, 1), dtype=torch.float
            )
            self.register_buffer(f"{id}_running_var", running_var)

            if self.affine:
                weight = Parameter(
                    torch.ones((1, self._nfields[id], 1, 1, 1, 1)), requires_grad=True
                )
                self.register_parameter(f"{id}_weight", weight)

        _indices = dict(_indices)

        self._order = list(_indices.keys())

        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        self.eps = eps
        self.momentum = momentum

    def reset_running_stats(self):
        for s in self._order:
            running_var = getattr(self, f"{s}_running_var")
            running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        for s in self._order:
            weight = getattr(self, f"{s}_weight")
            weight.data.fill_(1)

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""
        Apply norm non-linearities to the input feature map

        Args:
            input (GroupTensor): the input feature map
        Returns:
            the resulting feature map

        """

        assert input.type == self.in_type

        exponential_average_factor = 0.0

        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # compute the squares of the values of each channel
        # n = torch.mul(input.tensor, input.tensor)
        n = input.tensor.detach() ** 2

        b, c, h, w = input.tensor.shape

        output = input.tensor.clone()

        if self.training:
            # self.running_var *= 1 - exponential_average_factor

            next_var = 0
            # iterate through all field sizes
            for id in self._order:
                size, subfield_size = id
                n_subfields = int(size // subfield_size)

                indices = getattr(self, f"{id}_indices")
                running_var = getattr(self, f"{id}_running_var")

                # compute the norm of each field by summing the squares
                norms = (
                    n[:, indices[0] : indices[1], :, :]
                    .view(b, -1, n_subfields, subfield_size, h, w)
                    .sum(dim=3, keepdim=False)
                )

                # Since the mean of the fields is 0, we can compute the variance as the mean of the norms squared
                # corrected with Bessel's correction
                norms = norms.transpose(0, 1).reshape(self._nfields[id], -1)
                correction = (
                    norms.shape[1] / (norms.shape[1] - 1) if norms.shape[1] > 1 else 1
                )
                vars = norms.mean(dim=1).view(1, -1, 1, 1, 1, 1) / subfield_size
                vars *= correction
                # vars = norms.transpose(0, 1).reshape(self._nfields[s], -1).var(dim=1)

                # self.running_var[next_var:next_var + self._nfields[s]] += exponential_average_factor * vars
                running_var *= 1 - exponential_average_factor
                running_var += exponential_average_factor * vars  # .detach()

                next_var += self._nfields[id]

            # self.running_var = self.running_var.detach()

        next_var = 0

        # iterate through all field sizes
        for id in self._order:
            size, subfield_size = id
            n_subfields = int(size // subfield_size)

            indices = getattr(self, f"{id}_indices")

            # retrieve the running variances corresponding to the current fields
            vars = getattr(self, f"{id}_running_var")
            if self.affine:
                weight = getattr(self, f"{id}_weight")
            else:
                weight = 1.0

            # compute the scalar multipliers needed
            multipliers = weight / (vars + self.eps).sqrt()

            # expand the multipliers tensor to all channels for each field
            multipliers = multipliers.expand(
                b, -1, n_subfields, subfield_size, h, w
            ).reshape(b, -1, h, w)

            output[:, indices[0] : indices[1], :, :] *= multipliers

            # shift the position on the running_var and weight tensors
            next_var += self._nfields[id]

        # wrap the result in a GroupTensor
        return GroupTensor(output, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, atol: float = 1e-6, rtol: float = 1e-5
    ) -> List[Tuple[Any, float]]:
        # return super(NormBatchNorm, self).check_equivariance(atol=atol, rtol=rtol)
        pass
