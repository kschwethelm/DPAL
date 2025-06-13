from .....equivariant.group_theory import Representation, KernelBasis, EmptyBasisException

from collections import defaultdict
import operator
from typing import Callable, List, Iterable, Dict, Tuple

import torch
import numpy as np


__all__ = ["BasisExpansion"]


class BasisExpansion(torch.nn.Module):
    def __init__(
        self,
        in_reprs: List[Representation],
        out_reprs: List[Representation],
        basis_generator: Callable[[Representation, Representation], KernelBasis],
        points: np.ndarray,
        basis_filter: Callable[[dict], bool] = None,
    ):
        r"""
        Method that performs the expansion of a (already sampled) filter basis.

        Args:
            in_reprs (list): the input field type
            out_reprs (list): the output field type
            basis_generator (callable): method that generates the analytical filter basis
            points (~numpy.ndarray): points where the analytical basis should be sampled
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.

        Attributes:
            S (int): number of points where the filters are sampled

        """

        super(BasisExpansion, self).__init__()
        self._in_reprs = in_reprs
        self._out_reprs = out_reprs
        self._input_size = sum(r.size for r in in_reprs)
        self._output_size = sum(r.size for r in out_reprs)
        self.points = points

        # int: number of points where the filters are sampled
        self.S = self.points.shape[0]

        # Build a basis for all input/output representations pairs
        self._bases = {}
        self._masks = {}
        _sampled_bases = {}
        # Sort irreps
        in_reprs = sorted(in_reprs, key=operator.attrgetter("name"))
        out_reprs = sorted(out_reprs, key=operator.attrgetter("name"))

        for i_repr in set(in_reprs):
            for o_repr in set(out_reprs):
                reprs_names = (i_repr.name, o_repr.name)
                try:
                    basis = basis_generator(i_repr, o_repr)

                    # Compute sampled basis mask containing only elements allowed by filter
                    if basis_filter is not None:
                        mask = np.zeros(len(basis), dtype=bool)
                        for b, attr in enumerate(basis):
                            mask[b] = basis_filter(attr)
                    else:
                        mask = np.ones(len(basis), dtype=bool)

                    _sampled_bases[
                        reprs_names[0] + "->" + reprs_names[1]
                    ] = self._sample_bases(basis, points, mask)
                    self._bases[reprs_names] = basis
                    self._masks[reprs_names] = mask

                except EmptyBasisException:  # TODO: Delete?
                    # print(f"Empty basis at {reprs_names}")
                    pass

        # Register sampled kernel bases as attribute
        self.sampled_bases = _sampled_bases

        if len(_sampled_bases) == 0:
            print("WARNING! The basis for the block expansion of the filter is empty!")

        # List of all pairs of input/output representations which don't have an empty basis
        # NOTE: For discrete groups, this is only always one pair made out of i_repr.name and o_repr.name
        self._representations_pairs = sorted(
            [io[0] + "->" + io[1] for io in self._bases.keys()]
        )

        # retrieve for each representation in both input and output fields:
        # - the number of its occurrences,
        # - the indices where it occurs
        self._in_count, _in_indices = self._retrieve_indices(in_reprs)
        self._out_count, _out_indices = self._retrieve_indices(out_reprs)

        self._weights_ranges = {}

        last_weight_position = 0

        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        self.in_indices = {}
        self.out_indices = {}
        for io_pair in self._representations_pairs:
            in_indices = torch.LongTensor(
                [
                    _in_indices[io_pair.split("->")[0]].min(),
                    _in_indices[io_pair.split("->")[0]].max() + 1,
                    (_in_indices[io_pair.split("->")[0]].max() + 1)
                    - _in_indices[io_pair.split("->")[0]].min(),
                ]
            )
            out_indices = torch.LongTensor(
                [
                    _out_indices[io_pair.split("->")[1]].min(),
                    _out_indices[io_pair.split("->")[1]].max() + 1,
                    (_out_indices[io_pair.split("->")[1]].max() + 1)
                    - _out_indices[io_pair.split("->")[1]].min(),
                ]
            )

            self.in_indices[io_pair] = in_indices
            self.out_indices[io_pair] = out_indices

            # number of occurrences of the input/output pair `io_pair`
            n_pairs = (
                self._in_count[io_pair.split("->")[0]]
                * self._out_count[io_pair.split("->")[1]]
            )

            # count the actual number of parameters
            total_weights = _sampled_bases[io_pair].shape[0] * n_pairs

            # evaluate indices in the global weights tensor to use for the basis belonging to this group
            self._weights_ranges[io_pair] = (
                last_weight_position,
                last_weight_position + total_weights,
            )

            # increment the position counter
            last_weight_position += total_weights

        self._dim = last_weight_position

    def get_basis_info(self) -> Iterable[Dict]:
        out_irreps_counts = [0]
        out_block_counts = defaultdict(list)
        for o, o_repr in enumerate(self._out_reprs):
            out_irreps_counts.append(out_irreps_counts[-1] + len(o_repr.irreps))
            out_block_counts[o_repr.name].append(o)

        in_irreps_counts = [0]
        in_block_counts = defaultdict(list)
        for i, i_repr in enumerate(self._in_reprs):
            in_irreps_counts.append(in_irreps_counts[-1] + len(i_repr.irreps))
            in_block_counts[i_repr.name].append(i)

        # Iterate through the different group of blocks
        # i.e., through all input/output pairs
        idx = 0
        for reprs_names in sorted(list(self._bases.keys())):
            attrs = []
            for i, attr in enumerate(self._bases[reprs_names]):
                if self._masks[reprs_names][i]:
                    attr["id"] = len(attrs)
                    attrs += [attr]

            for o in out_block_counts[reprs_names[1]]:
                out_irreps_count = out_irreps_counts[o]
                for i in in_block_counts[reprs_names[0]]:
                    in_irreps_count = in_irreps_counts[i]

                    # Retrieve the attributes of each basis element and build a new list of
                    # attributes adding information specific to the current block
                    for attr in attrs:
                        attr = attr.copy()
                        attr.update(
                            {
                                "in_irreps_position": in_irreps_count
                                + attr["in_irrep_idx"],
                                "out_irreps_position": out_irreps_count
                                + attr["out_irrep_idx"],
                                "in_repr": reprs_names[0],
                                "out_repr": reprs_names[1],
                                "in_field_position": i,
                                "out_field_position": o,
                            }
                        )

                        attr["block_id"] = attr["id"]
                        attr["id"] = idx

                        assert idx < self._dim

                        idx += 1

                        yield attr

    def dimension(self) -> int:
        return self._dim

    def _retrieve_indices(self, reprs: List[Representation]):
        fiber_position = 0
        _indices = defaultdict(list)
        _count = defaultdict(int)

        for repr in reprs:
            _indices[repr.name] += list(
                range(fiber_position, fiber_position + repr.size)
            )
            fiber_position += repr.size
            _count[repr.name] += 1

        for name, indices in _indices.items():
            _indices[name] = torch.LongTensor(indices)

        return _count, _indices

    def _normalize_basis(
        self, basis: torch.Tensor, sizes: torch.Tensor
    ) -> torch.Tensor:
        r"""

        Normalize the filters in the input tensor.
        The tensor of shape :math:`(B, O, I, ...)` is interpreted as a basis containing ``B`` filters/elements, each with
        ``I`` inputs and ``O`` outputs. The spatial dimensions ``...`` can be anything.

        .. notice ::
            Notice that the method changes the input tensor inplace

        Args:
            basis (torch.Tensor): tensor containing the basis to normalize
            sizes (torch.Tensor): original input size of the basis elements, without the padding and the change of basis

        Returns:
            the normalized basis (the operation is done inplace, so this is ust a reference to the input tensor)

        """

        b = basis.shape[0]
        assert len(basis.shape) > 2, basis.shape
        assert sizes.shape == (b,), (sizes.shape, b, basis.shape)

        # compute the norm of each basis vector
        norms = torch.einsum("bop...,bpq...->boq...", (basis, basis.transpose(1, 2)))

        # Removing the change of basis, these matrices should be multiples of the identity
        # where the scalar on the diagonal is the variance
        # in order to find this variance, we can compute the trace (which is invariant to the change of basis)
        # and divide by the number of elements in the diagonal ignoring the padding.
        # Therefore, we need to know the original size of each basis element.
        norms = torch.einsum("bii...->b", norms)

        norms /= sizes

        norms[norms < 1e-15] = 0

        norms = torch.sqrt(norms)

        norms[norms < 1e-6] = 1
        norms[norms != norms] = 1

        norms = norms.view(b, *([1] * (len(basis.shape) - 1)))

        # divide by the norm
        basis /= norms

        return basis

    def _sample_bases(self, basis, points, mask):
        if mask is None:
            mask = np.ones(len(basis), dtype=bool)

        assert mask.shape == (len(basis),) and mask.dtype == bool

        if not mask.any():
            raise EmptyBasisException

        # we need to know the real output size of the basis elements (i.e. without the change of basis and the padding)
        # to perform the normalization
        sizes = []
        for attr in basis:
            sizes.append(attr["shape"][0])

        # sample the basis on the grid
        sampled_basis = basis.sample(torch.tensor(points, dtype=torch.float32)).permute(
            1, 2, 3, 0
        )

        # normalize the basis
        sizes = torch.tensor(sizes, dtype=sampled_basis.dtype)
        sampled_basis = self._normalize_basis(sampled_basis, sizes)

        # discard the basis which are close to zero everywhere
        norms = (sampled_basis**2).reshape(sampled_basis.shape[0], -1).sum(1) > 1e-2
        mask = torch.tensor(mask) & norms
        if not mask.any():
            raise EmptyBasisException
        sampled_basis = sampled_basis[mask, ...]

        # register the bases tensors as parameters of this module
        return sampled_basis.to(f"cuda:{torch.cuda.current_device()}")

    def _expand_blocks(
        self,
        weights: torch.Tensor,
        reprs_pairs: List[str],
        sampled_bases: Dict[str, torch.Tensor],
        output_size: int,
        input_size: int,
        out_count: Dict[str, int],
        in_count: Dict[str, int],
        out_indices: Dict[str, torch.Tensor],
        in_indices: Dict[str, torch.Tensor],
        weights_ranges: Dict[str, Tuple[int, int]],
        kernel_size: int,
    ):
        # Build tensor which will contain the filter
        _filter = torch.zeros(
            output_size,
            input_size,
            kernel_size,
            device=weights.device,
            dtype=torch.float32,
        )

        # Iterate through all input-output field representations pairs
        for io_pair in reprs_pairs:
            coefficients = weights[
                weights_ranges[io_pair][0] : weights_ranges[io_pair][1]
            ]
            # Reshape coefficients for the batch matrix multiplication
            coefficients = coefficients.view(-1, sampled_bases[io_pair].shape[0])

            assert len(coefficients.shape) == 2 and (
                coefficients.shape[1] == sampled_bases[io_pair].shape[0]
            )

            # Expand current subset of basis vectors and set result in the appropriate place in the filter
            _filter_block = torch.einsum(
                "boi...,kb->koi...",
                sampled_bases[io_pair],
                coefficients,
            )

            _filter_block = _filter_block.view(
                out_count[io_pair.split("->")[1]],
                in_count[io_pair.split("->")[0]],
                _filter_block.shape[1],
                _filter_block.shape[2],
                kernel_size,
            )
            _filter_block = _filter_block.transpose(1, 2).float()

            _filter[
                out_indices[io_pair][0] : out_indices[io_pair][1],
                in_indices[io_pair][0] : in_indices[io_pair][1],
                :,
            ] = _filter_block.reshape(
                out_indices[io_pair][2].item(),
                in_indices[io_pair][2].item(),
                kernel_size,
            )

        return _filter

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the Module which expands the basis and returns the filter built

        Args:
            weights (torch.Tensor): the learnable weights used to linearly combine the basis filters

        Returns:
            the filter built

        """
        assert weights.shape[0] == self.dimension()
        assert len(weights.shape) == 1

        _filter = self._expand_blocks(
            weights,
            self._representations_pairs,
            self.sampled_bases,
            self._output_size,
            self._input_size,
            self._out_count,
            self._in_count,
            self.out_indices,
            self.in_indices,
            self._weights_ranges,
            self.S,
        )

        return _filter

    def __hash__(self):
        _hash = 0
        for io in self._representations_pairs:
            n_pairs = (
                self._in_count[io.split("->")[0]] * self._out_count[io.split("->")[1]]
            )
            _hash += hash(self.sampled_bases[io].shape[0]) * n_pairs

        return _hash

    def __eq__(self, other):
        if not isinstance(other, BasisExpansion):
            return False

        if self._dim != other._dim:
            return False

        if self._representations_pairs != other._representations_pairs:
            return False

        for io in self._representations_pairs:
            if self._weights_ranges[io] != other._weights_ranges[io]:
                return False

            if torch.any(self.in_indices[io] != other.in_indices[io]):
                return False
            if torch.any(self.out_indices[io] != other.out_indices[io]):
                return False

            if not torch.any(self.sampled_bases[io] != other.sampled_bases[io]):
                return False

        return True
