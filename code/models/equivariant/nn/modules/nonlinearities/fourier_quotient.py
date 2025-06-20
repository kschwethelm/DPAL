from .....equivariant.group_theory import *
from .....equivariant.nn import GSpace, FieldType, GroupTensor

from ..equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any

import numpy as np

__all__ = ["QuotientFourierPointwise", "QuotientFourierELU"]


def _build_kernel(G: Group, subgroup_id: Tuple, irrep: List[tuple]):
    kernel = []

    X: HomSpace = G.homspace(subgroup_id)

    for irr in irrep:
        k = X._dirac_kernel_ft(irr, X.H.trivial_representation.id)
        # irr = G.irrep(*irr)
        # K *= np.sqrt(irr.size)
        kernel.append(k.T.reshape(-1))

    kernel = np.concatenate(kernel)
    return kernel


class QuotientFourierPointwise(EquivariantModule):
    def __init__(
        self,
        gspace: GSpace,
        subgroup_id: Tuple,
        channels: int,
        irreps: List,
        *grid_args,
        grid: List[GroupElement] = None,
        function: str = "p_relu",
        inplace: bool = True,
        **grid_kwargs,
    ):
        r"""

        Applies a Inverse Fourier Transform to sample the input features on a *quotient space* :math:`X`, apply the
        pointwise non-linearity in the spatial domain (Dirac-delta basis) and, finally, computes the Fourier Transform
        to obtain irreps coefficients.
        The quotient space used is isomorphic to :math:`X \cong G / H` where :math:`G` is ```gspace.fibergroup``` while
        :math:`H` is the subgroup of :math:`G` idenitified by ```subgroup_id```; see
        :meth:`~Group.subgroup` and :meth:`~Group.homspace`

        .. note::
            This operation is only *approximately* equivariant and its equivariance depends on the sampling grid and the
            non-linear activation used, as well as the original band-limitation of the input features.

        The same function is applied to every channel independently.
        The input representation is preserved by this operation and, therefore, it equals the output representation.

        The class first constructs a band-limited quotient representation of ```gspace.fibergroup``` using
        :meth:`Group.spectral_quotient_representation`.
        The band-limitation of this representation is specified by ```irreps``` which should be a list containing
        a list of ids identifying irreps of ```gspace.fibergroup``` (see :attr:`IrreducibleRepresentation.id`).
        This representation is used to define the input and output field types, each containing ```channels``` copies
        of a feature field transforming according to this representation.
        A feature vector transforming according to such representation is interpreted as a vector of coefficients
        parameterizing a function over the group using a band-limited Fourier basis.

        To approximate the Fourier transform, this module uses a finite number of samples from the group.
        The set of samples to be used can be specified through the parameter ```grid``` or by the ```grid_args``` and
        ```grid_kwargs``` which will then be passed to the method :meth:`~Group.grid`.

        .. warning ::
            By definition, an homogeneous space is invariant under a right action of the subgroup :math:`H`.
            That means that a feature representing a function over a homogeneous space :math:`X \cong G/H`, when
            interpreted as a function over :math:`G` (as we do here when sampling), the function will be constant along
            each coset, i.e. :math:`f(gh) = f(g)` if :math:`g \in G, h\in H`.
            An approximately uniform sampling grid over :math:`G` creates an approximately uniform grid over :math:`G/H`
            through projection but might contain redundant elements (if the grid contains :math:`g 'in G`, any element
            :math:`gh` in the grid will be redundant).
            It is therefore advised to create a grid directly in the quotient space, e.g. using
            :meth:`group.SO3.sphere_grid`, :meth:`group.O3.sphere_grid`.
            We do not support yet a general method and interface to generate grids over any homogeneous space for any
            group, so you should check each group's methods.

        .. todo::
            Mention the normalization of the transform we use

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            subgroup_id (tuple): identifier of the subgroup :math:`H` to construct the quotient space
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            *grid_args: parameters used to construct the discretization grid
            grid (list, optional): list containing the elements of the group to use for sampling. Optional (default ``None``).
            function (str): the identifier of the non-linearity.
                    It is used to specify which function to apply.
                    By default (``'p_relu'``), ReLU is used.
            inplace (bool): applies the non-linear activation in-place. Default: `True`
            **grid_kwargs: keyword parameters used to construct the discretization grid

        """

        assert isinstance(gspace, GSpace)

        super(QuotientFourierPointwise, self).__init__()

        self.space = gspace

        G: Group = gspace.fibergroup

        self.rho = G.spectral_quotient_representation(subgroup_id, *irreps, name=None)

        # the representation in input is preserved
        self.in_type = self.out_type = FieldType(self.space, [self.rho] * channels)

        # retrieve the activation function to apply
        if function == "p_relu":
            self._function = F.relu_ if inplace else F.relu
        elif function == "p_elu":
            self._function = F.elu_ if inplace else F.elu
        elif function == "p_sigmoid":
            self._function = torch.sigmoid_ if inplace else F.sigmoid
        elif function == "p_tanh":
            self._function = torch.tanh_ if inplace else F.tanh
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))

        kernel = _build_kernel(G, subgroup_id, irreps)
        assert kernel.shape[0] == self.rho.size

        kernel = kernel / np.linalg.norm(kernel)
        kernel = kernel.reshape(-1, 1)

        if grid is None:
            grid = G.grid(*grid_args, **grid_kwargs)

        A = np.concatenate([self.rho(g) @ kernel for g in grid], axis=1).T

        eps = 1e-8
        Ainv = np.linalg.inv(A.T @ A + eps * np.eye(self.rho.size)) @ A.T

        self.A = torch.tensor(A, dtype=torch.get_default_dtype()).to(
            f"cuda:{torch.cuda.current_device()}"
        )
        self.Ainv = torch.tensor(Ainv, dtype=torch.get_default_dtype()).to(
            f"cuda:{torch.cuda.current_device()}"
        )

    def forward(self, input: GroupTensor) -> GroupTensor:
        r"""

        Applies the pointwise activation function on the input fields

        Args:
            input (GroupTensor): the input feature map

        Returns:
            the resulting feature map after the non-linearities have been applied

        """

        assert input.type == self.in_type

        shape = input.shape
        x_hat = input.tensor.view(
            shape[0], len(self.in_type), self.rho.size, *shape[2:]
        )

        x = torch.einsum("bcf...,gf->bcg...", x_hat, self.A)

        y = self._function(x)

        y_hat = torch.einsum("bcg...,fg->bcf...", y, self.Ainv)

        y_hat = y_hat.reshape(*shape)

        return GroupTensor(y_hat, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, atol: float = 1e-4, rtol: float = 2e-2
    ) -> List[Tuple[Any, float]]:
        c = self.in_type.size
        B = 64
        x = torch.randn(B, c, *[3] * self.space.dimensionality)

        errors = []

        # for el in self.space.testing_elements:
        for _ in range(50):
            el = self.space.fibergroup.sample()

            x1 = GroupTensor(x.clone(), self.in_type)
            x2 = GroupTensor(x.clone(), self.in_type).transform_fibers(el)

            out1 = self(x1).transform_fibers(el)
            out2 = self(x2)

            out1 = (
                out1.tensor.view(B, len(self.in_type), self.rho.size, *out1.shape[2:])
                .detach()
                .numpy()
            )
            out2 = (
                out2.tensor.view(B, len(self.in_type), self.rho.size, *out2.shape[2:])
                .detach()
                .numpy()
            )

            errs = np.linalg.norm(out1 - out2, axis=2).reshape(-1)
            errs[errs < atol] = 0.0
            norm = np.sqrt(
                np.linalg.norm(out1, axis=2).reshape(-1)
                * np.linalg.norm(out2, axis=2).reshape(-1)
            )

            relerr = errs / norm

            # print(el, errs.max(), errs.mean(), relerr.max(), relerr.min())

            assert (
                relerr.mean() + relerr.std() < rtol
            ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {}, std ={}'.format(
                el, relerr.max(), relerr.mean(), relerr.std()
            )

            errors.append((el, errs.mean()))

        return errors


class QuotientFourierELU(QuotientFourierPointwise):
    def __init__(
        self,
        gspace: GSpace,
        subgroup_id: Tuple,
        channels: int,
        irreps: List,
        *grid_args,
        grid: List[GroupElement] = None,
        inplace: bool = True,
        **grid_kwargs,
    ):
        r"""

        Applies a Inverse Fourier Transform to sample the input features on a quotient space, apply ELU point-wise
        in the spatial domain (Dirac-delta basis) and, finally, computes the Fourier Transform to obtain irreps
        coefficients. See :class:`~nn.QuotientFourierPointwise` for more details.

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            subgroup_id (tuple): identifier of the subgroup :math:`H` to construct the quotient space
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            *grid_args: parameters used to construct the discretization grid
            grid (list, optional): list containing the elements of the group to use for sampling. Optional (default ``None``).
            inplace (bool): applies the non-linear activation in-place. Default: `True`
            **grid_kwargs: keyword parameters used to construct the discretization grid

        """

        super(QuotientFourierELU, self).__init__(
            gspace,
            subgroup_id,
            channels,
            irreps,
            *grid_args,
            function="p_elu",
            inplace=inplace,
            grid=grid,
            **grid_kwargs,
        )
