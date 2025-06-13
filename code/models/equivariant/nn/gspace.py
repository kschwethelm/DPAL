from __future__ import annotations

from ...equivariant import nn as enn
from ...equivariant.group_theory import (
    Group,
    GroupElement,
    Representation,
    IrreducibleRepresentation,
    KernelBasis,
)

from .utils import linear_transform_array_nd

from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Union

from collections import defaultdict

import numpy as np


__all__ = ["GSpace"]


class GSpace(ABC):
    def __init__(self, fibergroup: Group, dimensionality: int, name: str):
        r"""
        Abstract class for G-spaces.

        A ``GSpace`` describes the space where a signal lives (e.g. :math:`\R^2` for planar images) and its symmetries
        (e.g. rotations or reflections).
        As an `Euclidean` base space is assumed, a G-space is fully specified by the ``dimensionality`` of the space
        and a choice of origin-preserving symmetry group (``fibergroup``).

        .. seealso::

            :class:`~GSpace0D`,
            :class:`~GSpace2D`,
            or the factory methods
            :class:`~flipRot2dOnR2`,
            :class:`~rot2dOnR2`,
            :class:`~flip2dOnR2`,
            :class:`~trivialOnR2`

        .. note ::

            Mathematically, this class describes a *Principal Bundle*
            :math:`\pi : (\R^D, +) \rtimes G \to \mathbb{R}^D, tg \mapsto tG`,
            with the Euclidean space :math:`\mathbb{R}^D` (where :math:`D` is the ``dimensionality``) as `base space`
            and :math:`G` as `fiber group` (``fibergroup``).
            For more details on this interpretation we refer to
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.


        Args:
            fibergroup (Group): the fiber group
            dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            name (str): an identification name

        Attributes:
            ~.fibergroup (Group): the fiber group
            ~.dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            ~.name (str): an identification name
            ~.basespace (str): the name of the space whose symmetries are modeled. It is an Euclidean space :math:`\R^D`.

        """

        # TODO move this sub-package to PyTorch

        self.name = name
        self.dimensionality = dimensionality
        self.fibergroup = fibergroup
        self.basespace = f"R^{self.dimensionality}"

        # To not recompute the basis for the same intertwiner as many times as it appears,
        # the basis is stored in these dictionaries the first time we compute it

        # Store the computed intertwiners between general representations
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_repr, output_repr) pairs to the corresponding basis
        self._fields_intertwiners_basis_memory = defaultdict(dict)

        # Store the computed intertwiners between general representations
        # - key = (input_repr, output_repr)
        # - value = the corresponding basis
        self._fields_intertwiners_basis_memory_fiber_basis = dict()

    def type(self, *representations: Representation) -> enn.FieldType:
        f"""
        Shortcut to build a :class:`~equivariant.nn.FieldType`.
        This is equivalent to ```FieldType(gspace, representations)```.
        """
        return enn.FieldType(self, representations)

    @abstractmethod
    def restrict(self, id) -> Tuple[GSpace, Callable, Callable]:
        r"""

        Build the :class:`~GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``.
        This reduces the level of symmetries of the base space to be considered.

        Check the ``restrict`` method's documentation in the non-abstract subclass used for a description of the
        parameter ``id``.

        Args:
            id: id of the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)

        """
        pass

    @property
    @abstractmethod
    def basespace_action(self) -> Representation:
        r"""

        Defines how the fiber group transforms the base space.

        More precisely, this method defines how an element :math:`g \in G` of the fiber group transforms a point
        :math:`x \in X \cong \R^d` of the base space.
        This action is defined as a :math:`d`-dimensional linear :class:`~Representation` of :math:`G`.

        """
        pass

    def _interpolate_transform_basespace(
        self,
        input: np.ndarray,
        element: GroupElement,
        order: int = 2,
    ) -> np.ndarray:
        r"""

        Defines how the fiber group transforms the base space.

        The methods takes a tensor compatible with this space (i.e. whose spatial dimensions are supported by the
        base space) and returns the transformed tensor.

        More precisely, given an input tensor, interpreted as an :math:`n`-dimensional signal
        :math:`f: X \to \mathbb{R}^n` defined over the base space :math:`X`, and an element :math:`g \in G` of the
        fiber group, the methods return the transformed signal :math:`f'` defined as:

        .. math::
            f'(x) := f(g^{-1} x)

        This method is specific of the particular GSpace and defines how :math:`g^{-1}` transforms a point
        :math:`x \in X` of the base space.


        Args:
            input (~numpy.ndarray): input tensor
            element (GroupElement): element of the fiber group

        Returns:
            the transformed tensor

        """
        assert element.group == self.fibergroup
        action = self.basespace_action
        trafo = action(element)
        return linear_transform_array_nd(input, trafo, order=order)

    @property
    def irreps(self) -> List[IrreducibleRepresentation]:
        r"""
        list containing all the already built irreducible representations of the fiber group of this space.

        .. seealso::

            See :attr:`Group.irreps` for more details

        """
        return self.fibergroup.irreps()

    @property
    def representations(self):
        r"""
        Dictionary containing all the already built representations of the fiber group of this space.

        .. seealso::

            See :attr:`Group.representations` for more details

        """
        return self.fibergroup.representations

    @property
    def trivial_repr(self) -> Representation:
        r"""
        The trivial representation of the fiber group of this space.

        .. seealso::

            :attr:`Group.trivial_representation`

        """
        return self.fibergroup.trivial_representation

    def irrep(self, *id) -> IrreducibleRepresentation:
        r"""
        Builds the irreducible representation (:class:`~IrreducibleRepresentation`) of the fiber group
        identified by the input arguments.

        .. seealso::

            This method is a wrapper for :meth:`Group.irrep`. See its documentation for more details.
            Check the documentation of :meth:`~Group.irrep` of the specific fiber group used for more
            information on the valid ``id``.


        Args:
            *id: parameters identifying the irrep.

        """
        return self.fibergroup.irrep(*id)

    @property
    def regular_repr(self) -> Representation:
        r"""
        The regular representation of the fiber group of this space.

        .. seealso::

            :attr:`Group.regular_representation`

        """
        return self.fibergroup.regular_representation

    def induced_repr(self, subgroup_id, repr: Representation) -> Representation:
        r"""
        Builds the induced representation of the fiber group of this space from the representation ``repr`` of
        the subgroup identified by ``subgroup_id``.

        Check the :meth:`~GSpace.restrict` method's documentation in the non-abstract subclass used
        for a description of the parameter ``subgroup_id``.

        .. seealso::

            See :attr:`Group.induced_representation` for more details on the representation.

        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup to induce

        """
        return self.fibergroup.induced_representation(subgroup_id, repr)

    @property
    def testing_elements(self):
        return self.fibergroup.testing_elements()

    def build_kernel_basis(
        self,
        in_repr: Representation,
        out_repr: Representation,
        sigma: Union[float, List[float]],
        rings: List[float],
        **kwargs,
    ) -> KernelBasis:
        r"""
        Builds a basis for the space of the equivariant kernels with respect to the symmetries described by this
        :class:`~GSpace`.

        A kernel :math:`\kappa` equivariant to a group :math:`G` needs to satisfy the following equivariance constraint:

        .. math::
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1}  \qquad \forall g \in G, x \in \R^D

        where :math:`\rho_\text{in}` is ``in_repr`` while :math:`\rho_\text{out}` is ``out_repr``.


        Because the equivariance constraints only restrict the angular part of the kernels, any radial profile is
        permitted.
        The basis for the radial profile used here contains rings with different radii (``rings``)
        associated with (possibly different) widths (``sigma``).
        A ring is implemented as a Gaussian function over the radial component, centered at one radius
        (see also :class:`~GaussianRadialProfile`).

        .. note ::
            This method is a wrapper for the functions building the bases which are defined in :doc:`kernels`:
            - :meth:`kernel2d_o2`,
            - :meth:`kernel2d_so2`

        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            sigma (list or float): parameters controlling the width of each ring of the radial profile.
                    If only one scalar is passed, it is used for all rings
            rings (list): radii of the rings defining the radial profile
            **kwargs: Group-specific keywords arguments for ``_basis_generator`` method

        Returns:
            an instance of :class:`~KernelBasis` representing the analytical basis

        """
        assert isinstance(in_repr, Representation)
        assert isinstance(out_repr, Representation)

        assert in_repr.group == self.fibergroup
        assert out_repr.group == self.fibergroup

        if isinstance(sigma, float):
            sigma = [sigma] * len(rings)

        assert all([s > 0.0 for s in sigma])
        assert len(sigma) == len(rings)

        # build the key
        key = dict(**kwargs)
        key["sigma"] = tuple(sigma)
        key["rings"] = tuple(rings)
        key = tuple(sorted(key.items()))

        if (in_repr.name, out_repr.name) not in self._fields_intertwiners_basis_memory[
            key
        ]:
            # TODO - we could use a flag in the args to choose whether to store it or not

            basis = self._basis_generator(in_repr, out_repr, rings, sigma, **kwargs)

            # store the basis in the dictionary
            self._fields_intertwiners_basis_memory[key][
                (in_repr.name, out_repr.name)
            ] = basis

        # return the dictionary with all the basis built for this filter size
        return self._fields_intertwiners_basis_memory[key][
            (in_repr.name, out_repr.name)
        ]

    @abstractmethod
    def _basis_generator(
        self,
        in_repr: Representation,
        out_repr: Representation,
        rings: List[float],
        sigma: List[float],
        **kwargs,
    ):
        pass

    def __repr__(self):
        return self.name