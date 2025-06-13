from . import *
from ....equivariant.group_theory.groups import SO2, O2
from ....equivariant.group_theory.representations import Representation

import numpy as np

from typing import List, Union, Callable, Dict

__all__ = [
    "kernel2d_so2_subgroup",
    "kernel2d_o2_subgroup",
    "kernel2d_so2",
    "kernel2d_o2",
]


def kernel2d_so2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    filter: Callable[[Dict], bool] = None,
) -> KernelBasis:
    r"""
    Builds a basis for convolutional kernels equivariant to continuous rotations, modeled by the
    group :math:`SO(2)`.
    ``in_repr`` and ``out_repr`` need to be :class:`~Representation` s of :class:`~group.SO2`.
    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~GaussianRadialProfile`).
    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
    """
    assert in_repr.group == out_repr.group

    group = in_repr.group

    assert isinstance(group, SO2)

    radial_profile = GaussianRadialProfile(radii, sigma)

    if maximum_frequency is None:
        max_in_freq = max(freq for freq, in in_repr.irreps)
        max_out_freq = max(freq for freq, in out_repr.irreps)
        maximum_frequency = max_in_freq + max_out_freq

    return SteerableKernelBasis(
        CircularShellsBasis(maximum_frequency, radial_profile, filter=filter),
        in_repr,
        out_repr,
        RestrictedWignerEckartBasis,
        sg_id=(None, -1),
    )


def kernel2d_o2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    axis: float = np.pi / 2,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
) -> KernelBasis:
    r"""
    Builds a basis for convolutional kernels equivariant to reflections and continuous rotations, modeled by the
    group :math:`O(2)`.
    ``in_repr`` and ``out_repr`` need to be :class:`~Representation` s of :class:`~group.O2`.
    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~GaussianRadialProfile`).
    Because :math:`O(2)` contains all rotations, the reflection element of the group can be associated to any reflection
    axis. Reflections along other axes can be obtained by composition with rotations.
    However, a choice of this axis is required to align the basis with respect to the action of the group.
    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        axis (float, optional): angle of the axis of the reflection element
        adjoint (~numpy.ndarray, optional): instead of specifying a reflection axis, you can pass a 2x2 orthogonal
            matrix defining a change of basis on the base space
    """
    assert in_repr.group == out_repr.group

    group = in_repr.group
    assert isinstance(group, O2)

    radial_profile = GaussianRadialProfile(radii, sigma)

    if maximum_frequency is None:
        max_in_freq = max(freq for _, freq in in_repr.irreps)
        max_out_freq = max(freq for _, freq in out_repr.irreps)
        maximum_frequency = max_in_freq + max_out_freq

    basis = SteerableKernelBasis(
        CircularShellsBasis(
            maximum_frequency, radial_profile, filter=filter, axis=axis
        ),
        in_repr,
        out_repr,
        WignerEckartBasis,
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (2, 2)
        basis = AdjointBasis(basis, adjoint)

    return basis


###### Automatic subgroups kernel bases
def kernel2d_o2_subgroup(
    in_repr: Representation,
    out_repr: Representation,
    sg_id,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = 5,
    axis: float = np.pi / 2.0,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
) -> KernelBasis:
    o2 = O2(maximum_frequency)

    group, _, _ = o2.subgroup(sg_id)
    assert in_repr.group == group
    assert out_repr.group == group

    radial_profile = GaussianRadialProfile(radii, sigma)

    basis = SteerableKernelBasis(
        CircularShellsBasis(
            maximum_frequency, radial_profile, filter=filter, axis=axis
        ),
        in_repr,
        out_repr,
        RestrictedWignerEckartBasis,
        sg_id=sg_id,
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (2, 2)
        basis = AdjointBasis(basis, adjoint)

    return basis


def kernel2d_so2_subgroup(
    in_repr: Representation,
    out_repr: Representation,
    sg_id,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = 5,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
) -> KernelBasis:
    so2 = SO2(maximum_frequency)

    group, _, _ = so2.subgroup(sg_id)
    assert in_repr.group == group
    assert out_repr.group == group

    o2 = O2(maximum_frequency)
    sg_id = o2._combine_subgroups((None, -1), sg_id)

    radial_profile = GaussianRadialProfile(radii, sigma)

    basis = SteerableKernelBasis(
        CircularShellsBasis(maximum_frequency, radial_profile, filter=filter),
        in_repr,
        out_repr,
        RestrictedWignerEckartBasis,
        sg_id=sg_id,
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (2, 2)
        basis = AdjointBasis(basis, adjoint)

    return basis
