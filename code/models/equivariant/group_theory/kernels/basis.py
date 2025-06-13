import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple


class EmptyBasisException(Exception):
    def __init__(self):
        r"""
        Exception raised when a :class:`~torch-ecnn.KernelBasis` with no elements is built.

        """
        message = "The KernelBasis you tried to instantiate is empty (dim = 0). You should catch this exception."
        super(EmptyBasisException, self).__init__(message)


class KernelBasis(torch.nn.Module, ABC):
    def __init__(self, dim: int, shape: Tuple[int, int]):
        r"""

        Abstract class for implementing the basis of a kernel space.
        A kernel space is the space of functions in the form:

        .. math::
            \mathcal{K} := \{ \kappa: X \to \mathbb{R}^{c_\text{out} \times c_\text{in}} \}

        where :math:`X` is the base space on which the kernel is defined.
        For instance, for planar images :math:`X = \R^2`.
        One can also access the dimensionality ``dim`` of this basis via the ``len()`` method.

        Args:
            dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`

        Attributes:
            ~.dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            ~.shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`

        """
        assert isinstance(dim, int), (dim, type(dim))
        assert isinstance(shape, tuple) and len(shape) == 2, shape

        assert dim >= 0

        if dim == 0:
            raise EmptyBasisException()

        self.dim = dim
        self.shape = shape

        super(KernelBasis, self).__init__()

    @abstractmethod
    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""
        Sample the continuous basis elements on discrete points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.
        ``points`` must be an array of shape `(N, D)`, where `D` is the dimensionality of the (parametrization of the)
        base space while `N` is the number of points.
        Args:
            points (~torch.Tensor): points where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output
        Returns:
            the sampled basis
        """
        pass

    def forward(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""
        Alias for :meth:`~torch-ecnn.KernelBasis.sample`.
        """
        return self.sample(points, out=out)

    def __len__(self):
        return self.dim

    def __iter__(self):
        for i in range(self.dim):
            yield self[i]

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class AdjointBasis(KernelBasis):
    def __init__(self, basis: KernelBasis, adjoint: np.ndarray):
        r"""
        Transform the input ``basis`` by applying a change of basis ``adjoint`` on the points before sampling the basis.
        Args:
            basis (KernelBasis): a kernel basis
            adjoint (~numpy.ndarray): an orthonormal matrix defining the change of basis on the base space
        """
        n = adjoint.shape[0]

        assert adjoint.shape == (n, n)
        assert np.allclose(
            adjoint @ adjoint.T, np.eye(n)
        ), "Error! The adjunction matrix must be orthonormal"
        assert np.allclose(
            adjoint.T @ adjoint, np.eye(n)
        ), "Error! The adjunction matrix must be orthonormal"

        super(AdjointBasis, self).__init__(basis.dim, basis.shape)

        self.basis = basis

        self.register_buffer("adj", torch.tensor(adjoint, dtype=torch.float32))

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""
        Sample the continuous basis elements on the discrete set of points.
        Optionally, store the resulting multidimentional array in ``out``.
        ``points`` must be an array of shape `(N, d)`, where `N` is the number of points and `d` their
        dimensionality.
        Args:
            points (~numpy.ndarray): points where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output
        Returns:
            the sampled basis
        """
        assert len(points.shape) == 2
        assert points.shape[1] == self.adj.shape[0]

        transformed_points = (
            points @ self.adj.to(device=points.device, dtype=points.dtype).T
        )
        return self.basis.sample(transformed_points, out)

    def __getitem__(self, r):
        return self.basis[r]

    def __eq__(self, other):
        if isinstance(other, AdjointBasis):
            return self.basis == other.basis and torch.allclose(self.adj, other.adj)
        else:
            return False

    def __hash__(self):
        return hash(self.adj) + 1000 * hash(self.basis)
