########################################################################################################################
# Utils methods for decomposing or composing representations ###########################################################
########################################################################################################################

from __future__ import annotations

from ....equivariant import group_theory
from ....equivariant.group_theory.groups import Group

from typing import List, Tuple, Union

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import scipy.sparse.linalg as slinalg

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None


########################################################################################################################
# Numerical utilities
########################################################################################################################


def null(
    A: Union[np.matrix, sparse.linalg.LinearOperator],
    use_sparse: bool,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute a basis for the Kernel space of the matrix A.

    If ``use_sparse`` is ``True``, :meth:`scipy.sparse.linalg.svds` is used;
    otherwise, :meth:`scipy.linalg.svd` is used.

    Moreover, if the input is a sparse matrix, ``use_sparse`` has to be set to ``True``.

    Args:
        A: input matrix
        use_sparse: whether to use spare methods or not
        eps: threshold to consider a value zero. The default value is ``1e-12``

    Returns:
        A matrix whose columns are a basis of the kernel space

    """
    if use_sparse:
        k = min(A.shape) - 1
        u, s, vh = slinalg.svds(A, k=k)
    else:
        if randomized_svd is not None:
            k = min(A.shape)
            u, s, vh = randomized_svd(np.asarray(A), n_components=k)
        else:
            u, s, vh = linalg.svd(A, full_matrices=False)

    null_space = np.compress((s <= eps), vh, axis=0)
    return np.transpose(null_space)


def build_sylvester_constraint(
    rho_1: List[np.ndarray], rho_2: List[np.ndarray]
) -> sparse.linalg.LinearOperator:
    assert len(rho_1) == len(rho_2)
    assert len(rho_1) > 0

    d1 = rho_1[0].shape[0]
    d2 = rho_2[0].shape[0]

    constraints = []
    for rho_1_g, rho_2_g in zip(rho_1, rho_2):
        assert rho_1_g.shape == (d1, d1)
        assert rho_2_g.shape == (d2, d2)

        # build the linear system corresponding to the Sylvester Equation with the current group element
        constraint = sparse.kronsum(rho_1_g, -rho_2_g.T, format="csc")
        constraints.append(constraint)

    # stack all equations in one unique matrix
    return sparse.vstack(constraints, format="csc")  # .todense()


def find_intertwiner_basis_sylvester(
    rho_1: List[np.ndarray], rho_2: List[np.ndarray], eps: float = 1e-12
) -> np.ndarray:
    constraint = build_sylvester_constraint(rho_1, rho_2)
    # Kernel space of this matrix contains the solutions of our problem
    if constraint.shape[1] == 1:
        if np.count_nonzero(constraint.todense()) == 0:
            return np.ones([1, 1])
        else:
            return np.zeros((1, 0))
    else:
        # Compute the basis of the kernel.
        # The sparse method can not compute the eigenspace associated with the smallest eigenvalue,
        # which is a problem when the null space is one dimensional.
        basis = null(constraint.todense(), False, eps=eps)

        assert np.allclose(constraint @ basis, 0.0)

        return basis


########################################################################################################################
# Numeric methods for irrep decomposition for GENERAL GROUPS
########################################################################################################################


class InsufficientIrrepsException(Exception):
    def __init__(self, G: Group, message: str = None):
        self.G = G

        if message is None:
            from textwrap import dedent

            message = dedent(
                f"""
                Error! Did not find sufficient irreps to complete the decomposition of the input representation.
                It is likely this happened because not sufficiently many irreps in '{G}' have been instantiated.
                Try instantiating more irreps and then repeat this call.
            """
            )
        super(InsufficientIrrepsException, self).__init__(message)


def find_tensor_decomposition(J: Tuple, l: Tuple, G: Group) -> List[Tuple[Tuple, int]]:
    """Check if subgroup irreps are part of group?"""
    psi_J = G.irrep(*J)
    psi_l = G.irrep(*l)

    irreps = []

    size = 0
    for psi_j in G.irreps():
        CG = G.clebsch_gordan_coeff(psi_J, psi_l, psi_j)

        S = CG.shape[-2]

        if S > 0:
            irreps.append((psi_j.id, S))

        size += psi_j.size * S

    # check that size == psi_J.size * psi_l.size
    if size < psi_J.size * psi_l.size:
        from textwrap import dedent

        message = dedent(
            f"""
            Error! Did not find sufficient irreps to complete the decomposition of the tensor product of '{psi_J.name}' and '{psi_l.name}'.
            It is likely this happened because not sufficiently many irreps in '{G}' have been instantiated.
            Try instantiating more irreps and then repeat this call.
            The sum of the sizes of the irreps found is {size}, but the representation has size {psi_J.size * psi_l.size}.
        """
        )
        raise InsufficientIrrepsException(G, message)

    assert (
        size <= psi_J.size * psi_l.size
    ), f"""
        Error! Found too many irreps in the the decomposition of the tensor product of '{psi_J.name}' and '{psi_l.name}'.
        This should never happen!
    """

    return irreps


########################################################################################################################
# Numeric solutions for Clebsch-Gordan coefficients
########################################################################################################################


class UnderconstrainedCGSystem(Exception):
    def __init__(
        self,
        G: group_theory.Group,
        J: Tuple,
        l: Tuple,
        j: Tuple,
        S: int,
        message: str = "The algorithm to compute the CG coefficients failed due to an unsufficient number of samples to constraint the problem",
    ):
        self.G = G
        self.J = J
        self.l = l
        self.j = j
        self.S = S
        super(UnderconstrainedCGSystem, self).__init__(message)


def clebsch_gordan_tensor(J: Tuple, l: Tuple, j: Tuple, G: Group) -> np.ndarray:
    psi_J = G.irrep(*J)
    psi_l = G.irrep(*l)
    psi_j = G.irrep(*j)

    D = psi_J.size * psi_l.size * psi_j.size

    def build_matrices(samples):
        D_Jl = []
        D_j = []
        for g in samples:
            D_J_g = psi_J(g)
            D_l_g = psi_l(g)
            D_j_g = psi_j(g)

            D_Jl_g = np.kron(D_J_g, D_l_g)

            D_j.append(D_j_g)
            D_Jl.append(D_Jl_g)
        return D_Jl, D_j

    try:
        generators = G.generators
        S = len(generators)
    except ValueError:
        generators = []
        # number of samples to use to approximate the solutions usually 3 are sufficient
        S = 3

    while True:
        # sometimes it might not converge, so we need to try a few times
        attepts = 5
        while True:
            try:
                samples = generators + [G.sample() for _ in range(S - len(generators))]
                if len(samples) == 0:
                    basis = np.eye(D)
                else:
                    D_Jl, D_j = build_matrices(samples)
                    basis = find_intertwiner_basis_sylvester(D_Jl, D_j)

            except np.linalg.LinAlgError:
                if attepts > 0:
                    attepts -= 1
                    continue
                else:
                    raise
            else:
                break

        # check that the solutions found are also in the kernel of the constraint matrix built with other random samples
        D_Jl, D_j = build_matrices(generators + [G.sample() for _ in range(20)])
        tensor = build_sylvester_constraint(D_Jl, D_j).todense().reshape(-1, D)

        if np.allclose(tensor @ basis, 0.0):
            break
        elif S < 20:  # 20 is number of max samples
            # if this not the case, try again using more samples to build the constraint matrix
            S += 1
        else:
            raise UnderconstrainedCGSystem(G, psi_J.id, psi_l.id, psi_j.id, S)

    # the dimensionality of this basis corresponds to the multiplicity of `j` in the tensor-product `J x l`
    s = basis.shape[1]
    assert s % psi_j.sum_of_squares_constituents == 0

    jJl = s // psi_j.sum_of_squares_constituents

    # CG indexed as [J, l, s, j]
    CG = basis.reshape((psi_j.size, psi_J.size, psi_l.size, s)).transpose(1, 2, 3, 0)

    if s == 0:
        return CG

    norm = np.sqrt(
        (CG**2).mean(2, keepdims=True).sum(1, keepdims=True).sum(0, keepdims=True)
    )
    CG /= norm

    ortho = np.einsum("Jlsj,Jlti,kji->stk", CG, CG, psi_j.endomorphism_basis())

    ortho = (ortho**2).sum(2) > 1e-9
    assert ortho.astype(np.uint).sum() == s * psi_j.sum_of_squares_constituents, (
        ortho,
        s,
        jJl,
        psi_j.sum_of_squares_constituents,
    )

    n, dependencies = connected_components(
        csgraph=csr_matrix(ortho), directed=False, return_labels=True
    )
    assert n * psi_j.sum_of_squares_constituents == s, (
        ortho,
        n,
        s,
        psi_j.sum_of_squares_constituents,
    )

    mask = np.zeros((ortho.shape[0]), dtype=bool)
    for i in range(n):
        columns = np.nonzero(dependencies == i)[0]
        assert len(columns) == psi_j.sum_of_squares_constituents
        selected_column = columns[0]
        mask[selected_column] = 1

    assert mask.sum() == n

    CG = CG[..., mask, :]

    assert CG.shape[-2] == jJl

    B = CG.reshape(-1, psi_j.size * jJl)
    assert np.allclose(B.T @ B, np.eye(psi_j.size * jJl))

    return CG
