from __future__ import annotations

from ....equivariant import group_theory
from .representation import Representation

from typing import Callable, Any, List, Union, Dict, Tuple

import numpy as np

__all__ = [
    "IrreducibleRepresentation",
]


class IrreducibleRepresentation(Representation):
    def __init__(
        self,
        group: group_theory.Group,
        id: Tuple,
        name: str,
        representation: Union[
            Dict[group_theory.GroupElement, np.ndarray], Callable[[Any], np.ndarray]
        ],
        size: int,
        type: str,
        supported_nonlinearities: List[str],
        character: Union[
            Dict[group_theory.GroupElement, float], Callable[[Any], float]
        ] = None,
        **kwargs,
    ):
        """
        Describes an "*irreducible representation*" (*irrep*).
        It is a subclass of a :class:`~Representation`.

        Irreducible representations are the building blocks into which any other representation decomposes under a
        change of basis.
        Indeed, any :class:`~Representation` is internally decomposed into a direct sum of irreps.

        Args:
            group (group_theory.Group): the group which is being represented
            id (tuple): args to generate this irrep using ``group.irrep(*id)``
            name (str): an identification name for this representation
            representation (dict or callable): a callable implementing this representation or a dict mapping
                    each group element to its representation.
            size (int): the size of the vector space where this representation is defined (i.e. the size of the matrices)
            type (str): type of the irrep. It needs to be one of `R`, `C` or `H`, which represent respectively
                        real, complex and quaternionic types.
                        NOTE: this parameter substitutes the old `sum_of_squares_constituents` from *e2cnn*.
            supported_nonlinearities (list): list of nonlinearitiy types supported by this representation.
            character (callable or dict, optional): a callable returning the character of this representation for an
                    input element or a dict mapping each group element to its character.
            **kwargs: custom attributes the user can set and, then, access from the dictionary
                    in :attr:`Representation.attributes`

        Attributes:
            ~.id (tuple): tuple which identifies this irrep; it can be used to generate this irrep as ``group.irrep(*id)``
            ~.sum_of_squares_constituents (int): the sum of the squares of the multiplicities of pairwise distinct
                    irreducible constituents of the character of this representation over a non-splitting field (see
                    `Character Orthogonality Theorem <https://groupprops.subwiki.org/wiki/Character_orthogonality_theorem#Statement_over_general_fields_in_terms_of_inner_product_of_class_functions>`_
                    over general fields).
                    This attribute is fully determined by the irrep's `type` as:

                    +----------+---------------------------------+
                    |  `type`  |  `sum_of_squares_constituents`  |
                    +==========+=================================+
                    |  'R'     |    `1`                          |
                    +----------+---------------------------------+
                    |  'C'     |    `2`                          |
                    +----------+---------------------------------+
                    |  'H'     |    `4`                          |
                    +----------+---------------------------------+

        """

        assert type in {"R", "C", "H"}

        if type == "C":
            assert size % 2 == 0
        elif type == "H":
            assert size % 4 == 0

        super(IrreducibleRepresentation, self).__init__(
            group,
            name,
            [id],
            np.eye(size),
            supported_nonlinearities,
            representation=representation,
            character=character,
            **kwargs,
        )
        assert isinstance(id, tuple)
        self.id: tuple = id
        self.irreducible = True
        self.type = type

        if self.type == "R":
            self.sum_of_squares_constituents = 1
        elif self.type == "C":
            self.sum_of_squares_constituents = 2
        elif self.type == "H":
            self.sum_of_squares_constituents = 4
        else:
            raise ValueError()

    def endomorphism_basis(self) -> np.ndarray:
        if self.type == "R":
            return np.eye(self.size).reshape(1, self.size, self.size)
        elif self.type == "C":
            basis = np.stack([np.eye(2), np.diag([1.0, -1.0])[::-1]], axis=0)
            return np.kron(basis, np.eye(self.size // 2))
        elif self.type == "H":
            basis = np.stack(
                [
                    np.eye(4),
                    np.diag([1.0, -1.0, 1.0, -1.0])[::-1],
                    np.array(
                        [
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                        ]
                    ),
                    np.array(
                        [
                            [0.0, -1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, -1.0, 0.0],
                        ]
                    ),
                ],
                axis=0,
            )
            return np.kron(basis, np.eye(self.size // 4))
        else:
            raise ValueError()
