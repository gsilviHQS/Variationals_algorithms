"""The 1-body nuclear integrals."""

import itertools
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from qiskit_nature.properties.second_quantization.electronic.integrals import OneBodyElectronicIntegrals, ElectronicIntegrals
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.operators.second_quantization import FermionicOp


class OneBodyNuclearIntegrals(OneBodyElectronicIntegrals):
    """The 1-body electronic integrals."""

    def __init__(
        self,
        basis: ElectronicBasis,
        matrices: Union[np.ndarray, Tuple[Optional[np.ndarray], ...]],
        threshold: float = ElectronicIntegrals.INTEGRAL_TRUNCATION_LEVEL,
        shift: int = 0,
    ):
        """
        Args:
            basis: the basis which these integrals are stored in. If this is initialized with
                ``ElectronicBasis.SO``, these integrals will be used *ad verbatim* during the
                mapping to a ``SecondQuantizedOp``.
            matrices: the matrices (one or many) storing the actual electronic integrals. If this is
                a single matrix, ``basis`` must be set to ``ElectronicBasis.SO``. Otherwise, this
                must be a pair of matrices, the first one being the alpha-spin matrix (which is
                required) and the second one being an optional beta-spin matrix. If the latter is
                ``None``, the alpha-spin matrix is used in its place.
            threshold: the truncation level below which to treat the integral in the SO matrix as
                zero-valued.
        """

        self._shift = shift

        super().__init__(basis, matrices, threshold)

    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special ``ElectronicBasis.SO`` basis.

        In this case of the 1-body integrals, the returned matrix is a block matrix of the form:
        ``[[alpha_spin, zeros], [zeros, beta_spin]]``.

        Returns:
            A single matrix containing the ``n-body`` integrals in the spin orbital basis.
        """
        if self._basis == ElectronicBasis.SO:
            return self._matrices  # type: ignore

        matrix_a = self._matrices[0]
        matrix_b = matrix_a if self._matrices[1] is None else self._matrices[1]
        zeros = np.zeros(matrix_a.shape)
        so_matrix = np.block([[matrix_a, zeros], [zeros, matrix_b]])

        return np.where(np.abs(so_matrix) > self._threshold, so_matrix, 0.0)

    def _calc_coeffs_with_ops(self, indices: Tuple[int, ...]) -> List[Tuple[int, str]]:
        return [(indices[0], "+"), (indices[1], "-")]

    def to_second_q_op(self) -> FermionicOp:
        """Creates the operator representing the Hamiltonian defined by these electronic integrals.

        This method uses ``to_spin`` internally to map the electronic integrals into the spin
        orbital basis.

        Returns:
            The ``FermionicOp`` given by these electronic integrals.
        """
        spin_matrix = self.to_spin()
        register_length = len(spin_matrix)

        if not np.any(spin_matrix):
            return FermionicOp.zero(register_length)

        # A. Kovyrshin
        # register_length is shifted for a specific case of two 
        # different fermions i and j are incremented by register_length
        def transform_ind(i : int, j : int):
            return (i+self._shift,j+self._shift)

        return sum(  # type: ignore
            self._create_base_op(transform_ind(*indices), spin_matrix[indices], register_length + self._shift)
            for indices in itertools.product(
                range(register_length), repeat=2 * self._num_body_terms
            )
            if spin_matrix[indices]
        )

