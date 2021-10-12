"""The 2-body nuclear integrals."""

import itertools
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from qiskit_nature.properties.second_quantization.electronic.integrals import TwoBodyElectronicIntegrals, ElectronicIntegrals
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.operators.second_quantization import FermionicOp


class TwoBodyNuclearIntegrals(TwoBodyElectronicIntegrals):
    """The 2-body nuclear integrals."""

    EINSUM_AO_TO_MO = "pqrs,pi,qj,rk,sl->ijkl"
    EINSUM_CHEM_TO_PHYS = "ijkl->ljik"

    # TODO: provide symmetry testing functionality?

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
                must be a quartet of matrices, the first one being the alpha-alpha-spin matrix
                (which is required), followed by the beta-alpha-spin, beta-beta-spin, and
                alpha-beta-spin matrices (which are optional). The order of these matrices follows
                the standard assigned of quadrants in a plane geometry. If any of the latter three
                matrices are ``None``, the alpha-alpha-spin matrix will be used in their place.
                However, the final matrix will be replaced by the transpose of the second one, if
                and only if that happens to differ from ``None``.
            threshold: the truncation level below which to treat the integral in the SO matrix as
                zero-valued.
        """

        self._shift = shift

        super().__init__(basis, matrices, threshold)

    def to_spin(self) -> np.ndarray:
        """Transforms the integrals into the special ``ElectronicBasis.SO`` basis.

        Returns:
            A single matrix containing the ``n-body`` integrals in the spin orbital basis.
        """
        if self._basis == ElectronicBasis.SO:
            return self._matrices  # type: ignore

        so_matrix = np.zeros([2 * s for s in self._matrices[0].shape])
        one_indices = (
            (0, 0, 0, 0),  # alpha-alpha-spin
            (0, 1, 1, 0),  # beta-alpha-spin
            (1, 1, 1, 1),  # beta-beta-spin
            (1, 0, 0, 1),  # alpha-beta-spin
        )
        alpha_beta_spin_idx = 3
        for idx, (ao_mat, one_idx) in enumerate(zip(self._matrices, one_indices)):
            if ao_mat is None:
                if idx == alpha_beta_spin_idx:
                    ao_mat = self._matrices[0] if self._matrices[1] is None else self._matrices[1].T
                else:
                    ao_mat = self._matrices[0]
            phys_matrix = np.einsum(self.EINSUM_CHEM_TO_PHYS, ao_mat)
            kron = np.zeros((2, 2, 2, 2))
            kron[one_idx] = 1
            so_matrix -= 0.5 * np.kron(kron, phys_matrix)

        return np.where(np.abs(so_matrix) > self._threshold, so_matrix, 0.0)

    # works as good as lst method
    def toto_second_q_op(self) -> FermionicOp:
        """Creates the operator representing the Hamiltonian defined by these nuclear integrals.

        This method uses ``to_spin`` internally to map the nuclear integrals into the spin
        orbital basis.

        Returns:
            The ``FermionicOp`` given by these nuclear integrals.
        """
        spin_matrix = self.to_spin()
        register_length = len(spin_matrix)

        if not np.any(spin_matrix):
            return FermionicOp.zero(register_length)

        def transform_ind(i : int, j : int, k : int, l : int):
            return (i+self._shift, j+self._shift, k+self._shift, l+self._shift)

        return sum(  # type: ignore
            self._create_base_op(transform_ind(*indices), spin_matrix[indices], register_length + self._shift)
            for indices in itertools.product(
                range(register_length), repeat=2 * self._num_body_terms
            )
            if spin_matrix[indices]
        )

    def to_second_q_op(self) -> FermionicOp:
        """Creates the operator representing the Hamiltonian defined by these nuclear integrals.

        Returns:
            The ``FermionicOp`` given by these nuclear integrals.
        """
        matrix = 0.5*self._matrices[0]
        nn, _, _ , _ = self._matrices[0].shape
        register_length = nn*2

        if not np.any(matrix):
            return FermionicOp.zero(register_length)

        def transform_ind(i : int, j : int, k : int, l : int, s1 : int, s2 : int):
            return (i+self._shift+s1, j+self._shift+s1, k+self._shift+s2, l+self._shift+s2)

        return sum(  # type: ignore
            self._create_base_op(transform_ind(*indices,*spin), matrix[indices], register_length+self._shift)

            for spin in [(0,0),(0,nn),(nn,0),(nn,nn)]
            for indices in itertools.product(
                range(nn), repeat=2 * self._num_body_terms
            )
            if matrix[indices]
        )
