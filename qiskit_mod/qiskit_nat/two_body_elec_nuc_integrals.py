"""The 2-body electronic-nuclear integrals."""

import itertools
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from qiskit_nature.properties.second_quantization.electronic.integrals import TwoBodyElectronicIntegrals, ElectronicIntegrals
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.operators.second_quantization import FermionicOp


class TwoBodyElectronicNuclearIntegrals(TwoBodyElectronicIntegrals):
    """The 2-body electronic-nuclear integrals."""

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

    def _calc_coeffs_with_ops(self, indices: Tuple[int, ...]) -> List[Tuple[int, str]]:
        return [(indices[0], "+"), (indices[1], "-"), (indices[2], "+"), (indices[3], "-")]

    def to_second_q_op(self) -> FermionicOp:
        """Creates the operator representing the Hamiltonian defined by these electronic-nuclear integrals.

        Returns:
            The ``FermionicOp`` given by these electronic-nuclear integrals.
        """
        matrix = -1.*self._matrices[0]
        ne, _, nn, _ = self._matrices[0].shape
        register_length = (nn+ne)*2

        if not np.any(matrix):
            return FermionicOp.zero(register_length)

        def transform_ind(i : int, j : int, k : int, l : int, s1 : int, s2 : int):
            return (i+s1, j+s1, k+self._shift+s2, l+self._shift+s2)

        return sum(  # type: ignore
            self._create_base_op(transform_ind(*indices,*spin), matrix[indices], register_length)

            for spin in [(0,0),(0,nn),(ne,0),(ne,nn)]
            for indices in itertools.product(
                range(ne), repeat=2 * self._num_body_terms
            )
            if matrix[indices]
        )
