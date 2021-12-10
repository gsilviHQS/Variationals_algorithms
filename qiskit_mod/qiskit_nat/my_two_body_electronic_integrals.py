"""The 2-body electronic integrals for Total Nuclear Electronic problem."""

import itertools
from typing import List, Optional, Tuple, Union

import numpy as np

from qiskit_nature import QiskitNatureError

from qiskit_nature.properties.second_quantization.electronic.integrals import TwoBodyElectronicIntegrals
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.operators.second_quantization import FermionicOp


class MyTwoBodyElectronicIntegrals(TwoBodyElectronicIntegrals):
    """The 2-body electronic integrals for Total Nuclear Electronic problem."""

    def to_second_q_op(self) -> FermionicOp:
        """Creates the operator representing the Hamiltonian defined by these electronic integrals from GAMESS US.

        Returns:
            The ``FermionicOp`` given by these electronic integrals.
        """
        matrix = 0.5*self._matrices[0]
        ne, _, _ , _ = self._matrices[0].shape
        register_length = ne*2

        if not np.any(matrix):
            return FermionicOp.zero(register_length)

        def transform_ind(i : int, j : int, k : int, l : int, s1 : int, s2 : int):
            return (i+s1, j+s1, k+s2, l+s2)

        return sum(  # type: ignore
            self._create_base_op(transform_ind(*indices,*spin), matrix[indices], register_length)

            for spin in [(0,0),(0,ne),(ne,0),(ne,ne)]
            for indices in itertools.product(
                range(ne), repeat=2 * self._num_body_terms
            )
            if matrix[indices]
        )
