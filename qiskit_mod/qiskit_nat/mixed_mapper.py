"""The Mixed Mapper. """

import numpy as np
from typing import List, Tuple

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli

from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_nature.mappers.second_quantization.fermionic_mapper import FermionicMapper
from qiskit_nature.mappers.second_quantization.qubit_mapper import QubitMapper


class MixedMapper(QubitMapper):  # pylint: disable=missing-class-docstring
    def __init__(self):
        """The Mixed fermion-boson-to-qubit mapping."""
        super().__init__(allows_two_qubit_reduction=False)

    def map(self, second_q_op: FermionicOp, signature: List[Tuple[List[int],bool]]) -> PauliSumOp:

        # number of modes/sites for the Jordan-Wigner transform (= number of fermionic modes)
        nmodes = second_q_op.register_length
        print(nmodes)

        def direct(i: int, p: int):
            a_z = np.asarray([0] * p + [0] + [0] * (nmodes - p - 1), dtype=bool)
            a_x = np.asarray([0] * p + [1] + [0] * (nmodes - p - 1), dtype=bool)
            b_z = np.asarray([0] * p + [1] + [0] * (nmodes - p - 1), dtype=bool)
            b_x = np.asarray([0] * p + [1] + [0] * (nmodes - p - 1), dtype=bool)

            return (Pauli((a_z, a_x)), Pauli((b_z, b_x))) 

        def jordan(i: int, p: int):
            a_z = np.asarray([0] * (p-i) + [1]*i + [0] + [0] * (nmodes - p - 1), dtype=bool)
            a_x = np.asarray([0] * p + [1] + [0] * (nmodes - p - 1), dtype=bool)
            b_z = np.asarray([0] * (p-i) + [1] * i + [1] + [0] * (nmodes - p - 1), dtype=bool)
            b_x = np.asarray([0] * p + [1] + [0] * (nmodes - p - 1), dtype=bool)

            return (Pauli((a_z, a_x)), Pauli((b_z, b_x))) 
       
        pauli_table = []
        for pos, bos in signature:
            
            start = pos[0] 
            give_pauli = direct if bos else jordan

            for i, p in enumerate(pos):
                pauli_table.append(give_pauli(i,p))
                # TODO add Pauli 3-tuple to lookup table

        return QubitMapper.mode_based_mapping(second_q_op, pauli_table)
