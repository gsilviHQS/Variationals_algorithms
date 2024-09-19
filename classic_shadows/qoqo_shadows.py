# qoqo_shadows is a Python library for creating quantum shadow circuits.
from qoqo import operations as ops
from qoqo import Circuit, QuantumProgram
from qoqo_quest import Backend

import numpy as np
from typing import List, Optional, Dict, Any
import itertools
from functools import reduce

from typing import List, Optional
import numpy as np

XYZ_MAP = {
    "X": [lambda qubit: ops.RotateY(qubit, theta=np.pi / 2)],
    "Y": [lambda qubit: ops.RotateX(qubit, theta=-np.pi / 2)],
    "Z": None,
}

paulis = {
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


def get_unitary_matrix_from_ops(unitary_ops: Optional[List]) -> np.ndarray:
    """Given a list of operations, get the corresponding unitary matrix.
    If None, return the identity matrix (case of Z basis).

    Parameters:
        unitary_ops (Optional[List]): A list of unitary operations.

    Returns:
        unitary_matrix (np.ndarray): The corresponding unitary matrix.
    """
    if unitary_ops is None:
        return np.eye(2)

    unitary_ops_mats = [ops(qubit=0).unitary_matrix() for ops in unitary_ops]
    return reduce(np.dot, unitary_ops_mats)


XYZ_MAP_MATRICES = {
    key: get_unitary_matrix_from_ops(ops) for key, ops in XYZ_MAP.items()
}


def create_circuit_for_shadow_measure(
    num_qubits: int,
    n_measurement: int,
    map_for_ops: dict = {},
    base_label: str = "ro",
) -> Circuit:
    """Creates a quantum circuit for shadow measurements.

    Iterate over the map_for_ops and add the corresponding operations to the circuit.
    Important: the operations are reversed, which is relevant for multiple operation cases.

    Parameters:
    - num_qubits (int): The total number of qubits in the circuit.
    - initialization_circ (Circuit): The initial part of the circuit.
    - n_measurement (int): Number of times measurements are to be repeated.
    - map_for_ops (dict): Mapping of qubit indices to operations.

    Returns:
    - Circuit: Complete quantum circuit with shadow measurements.
    """
    shadow_circuit = Circuit()

    for qb, operations in map_for_ops.items():
        if operations:
            # Important! the reverse order is for ops, not matrices
            for op in reversed(operations):
                shadow_circuit += op(qubit=qb)

    shadow_circuit += ops.DefinitionBit(
        name=base_label, length=len(map_for_ops), is_output=True
    )
    for qb in range(num_qubits):
        shadow_circuit += ops.MeasureQubit(qb, base_label, qb)
    shadow_circuit += ops.PragmaSetNumberOfMeasurements(n_measurement, base_label)

    return shadow_circuit


def create_XYZ_shadow_circuit(
    num_qubits: int,
    n_measurement: int = 1,
    map_for_unitaries: dict = XYZ_MAP,
    max_circuits: Optional[int] = None,
) -> list:
    """Generates circuits for shadow measurements based on specified measurement bases for each qubit.

    Each circuit corresponds to a different combination of measurement bases (X, Y, Z, etc.),
    applied across all qubits. This function iterates over all possible combinations of measurement bases,
    constructs the circuit for each combination by applying the corresponding unitary transformations,
    and then stores the circuit along with its unitary matrices.

    Parameters:
    - num_qubits (int): The number of qubits in the circuit.
    - n_measurement (int, optional): The number of measurements to perform. Defaults to 1.
    - map_for_unitaries (dict, optional): A mapping from measurement basis labels (e.g., 'X', 'Y', 'Z') to the
      corresponding quantum operations (unitaries). Defaults to XYZ_MAP, a predefined dictionary.
    - max_circuits (int, optional): The maximum number of circuits to generate. Defaults to None, which means all combinations possible.

    Returns:
    - dict: A dictionary where keys are strings representing the combination of measurement bases applied to
      each qubit, and values are tuples containing the corresponding quantum circuit and a dictionary of unitary
      matrices for each qubit.
    """
    all_combinations = get_combinations(
        num_qubits, max_circuits, list(map_for_unitaries.keys())
    )
    num_circuits = len(all_combinations)
    if num_circuits > n_measurement:
        raise ValueError(
            "Not enough measurement shots to generate all circuits. Either increase the number of measurement shots or limit the number of circuits."
        )
    n_measurement_per_circuit = int(n_measurement / num_circuits)
    print(
        f"Created {num_circuits} measurement circuits, each with {n_measurement_per_circuit} shots"
    )
    all_circuits = []

    # Generate all combinations of measurement bases for n qubits
    for combination in all_combinations:
        map_for_ops = {}

        # Construct the list of unitary operations for the current combination
        for qubit_idx, base in enumerate(combination):
            # Get the operations to apply to the qubit
            ops = map_for_unitaries.get(base, None)
            # Update ops to include the qubit index, and append to unitary_operations
            map_for_ops[qubit_idx] = ops

        base_label = "".join(combination)  # Label for the measurement basis combination
        # Generate the corresponding circuit for shadow measurement
        circuit = create_circuit_for_shadow_measure(
            num_qubits, n_measurement_per_circuit, map_for_ops, base_label
        )

        all_circuits.append(circuit)

    return all_circuits


def get_combinations(
    num_qubits: int,
    max_combination: Optional[int] = None,
    measurement_bases: list = ["X", "Y", "Z"],
) -> List[List[str]]:
    """Generate all possible combinations of measurement bases for a given number of qubits.

    Parameters:
    - num_qubits (int): The number of qubits to measure.
    - max_combination (Optional[int]): The maximum number of combinations to generate.
    If None, all possible combinations are generated.
    - measurement_bases (list):A list of measurement bases to consider.Defaults to ['X', 'Y', 'Z'].

    Returns:
    List[List[str]]: A list of lists, where each inner list represents a combination of bases.
    """

    all_combinations = list(itertools.product(measurement_bases, repeat=num_qubits))
    if max_combination is not None and len(all_combinations) > max_combination:
        indices = np.random.choice(
            len(all_combinations), size=max_combination, replace=False
        )
        all_combinations = [all_combinations[i] for i in indices]
    return all_combinations


def inverse_channel(
    post_measurement_state: np.ndarray, n_qubits: int = 1
) -> np.ndarray:
    """Applies the inverse channel operation to a post-measurement state to estimate the pre-measurement state.

    This is a part of classical post-processing in quantum shadow tomography, where the aim is to reconstruct
    the quantum state before measurement based on the measurement outcomes.

    Parameters:
    - post_measurement_state (np.ndarray): The density matrix of the post-measurement state.
    - n_qubits (int, optional): The number of qubits in the state. This affects the scaling factor
      used in the inverse channel calculation. Defaults to 1.

    Returns:
    - np.ndarray: The estimated pre-measurement state as a density matrix, obtained by applying the
      inverse channel operation to the post-measurement state.
    """
    return (2**n_qubits + 1) * post_measurement_state - np.eye(2**n_qubits)


def get_shadow_from_outcome(
    measurement_outcome: bool, unitary: np.ndarray, n_qubits=1
) -> np.ndarray:
    """Constructs the shadow state from a single measurement outcome and a unitary matrix.

    This function first constructs a qubit state based on the measurement outcome, then applies the conjugate transpose
    (dagger) of the unitary to this qubit state to get the shadow state. It then constructs the density matrix
    for this shadow state and applies the inverse channel to this density matrix to obtain the final shadow
    state in density matrix form.

    Parameters:
    - measurement_outcome (bool): The outcome of the measurement, where False represents the state |0> and True represents the state |1>.
    - unitary (np.ndarray): The unitary matrix that was applied to the qubit before measurement.
    - n_qubits (int, optional): The number of qubits in the state. Defaults to 1.

    Returns:
    - np.ndarray: The density matrix representing the shadow state of the qubit post-measurement.
    """
    qubit_state = np.array([[0], [1]]) if measurement_outcome else np.array([[1], [0]])
    # Apply the dagger of the unitary to the qubit state
    qubit_shadow = unitary.conj().T @ qubit_state
    # Construct the density matrix for the qubit
    qubit_density_matrix = np.outer(qubit_shadow, qubit_shadow.conj())
    # Append the individual qubit density matrix to the list
    return inverse_channel(qubit_density_matrix, n_qubits)


def measure_and_process_circuit(
    trotterstep: int,
    quantum_program: QuantumProgram,
    backend: Backend,
    verbose: bool = False,
) -> dict:
    """Measure and process classical shadow circuts.

    Measures and processes a dictionary of quantum circuits with their corresponding unitary matrices,
    using a specified quantum backend. This function iterates over each circuit, performs the measurement
    using the backend, and then constructs the post-measurement state. For each measurement outcome,
    it calculates the shadow state by applying the inverse channel and the conjugate transpose (dagger)
    of the unitary. It then constructs the multi-qubit shadow state from individual qubit shadows.

    Parameters:
    - trotterstep (int): Integer index of the Trotter steps to apply.
    - quantum_program (QuantumProgram): The quantum program to execute.
    - backend (Backend): The quantum backend to run the circuits on.
    - verbose (bool, optional): If True, prints additional information during execution. Defaults to False.

    Returns:
    - list: A list of snapshot from the classical shadow measurement.
    """
    snapshots = []

    # Run the quantum program on the backend
    (multi_bit_registers, _, _) = quantum_program.run_registers(backend, [trotterstep])

    # Process the measurement outcomes to obtain the multi-qubit shadow states
    for base_label, bit_registers in multi_bit_registers.items():
        if verbose:
            print(
                "Processing: Circuit with unitaries",
                base_label,
                "\nMeasurement:",
                len(bit_registers),
            )

        # Construct the post-measurement state in the computational basis |0> or |1> ket
        for measurement_outcome in bit_registers:
            shadows_per_qb = []
            for qb_idx, single_outcome in enumerate(measurement_outcome):
                # Obtain the shadow from the outcome
                base_measured = base_label[qb_idx]
                unitary_matrix = XYZ_MAP_MATRICES[base_measured]
                shadow = get_shadow_from_outcome(single_outcome, unitary_matrix)

                # Insert in front to match little-endian convention
                shadows_per_qb.insert(0, shadow)

            # Tensor product to build up the multi-qubit shadow from individual qubit shadows
            multi_qubit_shadow = reduce(np.kron, shadows_per_qb)
            snapshots.append(multi_qubit_shadow)

    return snapshots


def generate_operators(num_qubits: int, locality: int) -> Dict[str, Any]:
    """Generate operators with a specified locality that include at least 'locality' number of non-identity Pauli operators.

    Each generated operator is placed in all possible positions across the qubits, with the rest filled with identity operators.

    Parameters:
    - num_qubits (int): The total number of qubits in the system.
    - locality (int): The number of non-identity Pauli operators that each generated operator must include.
    - allow_periodicity (bool): If True, allows operators to wrap around the qubit array, enabling periodic boundary conditions.


    Returns:
    - Dict[str, np.ndarray]: A dictionary where keys are labels representing the operators (e.g., 'XXIY')
      and values are the corresponding numpy arrays representing the operators.
    """
    assert (
        1 <= locality <= num_qubits
    ), "Locality must be between 1 and the number of qubits"

    combinations = itertools.product(paulis.keys(), repeat=locality)

    position_combinations = list(itertools.combinations(range(num_qubits), locality))

    operators = {}
    for combo in combinations:
        for positions in position_combinations:
            operator_label = [
                "I"
            ] * num_qubits  # Initialize operator label with all 'I's
            # Place each operator in its respective position
            for pos, op in zip(positions, combo):
                operator_label[pos] = op

            # Construct the operator based on the label
            operator = np.eye(1)  # Start with a scalar for Kronecker product
            for label in operator_label:
                operator = np.kron(operator, paulis.get(label, np.eye(2)))

            operators["".join(operator_label)] = operator

    return operators
