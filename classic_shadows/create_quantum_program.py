import json
from hqs_noise_app import HqsNoiseApp
from struqture_py.spins import SpinHamiltonianSystem
from qoqo import QuantumProgram, measurements
from qoqo.noise_models import ContinuousDecoherenceModel
from qoqo.devices import AllToAllDevice
from qoqo_shadows import create_XYZ_shadow_circuit


# Configuring the parameters
num_qubits = 3
noise_mode = "all_qubits"
trotter_timestep = 10
shot_per_trotter_step = 100
initialisation = [0, 0, 1]  # Initialization of spins

filename = "quantum_program.json"

# Hamiltonian values:
epsilon = -0.582863
J = 0.148819
K = 0.009731
J_S = 0.019255


# Device
device = AllToAllDevice(
    num_qubits,
    [
        "RotateX",
        "RotateZ",
        "RotateY",
    ],
    ["CNOT"],
    1.0,
)

# Noise model. Optional to use
noise_model_con = ContinuousDecoherenceModel().add_damping_rate(
    list(range(num_qubits)), 0.001
)
# Here insert the noise if needed
noise_to_use = []


# Function to create the simplified spin Hamiltonian
def create_simplified_tmb_hamiltonian(epsilon, J, K, J_S, number_spins=3):
    """
    Create a simplified spin Hamiltonian for TMB as described in equation 12.

    Parameters:
    epsilon: on-site energy
    J: Coulomb integral
    K: exchange integral
    J_S: effective spin coupling constant
    """
    hamiltonian = SpinHamiltonianSystem(number_spins)

    # Constant term
    constant_term = 3 * (epsilon + J - K) + (3 * J_S) / 4
    hamiltonian.add_operator_product("I", constant_term)

    # Two-spin interaction terms
    coupling = -J_S / 4
    for i in range(number_spins):
        j = (i + 1) % number_spins  # Next spin (circular)
        hamiltonian.add_operator_product(f"{i}X{j}X", coupling)
        hamiltonian.add_operator_product(f"{i}Y{j}Y", coupling)
        hamiltonian.add_operator_product(f"{i}Z{j}Z", coupling)

    return hamiltonian


hamiltonian = create_simplified_tmb_hamiltonian(epsilon, J, K, J_S, num_qubits)

# Define noise_app
noise_app = HqsNoiseApp(noise_mode)

# Create the quantum program, without providing measured_operators and operator_names
quantum_program_initial = noise_app.quantum_program(
    hamiltonian, trotter_timestep, initialisation, [], [], device
)


measurement_circuits = create_XYZ_shadow_circuit(
    num_qubits, shot_per_trotter_step, max_circuits=None
)


# Combine the constant circuit with the measurement circuits.
measurement = measurements.ClassicalRegister(
    constant_circuit=quantum_program_initial.measurement().constant_circuit(),
    circuits=measurement_circuits,
)

# Create the quantum program, and add noise if specified
program = QuantumProgram(
    measurement=measurement, input_parameter_names=["number_trottersteps"]
)
if noise_to_use:
    program = noise_app.add_noise(program, device, noise_to_use)


with open(filename, "w") as f:
    json.dump(program.to_json(), f)

print("Quantum program saved in ", filename)
