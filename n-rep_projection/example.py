"""Example file illustrating the usage of the n-representability projection."""
from projection import get_energy, best_fixed_trace_positive_projection
import numpy as np
from openfermion import (MolecularData, get_fermion_operator, jordan_wigner_code,
                         binary_code_transform, get_sparse_operator, get_ground_state,
                         get_density_matrix, FermionOperator)
from openfermion.utils import depolarizing_channel
from openfermionpyscf import run_pyscf


# System definition and FCI calculation of H2
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 1.))]
multiplicity = 1
basis = 'sto-3g'
prep_molecule = MolecularData(geometry, basis, multiplicity)
molecule = run_pyscf(prep_molecule, run_scf=True, run_fci=True, verbose=1)
hamiltonian = molecule.get_molecular_hamiltonian()

# Setting the Hamiltonian parameters
number_electrons = 2
number_orbitals = 4
energy_offset = hamiltonian.constant
one_electron_integrals = hamiltonian.one_body_tensor
two_electron_integrals = hamiltonian.two_body_tensor

# Getting the ground state density matrix of the system after conversion to a qubit system
fermion_hamiltonian = get_fermion_operator(hamiltonian)
code = jordan_wigner_code(number_orbitals)
qubit_hamiltonian = binary_code_transform(fermion_hamiltonian, code)
hamiltonian_original = get_sparse_operator(qubit_hamiltonian).toarray()
energy_original, wavefunction_original = get_ground_state(hamiltonian_original)
density_matrix = get_density_matrix(wavefunction_original[np.newaxis, :], [1]).toarray()

# Getting a noisy density matrix by adding depolarization
density_matrix_noisy = depolarizing_channel(density_matrix, 0.1, target_qubit="all")

# Constructing the 1- and 2-particle RDMs of perfect and noisy ground states
rdm1 = np.zeros((number_orbitals, number_orbitals), dtype='complex128')
for i in range(number_orbitals):
    for j in range(number_orbitals):
        rdm1_operator = FermionOperator(((i, 1), (j, 0)), 1.)
        rdm1_operator_jw = binary_code_transform(rdm1_operator, code)
        rdm1_op = get_sparse_operator(rdm1_operator_jw, number_orbitals).toarray()
        rdm1[i][j] = np.trace(density_matrix @ rdm1_op)

rdm2 = np.zeros((number_orbitals, number_orbitals, number_orbitals, number_orbitals), dtype='complex128')
for p in range(number_orbitals):
    for q in range(number_orbitals):
        for r in range(number_orbitals):
            for s in range(number_orbitals):
                rdm2_operator = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), 1.)
                rdm2_operator_jw = binary_code_transform(rdm2_operator, code)
                rdm2_op = get_sparse_operator(rdm2_operator_jw, number_orbitals).toarray()
                rdm2[p][q][r][s] = np.trace(density_matrix @ rdm2_op)

rdm1_noisy = np.zeros((number_orbitals, number_orbitals), dtype='complex128')
for i in range(number_orbitals):
    for j in range(number_orbitals):
        rdm1_operator = FermionOperator(((i, 1), (j, 0)), 1.)
        rdm1_operator_jw = binary_code_transform(rdm1_operator, code)
        rdm1_op = get_sparse_operator(rdm1_operator_jw, number_orbitals).toarray()
        rdm1_noisy[i][j] = np.trace(density_matrix_noisy @ rdm1_op)

rdm2_noisy = np.zeros((number_orbitals, number_orbitals, number_orbitals, number_orbitals), dtype='complex128')
for p in range(number_orbitals):
    for q in range(number_orbitals):
        for r in range(number_orbitals):
            for s in range(number_orbitals):
                rdm2_operator = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), 1.)
                rdm2_operator_jw = binary_code_transform(rdm2_operator, code)
                rdm2_op = get_sparse_operator(rdm2_operator_jw, number_orbitals).toarray()
                rdm2_noisy[p][q][r][s] = np.trace(density_matrix_noisy @ rdm2_op)

# Getting the enery of the perfect and noisy ground state
energy = get_energy(rdm1, rdm2, energy_offset, one_electron_integrals, two_electron_integrals)
energy_noisy = get_energy(rdm1_noisy, rdm2_noisy, energy_offset, one_electron_integrals, two_electron_integrals)

# Perform n-representability projection, get energy after projection incl. respective 1- and 2-RDM
energy_projected, rdm1_projected, rdm2 = best_fixed_trace_positive_projection(rdm1_noisy, rdm2_noisy, number_electrons, energy_offset, one_electron_integrals, two_electron_integrals)

# Print results
print()
print("Perfect ground state energy:", energy.real)
print("Noisy ground state energy:  ", energy_noisy.real)
print("Energy after projection:    ", energy_projected.real)
print()
print("Energy difference noisy to perfect ground state:    ", (energy_noisy - energy).real)
print("Energy difference projected to perfect ground state:", (energy_projected - energy).real)
