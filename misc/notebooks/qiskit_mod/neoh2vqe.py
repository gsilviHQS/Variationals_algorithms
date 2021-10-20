#!/usr/bin/env python

import numpy as np

import warnings
warnings.simplefilter('ignore', np.RankWarning)

from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver

from qiskit_nat import TotalProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nat import MixedMapper

driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 0.735", charge=0, spin=0, unit=UnitsType.ANGSTROM, basis='sto3g')

problem = TotalProblem(driver)

# generate the second-quantized operators
second_q_op = problem.second_q_ops()
el_energy = second_q_op[0]
el_num = second_q_op[1]
el_sz = second_q_op[3]
mx_energy = second_q_op[-1]

converter = QubitConverter(mapper=JordanWignerMapper())
el_en_op  = converter.convert(el_energy)

sgnt = [([0,1,2,3], False),([4,5,6,7],False)]

mixmapper = MixedMapper()
mix_en_op = mixmapper.map(mx_energy,sgnt)

# PARTICLE NUMBER
particle_number = problem.properties_transformed.get_property("ParticleNumber")
num_particles = (particle_number.num_alpha, particle_number.num_beta)
mixnum_particles = (2*particle_number.num_alpha, 2*particle_number.num_beta)
num_spin_orbitals = particle_number.num_spin_orbitals
print("number of spin orbitals=> {}".format(2*num_spin_orbitals))

# SPIN PROJECTION
from qiskit.opflow.operator_globals import I, Z
el_num_op = converter.convert(el_num)
el_n = el_num_op._expand_dim(4)
nuc_n = I.tensorpower(4)^el_num_op

e_sz4 = converter.convert(el_sz)
n_sz = e_sz4._expand_dim(4)
e_sz = I.tensorpower(4)^e_sz4

vqe_en_op = mix_en_op

#CONSTRAINTS ON PARTICLE NUMBER AND SPIN PROJECTION
cshift = 0
if cshift:
   print("CONSTRAINTS ARE APPLIED WITH SHIFT=> {}".format(cshift))
sz_nuc_c  = (n_sz-I.tensorpower(8))@(n_sz-I.tensorpower(8))
sz_el_c   = e_sz@e_sz
num_c     = nuc_n@nuc_n+el_n@el_n - 4.0*nuc_n - 4.0*el_n + 8.0*I.tensorpower(8)

print("Reference Energy=> ",-1.0537650432)

# setup the initial state for the ansatz
from qiskit_nat import NeoHartreeFock

init_state = NeoHartreeFock(2*num_spin_orbitals, mixnum_particles, converter)

# setup the ansatz for VQE
from qiskit.circuit.library import TwoLocal

ansatz = TwoLocal(2*num_spin_orbitals, ['ry', 'rz'], 'cz',reps=2)

# add the initial state
ansatz.compose(init_state, front=True)

# setup the classical optimizer for VQE
from qiskit.algorithms.optimizers import CG
optimizer = CG(maxiter=800)

# setup and run VQE
from qiskit.algorithms import VQE

# set the backend for the quantum computation
from qiskit.utils import QuantumInstance
from qiskit import Aer, BasicAer
be_options = {"max_parallel_threads":8,"max_parallel_experiments":0, "statevector_parallel_threshold":4}
qinstance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1, seed_simulator=2, 
                                                seed_transpiler=2, backend_options = be_options)

vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=qinstance)


from qiskit.opflow import StateFn, CircuitStateFn, Zero, One, PauliExpectation, OperatorBase
from qiskit.opflow.gradients import Gradient, NaturalGradient, Hessian
from qiskit.opflow.gradients.qfi import QFI
from qiskit.quantum_info.operators import Operator, Pauli

exact_state = StateFn(init_state)
h_measure = StateFn(mix_en_op).adjoint()
meas_state = h_measure@exact_state
diag_meas_op = PauliExpectation().convert(meas_state)
print("NEO HF Energy=> ",diag_meas_op.eval())

print("elecectron number=> ",PauliExpectation().convert(StateFn(el_n).adjoint()@exact_state).eval())
print("    nuclei number=> ",PauliExpectation().convert(StateFn(nuc_n).adjoint()@exact_state).eval())
print("    elecectron Sz=> ",PauliExpectation().convert(StateFn(e_sz).adjoint()@exact_state).eval())
print("        nuclei Sz=> ",PauliExpectation().convert(StateFn(n_sz).adjoint()@exact_state).eval())

def aux_print(res):
    print("the Total Electronic Nuclear Energy=> {}".format(res.optimal_value))
    print("[Num el.=> {0:.2f} nuc.=> {1:.2f}] [Sz el=> {2:.2f} nuc=> {3:.2f}]"
                                      .format(res.aux_operator_eigenvalues[0,0],
                                      res.aux_operator_eigenvalues[1,0],
                                      res.aux_operator_eigenvalues[2,0],
                                      res.aux_operator_eigenvalues[3,0]))
    print(','.join(map(str, result.optimal_point)))

# ROUGH INITIAL APPROXIMATION FOR THE PARAMETERS OF TWOLOCAL ANSATZ
vqe.initial_point = [6.,0.,4.,0.,0.,-3.,2.,3.,2.,0.,3.,-6.,-3.,6.,-2.,-0.,2.,0.,-2.,4.,-0.,0.,-6.,-6.,-4.,
                             -3.,3.,-3.,3.,6.,-5.,0.,0.,4.,0.,-4.,3.,-6.,0.,3.,-5.,0.,-4.,0.,2.,0.,-4.,-3.]

vqe_en_op = mix_en_op.reduce()
result = vqe.compute_minimum_eigenvalue(vqe_en_op,aux_operators=[el_n,nuc_n,e_sz,n_sz])
aux_print(result)

total_result = problem.interpret(result)
print(total_result)
