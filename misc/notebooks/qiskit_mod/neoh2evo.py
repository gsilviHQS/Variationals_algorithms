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

#CONSTRAINTS ON PARTICLE NUMBER AND SPIN PROJECTION
cshift = 0.25

# PARTICLE NUMBER
particle_number = problem.properties_transformed.get_property("ParticleNumber")
num_particles = (particle_number.num_alpha, particle_number.num_beta)
mixnum_particles = (2*particle_number.num_alpha, 2*particle_number.num_beta)
num_spin_orbitals = particle_number.num_spin_orbitals
print("number of spin orbitals=> {}".format(num_spin_orbitals))

from qiskit.opflow.operator_globals import I, Z
el_num_op = converter.convert(el_num)
el_n = el_num_op._expand_dim(4)
nuc_n = I.tensorpower(4)^el_num_op

e_sz4 = converter.convert(el_sz)
n_sz = e_sz4._expand_dim(4)
e_sz = I.tensorpower(4)^e_sz4

vqe_en_op = mix_en_op

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

ansatz = TwoLocal(2*num_spin_orbitals, ['ry', 'rz'], 'cz')

# add the initial state
#ansatz.compose(init_state, front=True)

# setup the classical optimizer for VQE
from qiskit.algorithms.optimizers import L_BFGS_B
optimizer = L_BFGS_B(maxiter=800)

# setup and run VQE
from qiskit.algorithms import VQE

# set the backend for the quantum computation
from qiskit.utils import QuantumInstance
from qiskit import Aer, BasicAer
qinstance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1, seed_simulator=2, seed_transpiler=2)
#qinstance = QuantumInstance(Aer.get_backend('aer_simulator_statevector_gpu'), shots=1, seed_simulator=2, seed_transpiler=2)

from qiskit_ter import LinCombFullmod,LinCombMod, EvoVQE
vqe = EvoVQE(ansatz, optimizer=optimizer, quantum_instance=qinstance)

def aux_print(res):
    print("the Total Electronic Nuclear Energy=> {}".format(res.optimal_value))
    print("[Num el.=> {0:.2f} nuc.=> {1:.2f}] [Sz el=> {2:.2f} nuc=> {3:.4f}] Constr. num=> {4:.4f} sz_nuc=> {5:.4f} sz_el=> {6:.4f}]"
                                                              .format(res.aux_operator_eigenvalues[0,0],
                                                                      res.aux_operator_eigenvalues[1,0],
                                                                      res.aux_operator_eigenvalues[2,0],
                                                                      res.aux_operator_eigenvalues[3,0],
                                                                      res.aux_operator_eigenvalues[4,0],
                                                                      res.aux_operator_eigenvalues[5,0],
                                                                      res.aux_operator_eigenvalues[6,0]))
    #print("optimal parameters \n{}".format(result.optimal_point))
    #print(','.join(map(str, result.optimal_point)))
    
vqe.initial_point = [-7.8,4.7,1.5,6.2,0.,4.3,6.2,0.,-4.2,-3.1,3.2,4.3,-5.6,2.1,0.9,4.3,3.1,-3.1,6.2,-0.1,-3.1,1.5,-3.1,-3.1,6.0,-1.3,1.2,5.8,2.2,4.7,5.7,-1.5,0.,6.2,6.2,3.3,3.1,1.5,-6.2,3.1,3.8,-3.4,-2.9,1.8,-3.0,-3.7,2.8,1.8,4.7,-1.5,1.5,-3.1,-3.1,0.6,-3.1,-6.2,1.5,0.,-5.1,1.1,-2.6,2.9,5.3,-2.2]
vqe_en_op = mix_en_op.reduce()
result = vqe.compute_evolve(vqe_en_op, isteps=20, rsteps=40 di=0.1, dr=0.01)
aux_print(result)
#
#vqe.initial_point=result.optimal_point
#vqe_en_op = (mix_en_op + cshift*num_c).reduce()
#result = vqe.compute_minimum_eigenvalue(vqe_en_op,aux_operators=[el_n,nuc_n,e_sz,n_sz,num_c,sz_nuc_c,sz_el_c])
#aux_print(result)
#
#vqe.initial_point=result.optimal_point
#vqe_en_op = (mix_en_op + cshift*(num_c + 5.*sz_nuc_c)).reduce()
#result = vqe.compute_minimum_eigenvalue(vqe_en_op,aux_operators=[el_n,nuc_n,e_sz,n_sz,num_c,sz_nuc_c,sz_el_c])
#aux_print(result)
#
#vqe.initial_point=result.optimal_point
##vqe_en_op = (mix_en_op + cshift*(num_c + 5.0*sz_nuc_c + 10.0*sz_el_c)).reduce()
#vqe_en_op = (mix_en_op + cshift*(num_c + 5.*(sz_nuc_c + sz_el_c))).reduce()
##vqe_en_op = (mix_en_op + cshift*(num_c + sz_nuc_c + sz_el_c)).reduce()
#result = vqe.compute_minimum_eigenvalue(vqe_en_op,aux_operators=[el_n,nuc_n,e_sz,n_sz,num_c,sz_nuc_c,sz_el_c])
#aux_print(result)
#
total_result = problem.interpret(result)
print(total_result)
