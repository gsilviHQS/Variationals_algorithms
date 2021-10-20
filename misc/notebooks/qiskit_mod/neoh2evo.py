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
from qiskit.algorithms.optimizers import L_BFGS_B
optimizer = L_BFGS_B(maxiter=800)

# setup and run VQE
from qiskit.algorithms import VQE

# set the backend for the quantum computation
from qiskit.utils import QuantumInstance
from qiskit import Aer, BasicAer
be_options = {"max_parallel_threads":8,"max_parallel_experiments":0, "statevector_parallel_threshold":4}
qinstance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=1, seed_simulator=2, 
                                                   seed_transpiler=2, backend_options=be_options)
#qinstance = QuantumInstance(Aer.get_backend('aer_simulator_statevector_gpu'), shots=1, seed_simulator=2, seed_transpiler=2)

from qiskit_ter import LinCombFullmod,LinCombMod, EvoVQE
vqe = EvoVQE(ansatz, optimizer=optimizer, quantum_instance=qinstance)

#vqe.initial_point = [6.,0.,4.,0.,0.,-3.,2.,3.,2.,0.,3.,-6.,-3.,6.,-2.,-0.,2.,0.,-2.,4.,-0.,0.,-6.,-6.,-4.,-3.,3.,-3.,3.,6.,-5.,0.,0.,4.,0.,-4.,3.,-6.,0.,3.,-5.,0.,-4.,0.,2.,0.,-4.,-3.]
vqe.initial_point = [6.2831848609129874,1.570543787554636,4.711297700245818,-1.569954059720944,-0.00016948680218153091,-3.1415911265531604,-6.822182672631528e-07,3.1416210172436205,2.0402096916548556,-0.07603280718678695,2.9356341241956114,-6.260527545120678,-2.9990325055739464,6.0077772114760375,-1.6529070475437981,0.003429653517713078,2.9955518372510346,0.0022045505208202823,-3.1415103683814114,3.1438770710772217,-3.818621105169968e-06,-9.153988035045935e-06,-6.283186243286005,-6.283185706788387,-3.8068804819715463,-3.0655600369127636,2.935635291028905,-3.118934856728131,3.005815571424384,6.014460455363797,-5.032449511052682,-0.0041493348713920915,4.969001863728771e-07,4.71264347562269,1.5718876776486708,-4.711544772975571,3.141418886443523,-6.283184008700246,9.829716636235556e-07,3.1416209606216636,-4.917035050213839,-0.08296782480382431,-3.9627139625557906,-0.037288990338437446,2.0025839032415687,-0.00258431339752288,-4.002880231659448,-2.997123494261866]
vqe_en_op = mix_en_op.reduce()
result = vqe.compute_evolve(vqe_en_op, isteps=2, rsteps=80, di=0.2, dr=0.01)

total_result = problem.interpret(result)
print(total_result)
