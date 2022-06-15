# -*- coding : utf-8 -*-

"""
Test for this module
"""

from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.algorithms import VQEUCCFactory,GroundStateEigensolver,AdaptVQE
from qiskit_nature.circuit.library import HartreeFock ,UCC

from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP,L_BFGS_B,SPSA, COBYLA
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
from qiskit.circuit.library import TwoLocal
from qiskit.test.mock import FakeVigo
from qat.interop.qiskit import QPUToBackend
from qat.qpus import get_default_qpu,PyLinalg


from qiskit_mod.my_junction import IterativeExplorationVQE
from qiskit_mod.wrapper2myqlm import build_QLM_stack,run_QLM_stack
from qiskit_mod.qiskit_ter import LinCombFullmod,LinCombMod
from qiskit_mod.qiskit_nat import VHA




class TestIterativeExplorationVQE:
    """
    Testing IterativeExplorationVQE
    """
    def initialize_requirement(self):
        """
        Initialize requirement for the class
        """
        qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
        molecule = 'H .0 .0 -{0}; H .0 .0 {0}'.format(.5)
        quantum_instance = QuantumInstance(backend=AerSimulator(method="statevector"), max_credits=None)
        vqe_solver = VQEUCCFactory(quantum_instance, optimizer=L_BFGS_B(maxiter=1000))
        calc = GroundStateEigensolver(qubit_converter, vqe_solver)
        return calc, molecule


    #1. test if the method is called
    def test_init(self):
        """
        Test if the method is called
        """
        #create a new instance of the class
        method,molecule = self.initialize_requirement()
        instance = IterativeExplorationVQE(method,molecule)
        #check if the method is called
        assert instance.__init__ is not None

 
    
    #3. test if the method returns the right result
    def test_run_locally(self):
        """
        Test if the method returns the right result
        """
        qpu_local = PyLinalg() #local QPU
        qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
        molecule = 'H .0 .0 -{0}; H .0 .0 {0}'.format(.5)
        driver = PySCFDriver(atom=molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
        es_problem = ElectronicStructureProblem(driver,q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=[])])
        quantum_instance = QuantumInstance(backend=AerSimulator(method="statevector"), max_credits=None)
        vqe_solver = VQEUCCFactory(quantum_instance, optimizer=L_BFGS_B(maxiter=1000))
        calc = GroundStateEigensolver(qubit_converter, vqe_solver)

        res_qiskit = calc.solve(es_problem)
        assert res_qiskit is not None
        print(res_qiskit.total_energies)

        stack = build_QLM_stack(calc,
                                molecule,
                                IterativeExplorationVQE,
                                qpu_local
                                )
        assert stack is not None
        res_qlm = run_QLM_stack(stack)
        assert res_qlm is not None
        print(res_qlm.total_energies)

        #compare the complex results to make sure th are almost the same, up to a certain tolerance
        assert abs(res_qiskit.total_energies - res_qlm.total_energies) < 1e-10

       

        

class TestVHA:
    """
    Testing VHA
    """
    def initialize_requirement(self):
        """
        Initialize the class
        """
        qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
        molecule = 'H .0 .0 -{0}; H .0 .0 {0}'.format(.5)
        quantum_instance = QuantumInstance(backend=AerSimulator(method="statevector"), max_credits=None)
        vqe_solver = VQEUCCFactory(quantum_instance, optimizer=L_BFGS_B(maxiter=1000))
        calc = GroundStateEigensolver(qubit_converter, vqe_solver)
        return calc, molecule
    
    #1. test if the method is called
    def test_init(self):
        """
        Test if the method is called
        """
        #create a new instance of the class
        instance = VHA(excitations='sd',
                       trotter_steps=1,
                       only_excitations=True,
                       num_particles=(1,1),
                       num_spin_orbitals=4)
        #check if the method is called
        assert instance.__init__ is not None

    def test_excitations(self):
        """
        Test if the method returns the right result
        """
        #create a new instance of the class
        instance = VHA(excitations='sd',
                       trotter_steps=1,
                       only_excitations=True,
                       num_particles=(1,1),
                       num_spin_orbitals=4)
        #check if the method is called
        assert instance.excitation_ops is not None
        labels = [('NIII', (1+0j)),
                ('INII', (1+0j)),
                ('IINI', (1+0j)),
                ('IIIN', (1+0j)),
                ('NNII', (1+0j)),
                ('NINI', (1+0j)),
                ('NIIN', (1+0j)),
                ('INNI', (1+0j)),
                ('ININ', (1+0j)),
                ('IINN', (1+0j))]
        ferm_op = instance.excitation_ops()[2]
        #check if the method returns the right result
        assert labels==ferm_op.to_list()

class TestLinCombFullmod:
    """
    Testing LinCombFullmod
    """
    def test_result(self):
        """
        Test if the method is called
        """
        driver = PySCFDriver(atom='H .0 .0 -{0}; H .0 .0 {0}'.format(.5),
                             unit=UnitsType.ANGSTROM,
                             basis='sto3g')
        es_problem = ElectronicStructureProblem(driver,
                                                q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True,
                                                                                               remove_orbitals=[])])
        qubit_converter = QubitConverter(mapper=JordanWignerMapper())
        instance_1 = NaturalGradient(grad_method=LinCombMod(img=False),
                                     qfi_method=LinCombFullmod(),
                                     regularization='ridge')
        instance_2 = NaturalGradient(qfi_method=LinCombFullmod(),
                                     regularization='ridge')
        hartree_fock_state = HartreeFock(num_spin_orbitals=4,
                                         num_particles=(1,1), 
                                         qubit_converter=qubit_converter)
        quantum_instance = QuantumInstance(backend=AerSimulator(method="statevector"))
        ansz = UCC(qubit_converter,
                   num_particles=(1,1),
                   num_spin_orbitals=4,
                   excitations='s' ,
                   initial_state=hartree_fock_state)
        Im_solver_1 = VQEUCCFactory(quantum_instance,
                                    gradient=instance_1,
                                    optimizer=COBYLA(maxiter=1000),
                                    ansatz=ansz.decompose())
        calcIevo_1 = GroundStateEigensolver(qubit_converter,
                                            Im_solver_1)
        Im_solver_2 = VQEUCCFactory(quantum_instance,
                                    gradient=instance_2,
                                    optimizer=COBYLA(maxiter=1000),
                                    ansatz=ansz.decompose())
        calcIevo_2 = GroundStateEigensolver(qubit_converter,
                                            Im_solver_2)

        resIevo_1 = calcIevo_1.solve(es_problem)
        resIevo_2 = calcIevo_2.solve(es_problem)
        assert resIevo_1 is not None
        assert resIevo_2 is not None
        print(resIevo_1.total_energies)
        print(resIevo_2.total_energies)
        assert abs(resIevo_1.total_energies - resIevo_2.total_energies) < 1e-3
    def test_result_2(self):
        
        

