import sys
import os
os.environ["TMPDIR"] = "/tmp"  # set the folder for temporary files

paths = ["/usr/local/lib64/python3.9/site-packages","/usr/local/lib/python3.9/site-packages","/usr/lib64/python3.9/site-packages"]

for path in paths:
    if path in sys.path:
            sys.path.remove(path)

sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages"))

import qiskit
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from qiskit.algorithms import VQE
from qiskit.opflow import OperatorBase,StateFn,CircuitSampler,CircuitStateFn,PauliOp, PauliSumOp
from qiskit.compiler import transpile
from qiskit_nature.algorithms import VQEUCCFactory,GroundStateEigensolver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit.quantum_info import Pauli,SparsePauliOp
import numpy as np
from typing import List, Callable, Union
from qat.core import Observable,Result, Term
from qat.plugins import Junction
from qat.qlmaas.result import AsyncResult
import qat
from importlib import reload
reload(qat)
from qat.interop.qiskit import qiskit_to_qlm,qlm_to_qiskit
import json
#Need to find a way to include those libraries, try uploading multiplt files
from qiskit_mod.qiskit_nat import VHA
from qiskit_mod.qiskit_ter import LinCombFullmod,LinCombMod
import inspect  

LIST_OF_GATES = ['i', 'id', 'iden', 'u', 'u0', 'u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'ch', 'crz', 'cu1', 'cu3', 'swap', 'ccx', 'cswap', 'r']

def encode_complex(z):
    """ Encode a complex number as a string. """
    if isinstance(z, complex):
        return (z.real, z.imag)
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")

class IterativeExplorationVQE(Junction):
    def __init__(self,
                 method=None,
                 molecule=None,
                 remove_orbitals=[],
                 shots=0):
        super(IterativeExplorationVQE, self).__init__()
        self.shots = shots
        self.method = method

        # create electronic structure problem
        driver = PySCFDriver(atom=molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
        
        if 'q_molecule_transformers' in inspect.getfullargspec(ElectronicStructureProblem).args:
            self.problem = ElectronicStructureProblem(driver, q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=remove_orbitals)])
        else:
            self.problem = ElectronicStructureProblem(driver, transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=remove_orbitals)])

    def run(self, initial_job, meta_data):
        self.method._solver._vqe.execute = self.execute
        self.method._solver._vqe.nb_shots = self.shots

        # run the problem
        result = self.method.solve(self.problem)

        #save metadata
        if hasattr(result,'raw_result'):
            meta_data['raw_result'] = str(result.raw_result)
            meta_data['optimal_parameters'] = json.dumps(list(result.raw_result.optimal_parameters.values()))
        if hasattr(result,'hartree_fock_energy'):    
            meta_data['hartree_fock_energy'] = str(result.hartree_fock_energy)
        if hasattr(result,'num_iterations'):
            meta_data['num_iterations'] = str(result.num_iterations)
        if hasattr(result,'finishing_criterion'):
            meta_data['finishing_criterion'] = str(result.finishing_criterion)
        meta_data['qiskit_version'] = str(qiskit.__qiskit_version__)#str(os.path.abspath(qat.__file__))
        
        return Result(value=result.total_energies[0], meta_data=meta_data)


    # def get_specs(self):
    #     return


class IterativeExplorationEvoVQE(Junction):
    def __init__(self,
                 method=None,
                 execute_function= None,
                 operator=None,
                 shots=0,
                 **kwargs):
        super(IterativeExplorationEvoVQE, self).__init__()
        self.shots = shots
        self.method = method
        self.method.nb_shots = self.shots
        self.method.execute_function = getattr(self.method, execute_function)
        self.operator = operator
        if kwargs:
            self.kwargs = kwargs
    
    def run(self, initial_job, meta_data):
        # include the method to execute job INSIDE the modified VQE
        self.method.execute = self.execute
        

        # run the problem
        if hasattr(self,'kwargs'):
            result = self.method.execute_function(self.operator, **self.kwargs)
        else:
            result = self.method.execute_function(self.operator)


        #save metadata
        if hasattr(result,'optimal_point'):
           meta_data['optimal_point'] = json.dumps(result.optimal_point.tolist())
        if hasattr(result,'optimal_parameters'):
            meta_data['optimal_parameters'] = json.dumps(list(result.optimal_parameters.values()))
        if hasattr(result,'optimal_value'):    
            meta_data['optimal_value'] = str(result.optimal_value)
        if hasattr(result,'cost_function_evals'):
            meta_data['cost_function_evals'] = str(result.cost_function_evals)
        if hasattr(result,'optimizer_time'):
            meta_data['optimizer_time'] = str(result.optimizer_time)
        if hasattr(result,'eigenvalue'):
            meta_data['eigenvalue'] = str(result.eigenvalue)
        if hasattr(result,'eigenstate'):
            meta_data['eigenstate'] = json.dumps(result.eigenstate.tolist(), default=encode_complex)
        if hasattr(result,'aux_operator_eigenvalues'):
            meta_data['aux_operator_eigenvalues'] = json.dumps(result.aux_operator_eigenvalues.tolist())
        meta_data['qiskit_version'] = str(qiskit.__qiskit_version__)#str(os.path.abspath(qat.__file__))
        

        return Result(value=result.eigenvalue, meta_data=meta_data)


def get_energy_evaluation_QLM(
        self,
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Returns a function handle to evaluates the energy at given parameters for the ansatz.

        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            operator: The operator whose energy to evaluate.
            return_expectation: If True, return the ``ExpectationBase`` expectation converter used
                in the construction of the expectation value. Useful e.g. to evaluate other
                operators with the same expectation value converter.
            MODIFIED: submit the job to QLM qpus


        Returns:
            Energy of the hamiltonian of each parameter, and, optionally, the expectation
            converter.

        Raises:
            RuntimeError: If the circuit is not parameterized (i.e. has 0 free parameters).

        """
        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        expect_op, expectation = self.construct_expectation(
            self._ansatz_params, operator, return_expectation=True
        )
        reversed_operator = sum([PauliOp(primitive= Pauli(op.to_pauli_op().primitive.to_label()[::-1]), coeff=op.to_pauli_op().coeff) for op in operator])
        #print('Operator type',type(reversed_operator), reversed_operator)
        #print('Here',expectation.convert(StateFn(operator, is_measurement=True)).to_matrix_op())

        def create_and_run_job(operator_meas: OperatorBase,
                               params: dict):
            """ Compose the qlm job from ansatz binded to params and measure operato on it"""
            # check if the parameters passed are as a range or single value
            if params is not None and len(params.keys()) > 0:
                p_0 = list(params.values())[0]
                if isinstance(p_0, (list, np.ndarray)):
                    num_parameterizations = len(p_0)
                    param_bindings = [
                        {param: value_list[i] for param, value_list in params.items()}
                        for i in range(num_parameterizations)
                    ]
                else:
                    num_parameterizations = 1
                    param_bindings = [params]

            else:
                param_bindings = None
                num_parameterizations = 1
            results = []
            
            for circ_params in param_bindings:
                ansatz_in_use = self._ansatz.assign_parameters(circ_params)

                transpiled_circ = transpile(ansatz_in_use.decompose(),
                                            basis_gates=LIST_OF_GATES,
                                            optimization_level=0)
                qcirc = qiskit_to_qlm(transpiled_circ)
                job = qcirc.to_job(observable=Observable(operator_meas.num_qubits, matrix=operator_meas.to_matrix()),nbshots=self.nb_shots) 
                # START COMPUTATION
                result_temp = self.execute(job)

                if isinstance(result_temp, AsyncResult):  # chek if we are dealing with remote job
                    #case for local plugin/remote qpu
                    result = result_temp.join()
                    
                else:
                    result = result_temp
                #Qiskit check
                # sampler = CircuitSampler(self.quantum_instance)
                # exp = ~StateFn(operator_meas) @ CircuitStateFn(transpiled_circ)
                # converted = sampler.convert(exp)
                # print('Evaluation (qiskit)',converted.eval(),'vs QLM',result.value)

                results.append(result.value)
            
            return results

        def energy_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(zip(self._ansatz_params, parameter_sets.transpose().tolist())) # combine values to parameters symbols
            #print('param_bindings',param_bindings)
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=param_bindings)
            # print(operator)
            
            means = np.real(create_and_run_job(reversed_operator, param_bindings)) #important to reverse the operator because of different conventions QLM/Qiskit

            if self._callback is not None:
                parameter_sets = np.reshape(parameters, (-1, num_parameters))
                param_bindings = dict(zip(self._ansatz_params, parameter_sets.transpose().tolist()))
                variance = np.real(expectation.compute_variance(self._circuit_sampler.convert(expect_op, params=param_bindings)))
                estimator_error = np.sqrt(variance / self.quantum_instance.run_config.shots)
                for i, param_set in enumerate(parameter_sets):
                    self._eval_count += 1
                    self._callback(self._eval_count, param_set, means[i], estimator_error[i])
            else:
                self._eval_count += len(means)
            return means if len(means) > 1 else means[0]

        if return_expectation:
            return energy_evaluation, expectation

        return energy_evaluation