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
from qiskit.opflow import OperatorBase
from qiskit.compiler import transpile
from qiskit_nature.algorithms import VQEUCCFactory,GroundStateEigensolver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
import numpy as np
from typing import List, Callable, Union
from qat.core import Observable
from qat.plugins import Junction
from qat.core import Result
from qat.qlmaas.result import AsyncResult
import qat
from importlib import reload
reload(qat)
from qat.interop.qiskit import qiskit_to_qlm
import json
#Need to find a way to include those libraries, try uploading multiplt files
from qiskit_mod.qiskit_nat import VHA
from qiskit_mod.qiskit_ter import LinCombFullmod,LinCombMod
import inspect  

class VQE_MyQLM(VQE):
    """Modified version of VQE to submit jobs to QLM qpus."""
    def get_energy_evaluation(
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

        def create_and_run_job(operator: OperatorBase,
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
                ansatz_in_use = self._ansatz.bind_parameters(circ_params)

                transpiled_circ = transpile(ansatz_in_use.decompose(),
                                            basis_gates=self._quantum_instance.backend.configuration().basis_gates,
                                            optimization_level=0)
                
                qcirc = qiskit_to_qlm(transpiled_circ)
                job = qcirc.to_job(observable=Observable(operator.num_qubits, matrix=operator.to_matrix())
                )#,nbshots=self.nb_shots)
                # START COMPUTATION
                result_temp = self.execute(job)

                if isinstance(result_temp, AsyncResult):  # chek if we are dealing with remote job
                    #case for local plugin/remote qpu
                    result = result_temp.join()
                    
                else:
                    result = result_temp
                
                results.append(result.value)
            #print('Temp results:',results,param_bindings)
            return results

        def energy_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(zip(self._ansatz_params, parameter_sets.transpose().tolist()))

            means = np.real(create_and_run_job(operator, param_bindings))

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


class IterativeExplorationVQE(Junction):
    def __init__(self,
                 method=None,
                 molecule=None,
                 remove_orbitals=[],
                 shots=0):
        super(IterativeExplorationVQE, self).__init__()
        self.shots = shots
        self.method = method
        self.updateVQE_MyQLM()

        # create electronic structure problem
        driver = PySCFDriver(atom=molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
        
        if 'q_molecule_transformers' in inspect.getfullargspec(ElectronicStructureProblem).args:
            self.problem = ElectronicStructureProblem(driver, q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=remove_orbitals)])
        else:
            self.problem = ElectronicStructureProblem(driver, transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=remove_orbitals)])

    def run(self, initial_job, meta_data):
        # include the method to execute job INSIDE the modified VQE
        #meta_data['single_job']= 'False' #test if this is necessary
        self.method._solver._vqe.execute = self.execute
        self.method._solver._vqe.nb_shots = self.shots

        print(self.method._solver._vqe.nb_shots)
        # run the problem
        result = self.method.solve(self.problem)

        #save metadata

        meta_data['optimal_parameters'] = json.dumps(list(result.raw_result.optimal_parameters.values()))
        meta_data['hartree_fock_energy'] = str(result.hartree_fock_energy)
        if hasattr(result,'num_iterations'):
            meta_data['num_iterations'] = str(result.num_iterations)
        if hasattr(result,'finishing_criterion'):
            meta_data['finishing_criterion'] = str(result.finishing_criterion)
        meta_data['qat'] = str(qiskit.__qiskit_version__)#str(os.path.abspath(qat.__file__))
        meta_data['raw_result'] = str(result.raw_result)

        # self.method._solver = self.old_solver
        return Result(value=result.total_energies[0], meta_data=meta_data)

    def updateVQE_MyQLM(self):

        self.method._solver._vqe = VQE_MyQLM(ansatz=None,
                                            quantum_instance=self.method._solver._quantum_instance,
                                            optimizer=self.method._solver._optimizer,
                                            initial_point=self.method._solver._initial_point,
                                            gradient=self.method._solver._gradient,
                                            expectation=self.method._solver._expectation,
                                            include_custom=self.method._solver._include_custom)
        return
    # def get_specs(self):
    #     return




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

        def create_and_run_job(operator: OperatorBase,
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
                ansatz_in_use = self._ansatz.bind_parameters(circ_params)

                transpiled_circ = transpile(ansatz_in_use.decompose(),
                                            basis_gates=self._quantum_instance.backend.configuration().basis_gates,
                                            optimization_level=0)
                
                qcirc = qiskit_to_qlm(transpiled_circ)
                job = qcirc.to_job(observable=Observable(operator.num_qubits, matrix=operator.to_matrix())
                )#,nbshots=self.nb_shots)
                # START COMPUTATION
                result_temp = self.execute(job)

                if isinstance(result_temp, AsyncResult):  # chek if we are dealing with remote job
                    #case for local plugin/remote qpu
                    result = result_temp.join()
                    
                else:
                    result = result_temp
                
                results.append(result.value)
            #print('Temp results:',results,param_bindings)
            return results

        def energy_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(zip(self._ansatz_params, parameter_sets.transpose().tolist()))

            means = np.real(create_and_run_job(operator, param_bindings))

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
class IterativeExplorationEvoVQE(Junction):
    def __init__(self,
                 method=None,
                 operator=None,
                 shots=0):
        super(IterativeExplorationEvoVQE, self).__init__()
        self.shots = shots
        self.method = method
        self.operator = operator
        self.updateVQE_MyQLM()
    
    def run(self, initial_job, meta_data):
        # include the method to execute job INSIDE the modified VQE
        #meta_data['single_job']= 'False' #test if this is necessary
        self.method._solver._vqe.execute = self.execute
        self.method._solver._vqe.nb_shots = self.shots

        # run the problem
        result = self.method.compute_evolve(self.operator)

        #save metadata

        meta_data['optimal_parameters'] = json.dumps(list(result.raw_result.optimal_parameters.values()))
        meta_data['hartree_fock_energy'] = str(result.hartree_fock_energy)
        if hasattr(result,'num_iterations'):
            meta_data['num_iterations'] = str(result.num_iterations)
        if hasattr(result,'finishing_criterion'):
            meta_data['finishing_criterion'] = str(result.finishing_criterion)
        meta_data['qat'] = str(qiskit.__qiskit_version__)#str(os.path.abspath(qat.__file__))
        meta_data['raw_result'] = str(result.raw_result)

        # self.method._solver = self.old_solver
        return Result(value=result.total_energies[0], meta_data=meta_data)

    def updateVQE_MyQLM(self):

        self.method.get_energy_evaluation = get_energy_evaluation_QLM
        return
    # def get_specs(self):
    #     return
