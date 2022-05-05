import sys
import os
#sys.path = []
sys.path.remove('/usr/local/lib64/python3.9/site-packages') #this has qiskit in it (wrong version for this case)
sys.path.remove('/usr/local/lib/python3.9/site-packages')
sys.path.remove('/usr/lib64/python3.9/site-packages')
sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages"))
#sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/bin"))
# sys.path.append(os.path.expanduser("/usr/lib64/python3.9"))
# sys.path.append(os.path.expanduser("/usr/lib64/python3.9/lib-dynload"))
# sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages/qat"))

from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from qiskit.algorithms import VQE
from qiskit.opflow import OperatorBase
from qiskit.compiler import transpile
from qiskit_nature.algorithms import VQEUCCFactory,GroundStateEigensolver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
import qiskit_nature
import numpy as np
from typing import List, Callable, Union
import qiskit
import inspect
# myqlm functions

from importlib.machinery import SourceFileLoader
  
# imports the module from the given path
qatinterop = SourceFileLoader("qat","/home_nfs/gsilvi/.local/lib/python3.9/site-packages/qat/__init__.py").load_module()
# from qat.interop.qiskit import qiskit_to_qlm

# from qatinterop.interop.qiskit import qiskit_to_qlm
from qat.core import Observable
from qat.plugins import Junction
from qat.core import Result
from qat.lang.AQASM import Program, RY
from qat.qlmaas.result import AsyncResult
import time


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
                qcirc = qatinterop.interop.qiskit.qiskit_to_qlm(transpiled_circ)
                #qcirc = transpiled_circ #wrong, to remove
                job = qcirc.to_job(observable=Observable(operator.num_qubits,
                                                         matrix=operator.to_matrix()))
                # START COMPUTATION
                result_temp = self.submit_job(job)

                if isinstance(result_temp, AsyncResult):  # chek if we are dealing with remote
                #     while result_temp.get_status() != 'done':
                #         time.sleep(.1)
                #     result = result_temp.get_result()
                    result_temp.join()
                else:
                    result = result_temp
                results.append(result.value)
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
    def __init__(self, method=None, molecule=None):
        super(IterativeExplorationVQE, self).__init__()
        print('Initialization')
        self.method = method  #updateVQE_MyQLM(method)
        self.updateVQE_MyQLM()
        self.molecule = molecule
        driver = PySCFDriver(atom=molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
        es_problem = ElectronicStructureProblem(driver, q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=[])])

        self.problem = es_problem

    def run(self, initial_job, meta_data):
        # include the method to execute job INSIDE the modified VQE
        #self.method.solver._vqe.submit_job = self.execute
        # run the problem
        vqe_solver = VQEUCCFactory(quantum_instance=None,
                                   optimizer=self.method.solver._optimizer
                                   )
        new_method = GroundStateEigensolver(self.method.qubit_converter, vqe_solver)
        result = self.method.solve(self.problem)
        #meta_data['result'] = result
        #return Result(value=result.total_energies[0], meta_data=meta_data)
        #meta_data['qiskit']=str(inspect.getfullargspec(ElectronicStructureProblem))
        meta_data['qiskit']=str(qiskit_nature.__version__)+str(inspect.getfullargspec(ElectronicStructureProblem))
        return Result(value=33, meta_data=meta_data)


    # def get_specs(x):  # OVERRIDE PROBLEMATIC FUNCTION THAT PREVENT USING REMOTE QLM
    #     return
    def updateVQE_MyQLM(self):

        self.method.solver._vqe = VQE_MyQLM(ansatz=None,
                                                quantum_instance=self.method.solver._quantum_instance,
                                                optimizer=self.method.solver._optimizer,
                                                initial_point=self.method.solver._initial_point,
                                                gradient=self.method.solver._gradient,
                                                expectation=self.method.solver._expectation,
                                                include_custom=self.method.solver._include_custom)
        return
