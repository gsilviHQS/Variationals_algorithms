import sys
import os
os.environ["TMPDIR"] = "/tmp" #set the folder for temporary files
#sys.path = []
sys.path.remove("/usr/local/lib64/python3.9/site-packages") #this has qiskit in it (wrong version for this case)
sys.path.remove("/usr/local/lib/python3.9/site-packages")
sys.path.remove("/usr/lib64/python3.9/site-packages")
sys.path.remove("/usr/lib64/python3.9") #problematic to remove, but it give away the wrong qat
sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages"))
sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages/qat"))
sys.path.append(os.path.expanduser("/usr/lib64/python3.9"))
os.environ["PYTHONPATH"] = '/home_nfs/gsilvi/.local/lib/python3.9/site-packages'
# from importlib import import_module

# interop = import_module('qat.interop', package='qat')

# qiskit_to_qlm = interop.qiskit.qlm_to_qiskit


#sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/bin"))

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


# from importlib.machinery import SourceFileLoader
# imports the module from the given path

from qat.core import Observable
from qat.plugins import Junction
from qat.core import Result
from qat.qlmaas.result import AsyncResult
import qat

import importlib.util
from importlib import reload 


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
                
                qcirc = self.q2q(transpiled_circ)
                job = qcirc.to_job(observable=Observable(operator.num_qubits,
                                                                  matrix=operator.to_matrix()))
                # START COMPUTATION
                result_temp = self.submit_job(job)

                if isinstance(result_temp, AsyncResult):  # chek if we are dealing with remote
                    print('mismacth plugin(local)| qpu(remote)')
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
    def __init__(self, method=None, molecule=None, remove_orbitals=[], converter=None, solver=None):
        super(IterativeExplorationVQE, self).__init__()
        #
        # spec = importlib.util.spec_from_file_location("qat","/home_nfs/gsilvi/.local/lib/python3.9/site-packages/qat/__init__.py",submodule_search_locations=[''])
        # module = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(module)
        # sys.modules['qat'] = module
        reload(qat)
        from qat.interop.qiskit import qiskit_to_qlm
        self.q2q = qiskit_to_qlm

        #self.method = method
        self.old_solver = solver
        self.method = GroundStateEigensolver(converter, solver)
        self.updateVQE_MyQLM()
        # self.molecule = molecule
        driver = PySCFDriver(atom=molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
        es_problem = ElectronicStructureProblem(driver, q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=remove_orbitals)])
        self.problem = es_problem

    def run(self, initial_job, meta_data):
        # include the method to execute job INSIDE the modified VQE
        self.method.solver._vqe.submit_job = self.execute
        self.method.solver._vqe.q2q = self.q2q
        # run the problem
        result = self.method.solve(self.problem)
        #meta_data['optimal_parameters'] = str(len(result.raw_result.optimal_parameters))
        # meta_data['optimal_parameters'] = str(os.path.abspath(qat.__file__))
        
        fout = os.path.expanduser("~/pip_log_out")
        ferr = os.path.expanduser("~/pip_log_err")
        os.system(f'echo $PYTHONPATH> {fout} 2> {ferr}')
        with open(fout, 'r') as fin:
            data_out = fin.read()
        with open(ferr, 'r') as fin:
            data_err = fin.read()
        meta_data['optimal_parameters'] = str(data_out)+str(data_err)+'__'+str(sys.path)+'_QAT:'+str(os.path.abspath(qat.__file__))
        self.method.solver = self.old_solver
        return Result(value=result.total_energies[0], meta_data=meta_data)

    def updateVQE_MyQLM(self):

        self.method.solver._vqe = VQE_MyQLM(ansatz=None,
                                            quantum_instance=self.method.solver._quantum_instance,
                                            optimizer=self.method.solver._optimizer,
                                            initial_point=self.method.solver._initial_point,
                                            gradient=self.method.solver._gradient,
                                            expectation=self.method.solver._expectation,
                                            include_custom=self.method.solver._include_custom)
        return
