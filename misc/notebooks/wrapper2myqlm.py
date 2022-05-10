from qiskit.algorithms import VQE
from qiskit.opflow import OperatorBase
from qiskit.compiler import transpile

import numpy as np
from typing import List, Callable, Union

# myqlm functions
from qat.interop.qiskit import qiskit_to_qlm
from qat.core import Observable
from qat.lang.AQASM import Program, RY
from qat.qlmaas.result import AsyncResult
import time
#import misc.notebooks.uploader.my_junction as my_junction


class VQE_MyQLM_local(VQE):
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

                job = qcirc.to_job(observable=Observable(operator.num_qubits,
                                                         matrix=operator.to_matrix()))
                # START COMPUTATION
                result_temp = self.submit_job(job)

                if isinstance(result_temp, AsyncResult):  # chek if we are dealing with remote
                    while result_temp.get_status() != 'done':
                        time.sleep(.1)
                    result = result_temp.get_result()
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


def updateVQE_MyQLM(groundstatesolver):
    groundstatesolver.solver._vqe = VQE_MyQLM_local(ansatz=None,
                                                    quantum_instance=groundstatesolver.solver._quantum_instance,
                                                    optimizer=groundstatesolver.solver._optimizer,
                                                    initial_point=groundstatesolver.solver._initial_point,
                                                    gradient=groundstatesolver.solver._gradient,
                                                    expectation=groundstatesolver.solver._expectation,
                                                    include_custom=groundstatesolver.solver._include_custom)
    return groundstatesolver


def simple_qlm_job():
    prog = Program()
    qbits = prog.qalloc(1)
    prog.apply(RY(prog.new_var(float, r"\beta")), qbits)
    job = prog.to_circ().to_job(observable=Observable.sigma_z(0, 1))
    return job


# def build_QLM_stack(groundstatesolver, es_problem, qpu):
#     new_groundstatesolver = updateVQE_MyQLM(groundstatesolver)
#     plugin = my_junction.IterativeExplorationVQE(new_groundstatesolver, es_problem)
#     stack = plugin | qpu
#     return stack


def build_QLM_stack(groundstatesolver, molecule, plugin, qpu):
    # new_groundstatesolver = updateVQE_MyQLM(groundstatesolver)
    plugin_ready = plugin(method=groundstatesolver, molecule=molecule)
    stack = plugin_ready | qpu
    return stack


def run_QLM_stack(stack):
    solution = stack.submit(simple_qlm_job())
    qiskit_result = solution.meta_data['result']
    return qiskit_result
