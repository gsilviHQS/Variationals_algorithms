"""The Variational Quantum Imaginary/Real bolution.

See https://arxiv.org/abs/1304.3061
"""

from typing import Optional, List#, Callable, Union, Dict, Tuple
import logging
from time import time
import numpy as np

from qiskit.opflow import (
    OperatorBase,
    StateFn,
    I,
)
from qiskit.opflow.gradients import GradientBase, Gradient, QFI, NaturalGradient
from qiskit.algorithms.minimum_eigen_solvers import VQE, MinimumEigensolverResult, VQEResult
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_initial_point, _validate_bounds
from qiskit.algorithms.exceptions import AlgorithmError
from .lin_comb_mod import LinCombMod
from .lin_comb_full_mod import LinCombFullmod

logger = logging.getLogger(__name__)

# disable check for ansatzes, optimizer setter because of pylint bug
# pylint: disable=no-member

class EvoVQE(VQE):
    r"""VQE class plus Real/Imaginary time evolution algorithm"""

    def compute_evolve(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None,
        isteps = 10, rsteps = 30, di= 0.5, dr = 0.05
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)

        if self.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend must be supplied to run the quantum algorithm."
            )
        self.quantum_instance.circuit_summary = True

        # this sets the size of the ansatz, so it must be called before the initial point
        # validation
        self._check_operator_ansatz(operator)

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)

        bounds = _validate_bounds(self.ansatz)

        # We need to handle the array entries being Optional i.e. having value None
        if aux_operators:
            zero_op = I.tensorpower(operator.num_qubits) * 0.0
            converted = []
            for op in aux_operators:
                if op is None:
                    converted.append(zero_op)
                else:
                    converted.append(op)

            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in converted]
        else:
            aux_operators = None

        self._eval_count = 0

        energy_evaluation, expectation = self.get_energy_evaluation(
            operator, return_expectation=True
        )

        #opt_params = [0.2]*self.ansatz.num_parameters
        opt_params = initial_point

        print("initial parameters\n",opt_params)

        # SETUP NATURAL GRADIENT
        #grad = NaturalGradient(grad_method='lin_comb', imaginary=False, 
        #                        qfi_method='lin_comb_full', regularization='ridge')
        grad = NaturalGradient(grad_method=LinCombMod(img=False), 
                                qfi_method=LinCombFullmod(), regularization='ridge')
        grad = grad.gradient_wrapper(
                     #~StateFn(operator) @ StateFn(self._ansatz),
                     ~StateFn(operator) @ StateFn(self._ansatz.decompose()),
                     bind_params=self._ansatz_params,
                     backend=self._quantum_instance)

        start_time = time()

        print("IMAGINARY TIME EVOLUTION")
        weight = di
        opt_value = 0.
        nfev = 0
        print("dt=> {} initial energy=> {}".format(weight, 
                                 energy_evaluation(opt_params)))

        for nfev in range(0,isteps):
            grd = grad(opt_params)
            norm= np.linalg.norm(grd)
            #scale = weight/norm if norm > 1.0 else weight
            #if (nfev%10 == 0) : print("NATURL GRADIENT:\n",grd)
            #opt_params -= scale*grd
            opt_params -= weight*grd

            opt_value = energy_evaluation(opt_params)
            print("time=> {0:4.2f} energy=> {1:} norm=> {2:}".format((nfev+1)*weight, opt_value, norm))

        # SETUP NATURAL GRADIENT
        #grad = NaturalGradient(grad_method='lin_comb', imaginary=True, 
        #                        qfi_method='lin_comb_full', regularization='ridge')
        grad = NaturalGradient(grad_method=LinCombMod(img=True), 
                                qfi_method=LinCombFullmod(), regularization='ridge')
        grad = grad.gradient_wrapper(
                     #~StateFn(operator) @ StateFn(self._ansatz),
                     ~StateFn(operator) @ StateFn(self._ansatz.decompose()),
                     bind_params=self._ansatz_params,
                     backend=self._quantum_instance)

        print("parameters\n",opt_params)
        print("REAL TIME EVOLUTION")
        weight = dr
        print("dt=> {} initial energy=> {}".format(weight, opt_value))
        for nfev in range(0,rsteps):
            grd = grad(opt_params)
            norm= np.linalg.norm(grd)
            #if (nfev%10 == 0) : print("NATURL GRADIENT:\n",grd)
            opt_params += weight*grd

            opt_value = energy_evaluation(opt_params)
            #print("time=> {0:4.2f} energy=> {1:}".format(((nfev+1)*weight), opt_value))
            print("time=> {0:4.2f} energy=> {1:} norm=> {2:}".format((nfev+1)*weight, opt_value, norm))

        eval_time = time() - start_time

        result = VQEResult()
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self._ansatz_params, opt_params))
        result.optimal_value = opt_value
        result.cost_function_evals = nfev
        result.optimizer_time = eval_time
        result.eigenvalue = opt_value + 0j
        result.eigenstate = self._get_eigenstate(result.optimal_parameters)

        logger.info(
            "Evolution complete in %s seconds.\nFound opt_params %s in %s evals",
            eval_time,
            result.optimal_point,
            self._eval_count,
        )

        # TODO delete as soon as get_optimal_vector etc are removed
        self._ret = result

        if aux_operators is not None:
            aux_values = self._eval_aux_ops(opt_params, aux_operators, expectation=expectation)
            result.aux_operator_eigenvalues = aux_values[0]

        return result
