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
# from qiskit.algorithms import VQE
# VQE.get_energy_evaluation = get_energy_evaluation_QLM
from qiskit.opflow import OperatorBase,StateFn,CircuitSampler,CircuitStateFn,PauliOp, PauliSumOp
from qiskit.compiler import transpile
from qiskit_nature.algorithms import VQEUCCFactory,GroundStateEigensolver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit.quantum_info import Pauli,SparsePauliOp
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.opflow import ExpectationBase, CircuitSampler, PauliExpectation, OperatorStateFn, CircuitStateFn, PauliOp, ListOp, SummedOp,ComposedOp
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance

import numpy as np
from typing import Callable, Iterable, List, Optional, Tuple, Union
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
                 shots = None,
                 remove_orbitals=[],
                 ):
        super(IterativeExplorationVQE, self).__init__()
        self.method = method
        if self.method._solver.gradient is not None:
            self.method._solver.gradient.nb_shots = shots
        self.method._solver._vqe.nb_shots = shots

        # create electronic structure problem
        driver = PySCFDriver(atom=molecule, unit=UnitsType.ANGSTROM, basis='sto3g')
        
        if 'q_molecule_transformers' in inspect.getfullargspec(ElectronicStructureProblem).args:
            self.problem = ElectronicStructureProblem(driver, q_molecule_transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=remove_orbitals)])
        else:
            self.problem = ElectronicStructureProblem(driver, transformers=[FreezeCoreTransformer(freeze_core=True, remove_orbitals=remove_orbitals)])

    def run(self, initial_job, meta_data):
        self.method._solver._vqe.execute = self.execute
        if self.method._solver.gradient is not None:
            self.method._solver.gradient.execute = self.execute
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
            if result.aux_operator_eigenvalues is not None:
                meta_data['aux_operator_eigenvalues'] = json.dumps(result.aux_operator_eigenvalues.tolist())
        meta_data['qiskit_version'] = str(qiskit.__qiskit_version__)#str(os.path.abspath(qat.__file__))
        

        return Result(value=result.eigenvalue, meta_data=meta_data)



##################################################################

def create_and_run_job(self,
                       operator_meas: OperatorBase,
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
                transpiled_circ = transpile(ansatz_in_use,
                                            basis_gates=LIST_OF_GATES,
                                            optimization_level=0)
                qcirc = qiskit_to_qlm(transpiled_circ)
                print('Transp',transpiled_circ)
                print('Circuit',qcirc)
                job = qcirc.to_job(observable=Observable(operator_meas.num_qubits, matrix=operator_meas.to_matrix()),nbshots=self.nb_shots) 
                # print('Shots= ',job.nbshots)
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

        

        def energy_evaluation(parameters):
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(zip(self._ansatz_params, parameter_sets.transpose().tolist())) # combine values to parameters symbols
            #print('param_bindings',param_bindings)
            sampled_expect_op = self._circuit_sampler.convert(expect_op, params=param_bindings)
            # print('Qiskit result EE',np.real(sampled_expect_op.eval()))
            means = np.real(create_and_run_job(self,reversed_operator, param_bindings)) #important to reverse the operator because of different conventions QLM/Qiskit
            # print('QLM result EE',means)
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

def extract_and_compute_circ(self, operator: OperatorBase,prev_coeff=1) -> None:
            """
            Recursively extract the ``CircuitStateFns`` contained in operator into the
            ``_circuit_ops_cache`` field.
            """
            # print(operator,'vs',operator.coeff)
            #print('Coeff',operator.coeff, type(operator))
            if isinstance(operator, ComposedOp):
               
                coefficient = prev_coeff*operator.coeff
                for i,op in enumerate(operator.oplist):
                    if isinstance(op,CircuitStateFn):
                        circuit = op.to_circuit_op()
                        #coefficient*=op.coeff
                    elif isinstance(op,OperatorStateFn):
                        operator_to_run = op
                        #coefficient*=op.coeff
                    elif isinstance(op,ListOp):
                        for op2 in op.oplist:
                            if isinstance(op2,CircuitStateFn):
                                circuit = op2.to_circuit_op()
                                #coefficient*=op2.coeff
                if circuit is not None and operator_to_run is not None:
                    value = run_circuit_in_QLM(self,circuit,operator_to_run, coefficient)
                    #print('Computed',value, 'with coefficient',coefficient)
                return value
                
            elif isinstance(operator, SummedOp):
                list_of_values=[]
                coefficient = prev_coeff*operator.coeff
                for i,op in enumerate(operator.oplist):
                    list_of_values.append(extract_and_compute_circ(self, op,prev_coeff=coefficient))
                summation= sum(list_of_values)
                #print('Computed sum',summation)
                return summation

            elif isinstance(operator, ListOp):
                list_of_values=[]
                
                coefficient = prev_coeff*operator.coeff
                #print(coefficient)
                for i,op in enumerate(operator.oplist):
                    list_of_values.append(extract_and_compute_circ(self, op, prev_coeff=coefficient))
                return list_of_values



def run_circuit_in_QLM(self,circuit,operator, coefficient):

    operator_pauli = operator._primitive.to_pauli_op()
    operator_meas_inv = PauliOp(primitive= Pauli(operator_pauli.primitive.to_label()[::-1]), coeff=1)#operator_pauli.coeff)
    transpiled_circ = transpile(circuit.to_circuit(),
                            basis_gates=LIST_OF_GATES,
                            optimization_level=0)
    qcirc = qiskit_to_qlm(transpiled_circ)
    job = qcirc.to_job(observable=Observable(operator_meas_inv.num_qubits, matrix=operator_meas_inv.to_matrix()),nbshots=self.nb_shots) 
    result_temp = self.execute(job)
    if isinstance(result_temp, AsyncResult):  # chek if we are dealing with remote job
        #case for local plugin/remote qpu
        result = result_temp.join()
        
    else:
        result = result_temp
    return coefficient * np.real(result.value)




def gradient_wrapper_for_QLM(
        self,
        operator: OperatorBase,
        bind_params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]],
        grad_params: Optional[
            Union[
                ParameterExpression,
                ParameterVector,
                List[ParameterExpression],
                Tuple[ParameterExpression, ParameterExpression],
                List[Tuple[ParameterExpression, ParameterExpression]],
            ]
        ] = None,
        backend: Optional[Union[BaseBackend, Backend, QuantumInstance]] = None,
        expectation: Optional[ExpectationBase] = None,
    ) -> Callable[[Iterable], np.ndarray]:
        """Get a callable function which provides the respective gradient, Hessian or QFI for given
        parameter values. This callable can be used as gradient function for optimizers.

        Args:
            operator: The operator for which we want to get the gradient, Hessian or QFI.
            bind_params: The operator parameters to which the parameter values are assigned.
            grad_params: The parameters with respect to which we are taking the gradient, Hessian
                or QFI. If grad_params = None, then grad_params = bind_params
            backend: The quantum backend or QuantumInstance to use to evaluate the gradient,
                Hessian or QFI.
            expectation: The expectation converter to be used. If none is set then
                `PauliExpectation()` is used.

        Returns:
            Function to compute a gradient, Hessian or QFI. The function
            takes an iterable as argument which holds the parameter values.
        """
        from qiskit.opflow.converters import CircuitSampler


        if not grad_params:
            grad_params = bind_params

        grad = self.convert(operator, grad_params)
        if expectation is None:
            expectation = PauliExpectation()
        grad = expectation.convert(grad)
        #print(grad)
        #print('General operator',operator)
        #print('bind_params',bind_params)


        def gradient_fn(p_values):
            p_values_dict = dict(zip(bind_params, p_values))
            if not backend:
                converter = grad.assign_parameters(p_values_dict)
                return np.real(converter.eval())
            else:
                p_values_dict = {k: [v] for k, v in p_values_dict.items()} # remake the dict with list of values
                # converter = grad.assign_parameters(p_values_dict) # assign the values to the parameters
                #print(converter)
                #converted_circ = expectation.convert(grad)
                circuit_assigned = grad.assign_parameters(p_values_dict) # assign the values to the parameters
                dictio = extract_and_compute_circ(self,circuit_assigned) 
                #print('LIST',dictio[0], dictio[1])
                
                # set up the matrix here
                val = dictio[1]
                matrix = np.zeros((len(val),len(val)))
                upper_w_diag = np.triu_indices(len(val))
                matrix[upper_w_diag] = np.array(np.concatenate(val))
                i_upper = np.triu_indices(len(val), 1)
                i_lower = np.tril_indices(len(val), -1)
                matrix[i_lower] = matrix[i_upper]

                
                converter = CircuitSampler(backend=backend).convert(grad, p_values_dict)
                #print('Qiskit first 5 circuits',[op.eval() for op in converter[0][0][0].oplist])
                #print('Qiskit first sums',[op.eval() for op in converter[0][0].oplist])
                QLM_result = converter[0].combo_fn([dictio[0],matrix])
                # print('QLM_result',QLM_result)
                # print('Qiskit result',np.real(converter.eval()[0]))

                return QLM_result
                

        return gradient_fn