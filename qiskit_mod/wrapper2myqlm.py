#from git import Object
from qiskit.algorithms import VQE
from qiskit.opflow import OperatorBase
from qiskit.compiler import transpile
from qiskit.algorithms.minimum_eigen_solvers import VQEResult

import numpy as np
from typing import List, Callable, Union

# myqlm functions
from qat.interop.qiskit import qiskit_to_qlm
from qat.core import Observable
from qat.lang.AQASM import Program, RY
from qat.qlmaas.result import AsyncResult
import time
#import misc.notebooks.uploader.my_junction as my_junction
import json
import io


def simple_qlm_job():
    prog = Program()
    qbits = prog.qalloc(1)
    prog.apply(RY(prog.new_var(float, r"\beta")), qbits)
    job = prog.to_circ().to_job(observable=Observable.sigma_z(0, 1))
    return job


def build_QLM_stack(groundstatesolver, molecule, plugin, qpu, shots=None, remove_orbitals=None,):
    plugin_ready = plugin(method=groundstatesolver, molecule=molecule, shots=shots, remove_orbitals=remove_orbitals)
    stack = plugin_ready | qpu
    return stack

class QiskitResult:
    def __init__(self, solution):
        self.total_energies = solution.value
        self.hartree_fock_energy = float(solution.meta_data['hartree_fock_energy'])
        setattr(self, 'raw_result', solution)
        self.raw_result.optimal_parameters = json.loads(solution.meta_data['optimal_parameters'])

def pack_result(qlm_result):
    result = VQEResult()

    result.optimal_point = np.load(io.BytesIO(qlm_result.meta_data['optimal_point']))
    result.optimal_parameters =json.loads(qlm_result.meta_data['optimal_parameters'])
    result.optimal_value = float(qlm_result.meta_data['optimal_value'])
    result.cost_function_evals = float(qlm_result.meta_data['cost_function_evals'])
    result.optimizer_time = float(qlm_result.meta_data['optimizer_time'])
    result.eigenvalue = complex(qlm_result.meta_data['eigenvalue'])
    result.eigenstate =  np.load(io.BytesIO(qlm_result.meta_data['eigenstate']))
    return result

def run_QLM_stack(stack):
    solution = stack.submit(simple_qlm_job(),
                            meta_data={"optimal_parameters": "",
                                       "hartree_fock_energy": "",
                                       "qiskit_version": "",
                                       "num_iterations": "",
                                       "finishing_criterion": "",
                                       "raw_result": "",
                                       "optimal point": "",
                                       "optimal_value": "",
                                       "cost_function_evals": "",
                                       "eigenvalue": "",
                                       "eigenstate": ""})
    if isinstance(solution, AsyncResult):  # chek if we are dealing with remote
        print('Waiting for remote job to complete....', end='\t')
        qlm_result = solution.join()
        print('done')
    else:
        qlm_result = solution
    
    return pack_result(qlm_result)
