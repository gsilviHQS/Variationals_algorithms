#from git import Object
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
import json



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

def run_QLM_stack(stack):
    solution = stack.submit(simple_qlm_job(),
                            meta_data={"optimal_parameters": "",
                                       "hartree_fock_energy": "",
                                       "qat": "",
                                       "num_iterations": "",
                                       "finishing_criterion": "",
                                       "raw_result": ""})
    if isinstance(solution, AsyncResult):  # chek if we are dealing with remote
        print('Waiting for remote job to complete....', end='\t')
        result = solution.join()
        print('done')
    else:
        result = solution
    return QiskitResult(result)
