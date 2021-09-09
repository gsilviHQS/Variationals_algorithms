"""
The Variational Hamiltonia Ansatz
"""

from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import logging

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliTrotterEvolution
from qiskit_nature import QiskitNatureError
from qiskit_nature.operators.second_quantization import FermionicOp, SecondQuantizedOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.circuit.library.ansatzes.utils.fermionic_excitation_generator import generate_fermionic_excitations
# from .evolved_operator_ansatz import EvolvedOperatorAnsatz
# from .utils.fermionic_excitation_generator import generate_fermionic_excitations

logger = logging.getLogger(__name__)


class VHA(EvolvedOperatorAnsatz):
    r"""The Variational Hamiltonia Ansatz For more information, see [1].

    This Ansatz is an :class:`~.EvolvedOperatorAnsatz` given by :math:`e^{T - T^{\dagger}}` where
    :math:`T` is the *cluster operator*. This cluster operator generally consists all operators.

    A utility class :class:`UCCSD` exists, which is equivalent to:

    .. code-block:: python

        uccsd = UCC(excitations='sd', alpha_spin=True, beta_spin=True, max_spin_excitation=None)

    If you want to use a tailored Ansatz, you have multiple options to do so. Below, we provide some
    examples:

    .. code-block:: python

        # pure single excitations (equivalent options):
        uccs = UCC(excitations='s')
        uccs = UCC(excitations=1)
        uccs = UCC(excitations=[1])

        # pure double excitations (equivalent options):
        uccd = UCC(excitations='d')
        uccd = UCC(excitations=2)
        uccd = UCC(excitations=[2])

        # combinations of excitations:
        custom_ucc_sd = UCC(excitations='sd')  # see also the convenience sub-class UCCSD
        custom_ucc_sd = UCC(excitations=[1, 2])  # see also the convenience sub-class UCCSD
        custom_ucc_sdt = UCC(excitations='sdt')
        custom_ucc_sdt = UCC(excitations=[1, 2, 3])
        custom_ucc_st = UCC(excitations='st')
        custom_ucc_st = UCC(excitations=[1, 3])

        # you can even define a fully custom list of excitations:

        def custom_excitation_list(num_spin_orbitals: int,
                                   num_particles: Tuple[int, int]
                                   ) -> List[Tuple[Tuple[Any, ...], ...]]:
            # generate your list of excitations...
            my_excitation_list = [...]
            # For more information about the required format of the return statement, please take a
            # look at the documentation of
            # `qiskit_nature.circuit.library.ansatzes.utils.fermionic_excitation_generator`
            return my_excitation_list

        my_custom_ucc = UCC(excitations=custom_excitation_list)

    Keep in mind, that in all of the examples above we have not set any of the following keyword
    arguments, which must be specified before the Ansatz becomes usable:
    - `qubit_converter`
    - `num_particles`
    - `num_spin_orbitals`

    If you are using this Ansatz with a Qiskit Nature algorithm, these arguments will be set for
    you, depending on the rest of the stack.


    References:

        [1] https://arxiv.org/abs/1805.04340
    """

    EXCITATION_TYPE = {
        's': 1,
        'd': 2,
        't': 3,
        'q': 4,
    }

    def __init__(self, qubit_converter: Optional[QubitConverter] = None,
                 num_particles: Optional[Tuple[int, int]] = None,
                 num_spin_orbitals: Optional[int] = None,
                 excitations: Optional[Union[str, int, List[int],
                                             Callable[[int, Tuple[int, int]],
                                                      List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
                                             ]] = None,
                 alpha_spin: bool = True,
                 beta_spin: bool = True,
                 max_spin_excitation: Optional[int] = None,
                 reps: int = 1,
                 initial_state: Optional[QuantumCircuit] = None,
                 trotter_steps: int = 1,
                 only_excitations: bool=False,
                 splitting: bool=False):
        """

        Args:
            qubit_converter: the QubitConverter instance which takes care of mapping a
                :class:`~.SecondQuantizedOp` to a :class:`PauliSumOp` as well as performing all
                configured symmetry reductions on it.
            num_particles: the tuple of the number of alpha- and beta-spin particles.
            num_spin_orbitals: the number of spin orbitals.
            excitations: this can be any of the following types:

                :`str`: which contains the types of excitations. Allowed characters are
                    + `s` for singles
                    + `d` for doubles
                    + `t` for triples
                    + `q` for quadruples
                :`int`: a single, positive integer which denotes the number of excitations
                    (1 == `s`, etc.)
                :`List[int]`: a list of positive integers generalizing the above
                :`Callable`: a function which is used to generate the excitations.
                    The callable must take the __keyword__ arguments `num_spin_orbitals` and
                    `num_particles` (with identical types to those explained above) and must return
                    a `List[Tuple[Tuple[int, ...], Tuple[int, ...]]]`. For more information on how
                    to write such a callable refer to the default method
                    :meth:`generate_fermionic_excitations`.
            alpha_spin: boolean flag whether to include alpha-spin excitations.
            beta_spin: boolean flag whether to include beta-spin excitations.
            max_spin_excitation: the largest number of excitations within a spin. E.g. you can set
                this to 1 and `num_excitations` to 2 in order to obtain only mixed-spin double
                excitations (alpha,beta) but no pure-spin double excitations (alpha,alpha or
                beta,beta).
            reps: The number of times to repeat the evolved operators.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
            trotter_steps: number of repeat list of operators to pass to the variational algorithm
            only_excitations: limit the operators generated to only excitations
            splitting: if True it split the Hamiltonian to commuting terms, default is False
        """
        self._qubit_converter = qubit_converter
        self._num_particles = num_particles
        self._num_spin_orbitals = num_spin_orbitals
        self._excitations = excitations
        self._alpha_spin = alpha_spin
        self._beta_spin = beta_spin
        self._max_spin_excitation = max_spin_excitation
        self._trotter_steps=trotter_steps
        self._only_excitations=only_excitations
        self._splitting=splitting

        super().__init__(reps=reps, evolution=PauliTrotterEvolution(), initial_state=initial_state)

        # We cache these, because the generation may be quite expensive (depending on the generator)
        # and the user may want quick access to inspect these. Also, it speeds up testing for the
        # same reason!
        self._excitation_ops: List[SecondQuantizedOp] = None

    @property
    def qubit_converter(self) -> QubitConverter:
        """The qubit operator converter."""
        return self._qubit_converter

    @qubit_converter.setter
    def qubit_converter(self, conv: QubitConverter) -> None:
        """Sets the qubit operator converter."""
        self._invalidate()
        self._qubit_converter = conv

    @property
    def num_spin_orbitals(self) -> int:
        """The number of spin orbitals."""
        return self._num_spin_orbitals

    @num_spin_orbitals.setter
    def num_spin_orbitals(self, n: int) -> None:
        """Sets the number of spin orbitals."""
        self._invalidate()
        self._num_spin_orbitals = n

    @property
    def num_particles(self) -> Tuple[int, int]:
        """The number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, n: Tuple[int, int]) -> None:
        """Sets the number of particles."""
        self._invalidate()
        self._num_particles = n

    @property
    def excitations(self) -> Union[str, int, List[int], Callable]:
        """The excitations."""
        return self._excitations

    @excitations.setter
    def excitations(self, exc: Union[str, int, List[int], Callable]) -> None:
        """Sets the excitations."""
        self._invalidate()
        self._excitations = exc

    def _invalidate(self):
        self._excitation_ops = None
        super()._invalidate()

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        if self.num_spin_orbitals < 0:
            if raise_on_failure:
                raise ValueError('The number of spin orbitals cannot be smaller than 0.')
            return False

        if any(n < 0 for n in self.num_particles):
            if raise_on_failure:
                raise ValueError('The number of particles cannot be smaller than 0.')
            return False

        if self.excitations is None:
            if raise_on_failure:
                raise ValueError('The excitations cannot be `None`.')
            return False

        if self.qubit_converter is None:
            if raise_on_failure:
                raise ValueError('The qubit_converter cannot be `None`.')
            return False

        return True

    def _build(self) -> None:
        if self._data is not None:
            return

        if self.operators is None or self.operators == [None]:
            # The qubit operators are cached by the `EvolvedOperatorAnsatz` class. We only generate
            # them from the `SecondQuantizedOp`s produced by the generators, if they are not already
            # present. This behavior also enables the adaptive usage of the `UCC` class by
            # algorithms such as `AdaptVQE`.
            excitation_ops = self.excitation_ops()

            logger.debug('Converting SecondQuantizedOps into PauliSumOps...')
            # Convert operators according to saved state in converter from the conversion of the
            # main operator since these need to be compatible. If Z2 Symmetry tapering was done
            # it may be that one or more excitation operators do not commute with the
            # symmetry. Normally the converted operators are maintained at the same index by
            # the converter inserting None as the result if an operator did not commute. Here
            # we are not interested in that just getting the valid set of operators so that
            # behavior is suppressed.
            self.operators = self.qubit_converter.convert_match(excitation_ops, suppress_none=True)

        logger.debug('Building QuantumCircuit...')
        super()._build()

    def excitation_ops(self) -> List[SecondQuantizedOp]:
        """Parses the excitations and generates the list of operators.

        Raises:
            QiskitNatureError: if invalid excitations are specified.

        Returns:
            The list of generated excitation operators.
        """
        if self._excitation_ops is not None:
            return self._excitation_ops

        excitations = self._get_excitation_list()

        logger.debug('Converting excitations into SecondQuantizedOps...')
        excitation_ops = self._build_fermionic_excitation_ops(excitations)

        self._excitation_ops = excitation_ops
        return excitation_ops

    def _get_excitation_list(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        generators = self._get_excitation_generators()

        logger.debug('Generating excitation list...')
        excitations = []
        for gen in generators:
            excitations.extend(gen(
                num_spin_orbitals=self.num_spin_orbitals,
                num_particles=self.num_particles
            ))

        return excitations

    def _get_excitation_generators(self) -> List[Callable]:
        logger.debug('Gathering excitation generators...')
        generators: List[Callable] = []

        extra_kwargs = {'alpha_spin': self._alpha_spin,
                        'beta_spin': self._beta_spin,
                        'max_spin_excitation': self._max_spin_excitation}

        if isinstance(self.excitations, str):
            for exc in self.excitations:
                generators.append(partial(
                    generate_fermionic_excitations,
                    num_excitations=self.EXCITATION_TYPE[exc],
                    **extra_kwargs
                ))
        elif isinstance(self.excitations, int):
            generators.append(partial(
                generate_fermionic_excitations,
                num_excitations=self.excitations,
                **extra_kwargs
            ))
        elif isinstance(self.excitations, list):
            for exc in self.excitations:  # type: ignore
                generators.append(partial(
                    generate_fermionic_excitations,
                    num_excitations=exc,
                    **extra_kwargs
                ))
        elif callable(self.excitations):
            generators = [self.excitations]
        else:
            raise QiskitNatureError("Invalid excitation configuration: {}".format(self.excitations))

        return generators
    def _chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    def _build_fermionic_excitation_ops(self, excitations: Sequence) -> List[FermionicOp]:
        """Builds all possible excitation operators with the given number of excitations for the
        specified number of particles distributed in the number of orbitals.

        Args:
            excitations: the list of excitations.

        Returns:
            The list of excitation operators in the second quantized formalism.
        """
        
        operators = []
        H_diag = None 
        H_hop = []
        H_ex = []


        
        
        for i in range(0,self.num_spin_orbitals): #make the number operators (N) terms
            label = ['I'] * self.num_spin_orbitals
            label[i]= 'N'
            op = FermionicOp(''.join(label))
            #op *= 1j
            if H_diag == None:
                H_diag = op
            else:
                H_diag += op

        for p in range(0,self.num_spin_orbitals): #make the 2 number operators (N) terms
            for q in range(p+1,self.num_spin_orbitals):
                label = ['I'] * self.num_spin_orbitals
                label[p]= 'N'
                label[q]= 'N'
                op = FermionicOp(''.join(label))
                H_diag += op

        



        if self._only_excitations==True: # True=produce only single and double excitation, like the UCCSD
            #H_hop
            for exc in excitations: 
                if len(exc[0])==1: #single
                    label = ['I'] * self.num_spin_orbitals
                    for occ in exc[0]:
                        label[occ] = '+'
                    for unocc in exc[1]:
                        label[unocc] = '-'
                    op = FermionicOp(''.join(label))
                    op -= op.adjoint() #to respect antihermitian, if you subtract the adjoint, multiply by 1j in the line below
                    op *= 1j
                    H_hop.append(op)   
            #H_ex
            for exc in excitations: 
                if len(exc[0])==2: #double
                    label = ['I'] * self.num_spin_orbitals
                    for occ in exc[0]:
                        label[occ] = '+'
                    for unocc in exc[1]:
                        label[unocc] = '-'
                    op = FermionicOp(''.join(label))
                    op -= op.adjoint()
                    op *= 1j
                    H_ex.append(op)
        else: #all possible combinations
        
            #H_hop
            for p in range(0,self.num_spin_orbitals): 
                for q in range(p+1,self.num_spin_orbitals):
                    label = ['I'] * self.num_spin_orbitals
                    label[p] = '+'
                    label[q] = '-'
                    op = FermionicOp(''.join(label))
                    op -= op.adjoint()
                    op *= 1j
                    H_hop.append(op)   
                    
                    for r in range(0,self.num_spin_orbitals):
                        pqlabel=label[:]
                        if r!=p and r!=q :
                            pqlabel[r]= 'N'
                            op = FermionicOp(''.join(pqlabel))
                            op -= op.adjoint()
                            op *= 1j
                            H_hop.append(op)   
            
                    
            #H_ex
    
            for p in range(0,self.num_spin_orbitals):
                for q in range(0,self.num_spin_orbitals):
                    for r in range(p+1,self.num_spin_orbitals):
                        for s in range(q+1,self.num_spin_orbitals):
                            if p not in [q,r,s] and q not in [r,s] and r!=s:
                                
                                label = ['I'] * self.num_spin_orbitals
                                label[p] = '+'
                                label[q] = '+'
                                label[r] = '-'
                                label[s] = '-'
                                op = FermionicOp(''.join(label))
                                op -= op.adjoint()
                                op *= 1j
                                H_ex.append(op)
                  

        if self._splitting==True: #split the H_hop and H_ex in commuting terms
            
            for H_to_consider in [H_hop,H_ex]:
                copyH=H_to_consider[:] #copy all b
                
                for op in copyH:
                    labA=op._labels[0]
                    for i,op2 in enumerate(copyH):
                        labB=op2._labels[0]
                        bitw_res=[ord(a) & ord(b) for a,b in zip(labA,labB)] #make a bitwise to see if there is match in position of + or - (43=++ or 45=-- or 41=+-/-+)
                        if 43 not in bitw_res and 45 not in bitw_res and 41 not in bitw_res : #if true put them together
                            op+=op2
                            labA=[chr(a) for a in bitw_res]
                            for j,c in enumerate(labA):
                                if c=='\t': labA[j]='+' #merge labels + and -
                            copyH.pop(i)
                    operators.append(op)
            # for n in range(self._trotter_steps):
            #     chunck=int(len(H_hop)/(n+1))
            #     H_comb=[ sum(H_hop[i:i+chunck]) for i in range(0, len(H_hop), chunck)]
            #     operators.extend(H_comb)


            #     chunck=int(len(H_ex)/(n+1))
            #     H_comb=[ sum(H_ex[i:i+chunck]) for i in range(0, len(H_ex), chunck)]
            #     operators.extend(H_comb)
            
            operators.append(H_diag)

        else:
            
            for n in range(self._trotter_steps):
                operators.append(sum(H_ex))
                operators.append(sum(H_hop))
                operators.append(H_diag)


        
        
            



        return operators

    