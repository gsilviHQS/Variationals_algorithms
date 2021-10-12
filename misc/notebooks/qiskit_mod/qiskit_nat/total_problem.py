"""The Total Electronic and Nuclei Problem class."""
from functools import partial
from typing import cast, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit.algorithms import EigensolverResult, MinimumEigensolverResult
from qiskit.opflow import PauliSumOp
from qiskit.opflow.primitive_ops import Z2Symmetries

from qiskit_nature.circuit.library.initial_states.hartree_fock import hartree_fock_bitstring
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriver, QMolecule
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicStructureDriverResult,
    ParticleNumber,
)
from .total_driver_result import TotalDriverResult
from .total_result import TotalResult
from qiskit_nature.results import EigenstateResult
from .myqmolecule import myQMolecule
from qiskit_nature.transformers import BaseTransformer as LegacyBaseTransformer
from qiskit_nature.transformers.second_quantization import BaseTransformer

#from .builders.hopping_ops_builder import _build_qeom_hopping_ops
from qiskit_nature.problems.second_quantization.base_problem import BaseProblem


class TotalProblem(BaseProblem):
    """The Total Electronic and Nuclei Problem"""

    def __init__(
        self,
        driver: ElectronicStructureDriver,
        q_molecule_transformers: Optional[
            List[Union[LegacyBaseTransformer, BaseTransformer]]
        ] = None,
    ):
        """

        Args:
            driver: A fermionic driver encoding the molecule information.
            q_molecule_transformers: A list of transformations to be applied to the molecule.
        """
        super().__init__(driver, q_molecule_transformers)

    @property
    def num_particles(self) -> Tuple[int, int]:
        return self._properties_transformed.get_property("ParticleNumber").num_particles

    def second_q_ops(self) -> List[SecondQuantizedOp]:
        """Returns a list of `SecondQuantizedOp` created based on a driver and transformations
        provided.

        Returns:
            A list of `SecondQuantizedOp` in the following order: Hamiltonian operator,
            total particle number operator, total angular momentum operator, total magnetization
            operator, and (if available) x, y, z dipole operators.
        """
        tmp = cast(QMolecule, self.driver.run())
        qmol = myQMolecule()
        qmol.__dict__.update(tmp.__dict__)
        self._molecule_data = cast(myQMolecule, qmol)
        
        if self._legacy_transform:
            qmol_transformed = self._transform(self._molecule_data)
            self._properties_transformed = (
                TotalDriverResult.from_legacy_driver_result(qmol_transformed)
            )
        else:
            prop = TotalDriverResult.from_legacy_driver_result(self._molecule_data)
            self._properties_transformed = self._transform(prop)

        second_quantized_ops_list = self._properties_transformed.second_q_ops()

        return second_quantized_ops_list

    def hopping_qeom_ops(
        self,
        qubit_converter: QubitConverter,
        excitations: Union[
            str,
            int,
            List[int],
            Callable[[int, Tuple[int, int]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]],
        ] = "sd",
    ) -> Tuple[
        Dict[str, PauliSumOp],
        Dict[str, List[bool]],
        Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ]:
        """Generates the hopping operators and their commutativity information for the specified set
        of excitations.

        Args:
            qubit_converter: the `QubitConverter` to use for mapping and symmetry reduction. The
                             Z2 symmetries stored in this instance are the basis for the
                             commutativity information returned by this method.
            excitations: the types of excitations to consider. The simple cases for this input are:

                :`str`: containing any of the following characters: `s`, `d`, `t` or `q`.
                :`int`: a single, positive integer denoting the excitation type (1 == `s`, etc.).
                :`List[int]`: a list of positive integers.
                :`Callable`: a function which is used to generate the excitations.
                    For more details on how to write such a function refer to the default method,
                    :meth:`generate_fermionic_excitations`.

        Returns:
            A tuple containing the hopping operators, the types of commutativities and the
            excitation indices.
        """
        raise NotImplementedError()

    def interpret(
        self,
        raw_result: Union[EigenstateResult, EigensolverResult, MinimumEigensolverResult],
    ) -> TotalResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            raw_result: an eigenstate result object.

        Returns:
            An electronic structure result.
        """
        eigenstate_result = None
        if isinstance(raw_result, EigenstateResult):
            eigenstate_result = raw_result
        elif isinstance(raw_result, EigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = raw_result.eigenvalues
            eigenstate_result.eigenstates = raw_result.eigenstates
            eigenstate_result.aux_operator_eigenvalues = raw_result.aux_operator_eigenvalues
        elif isinstance(raw_result, MinimumEigensolverResult):
            eigenstate_result = EigenstateResult()
            eigenstate_result.raw_result = raw_result
            eigenstate_result.eigenenergies = np.asarray([raw_result.eigenvalue])
            eigenstate_result.eigenstates = [raw_result.eigenstate]
            eigenstate_result.aux_operator_eigenvalues = [raw_result.aux_operator_eigenvalues]
        result = TotalResult()
        result.combine(eigenstate_result)
        result.total_angular_momentum = None
        result.num_particles = None
        result.magnetization = None
        #not yet working
        #self._properties_transformed.interpret(result)
        result.computed_energies = np.asarray([e.real for e in eigenstate_result.eigenenergies])
        return result

    def get_default_filter_criterion(
        self,
    ) -> Optional[Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """Returns a default filter criterion method to filter the eigenvalues computed by the
        eigen solver. For more information see also
        qiskit.algorithms.eigen_solvers.NumPyEigensolver.filter_criterion.

        In the fermionic case the default filter ensures that the number of particles is being
        preserved.
        """

        raise NotImplementedError()

    def symmetry_sector_locator(self, z2_symmetries: Z2Symmetries) -> Optional[List[int]]:
        """Given the detected Z2Symmetries can determine the correct sector of the tapered
        operators so the correct one can be returned

        Args:
            z2_symmetries: the z2 symmetries object.

        Returns:
            The sector of the tapered operators with the problem solution.
        """
        raise NotImplementedError()

    @staticmethod
    def _pick_sector(z2_symmetries: Z2Symmetries, hf_str: List[bool]) -> List[int]:
        # Finding all the symmetries using the find_Z2_symmetries:
        raise NotImplementedError()
