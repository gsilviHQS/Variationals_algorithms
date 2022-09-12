"""The Electronic Kinetic and Cuolomb-Interaction Energy property."""

from typing import Dict, List, Optional, cast

import numpy as np

from qiskit_nature.drivers.second_quantization import QMolecule
from .myqmolecule import myQMolecule
from qiskit_nature.results import EigenstateResult

from qiskit_nature.properties.second_quantization.second_quantized_property import LegacyDriverResult, LegacyElectronicStructureDriverResult
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    IntegralProperty,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)
from .my_two_body_electronic_integrals import MyTwoBodyElectronicIntegrals


class ElectronicKineticCoulomb(IntegralProperty):
    """The Kinetic and Cuolomb Interaction Energy property."""

    def __init__(
        self,
        electronic_integrals: List[ElectronicIntegrals],
        energy_shift: Optional[Dict[str, complex]] = None,
        nuclear_repulsion_energy: Optional[complex] = None,
        reference_energy: Optional[complex] = None,
    ):
        """
        Args:
            basis: the basis which the integrals in ``electronic_integrals`` are stored in.
            electronic_integrals: a dictionary mapping the ``# body terms`` to the corresponding
                ``ElectronicIntegrals``.
            reference_energy: an optional reference energy (such as the HF energy).
            energy_shift: an optional dictionary of energy shifts.
        """
        super().__init__(self.__class__.__name__, electronic_integrals, shift=energy_shift)

    @classmethod
    def from_legacy_driver_result(cls, result: LegacyDriverResult) -> "ElectronicEnergy":
        """Construct an Kinetic and Cuolomb Interaction Energy property instance from a myQMolecule.

        Args:
            result: the driver result from which to extract the raw data. For this property, a
                myQMolecule is required!

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, LegacyElectronicStructureDriverResult)

        qmol = cast(myQMolecule, result)

        energy_shift = qmol.energy_shift.copy()

        integrals: List[ElectronicIntegrals] = []
        if qmol.el_kinetic is not None:
            integrals.append(
                #OneBodyElectronicIntegrals(ElectronicBasis.MO, (qmol.oneeints2mo(qmol.kinetic, qmol.mo_coeff), None))
                OneBodyElectronicIntegrals(ElectronicBasis.MO, (qmol.el_kinetic, None))
            )
        else:
            raise ValueError("Kinetic integrals must be provided")

        #if qmol.eri is not None:
        if qmol.el_eri is not None:
            integrals.append(
                #TwoBodyElectronicIntegrals(ElectronicBasis.MO, (qmol.mo_eri_ints, None, None, None))
                #TwoBodyElectronicIntegrals(ElectronicBasis.MO, (qmol.el_eri, None, None, None))
                MyTwoBodyElectronicIntegrals(ElectronicBasis.MO, (qmol.el_eri, None, None, None))
            )

        ret = cls(
            integrals,
            #energy_shift=energy_shift,
            energy_shift=None,
            #nuclear_repulsion_energy=qmol.nuclear_repulsion_energy,
            nuclear_repulsion_energy=None,
            reference_energy=qmol.hf_energy,
        )

        return ret

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:~qiskit_nature.result.EigenstateResult in this property's context.

        Args:
            result: the result to add meaning to.
        """
        result.hartree_fock_energy = self._reference_energy
        result.nuclear_repulsion_energy = self._nuclear_repulsion_energy
        result.extracted_transformer_energies = self._shift.copy()
