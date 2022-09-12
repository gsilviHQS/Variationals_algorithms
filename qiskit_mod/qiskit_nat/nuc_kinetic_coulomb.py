"""The Nuclear Kinetic Coulomb Energy property."""

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
from .one_body_nuclear_integrals import OneBodyNuclearIntegrals
from .two_body_nuclear_integrals import TwoBodyNuclearIntegrals


class NuclearKineticCoulomb(IntegralProperty):
    """The Nuclear Kinetic and Cuolomb Interaction Energy property."""

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
        """Construct an NuclearKineticCoulomb instance from a myQMolecule.

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
        if qmol.nuc_kinetic is not None:
            integrals.append(
                #OneBodyNuclearIntegrals(ElectronicBasis.MO, (qmol.oneeints2mo(qmol.kinetic, qmol.mo_coeff), None), shift = 2*len(qmol.mo_coeff))
                OneBodyNuclearIntegrals(ElectronicBasis.MO, (qmol.nuc_kinetic, None), shift = 2*len(qmol.el_kinetic))
            )
        else:
            raise ValueError("Kinetic nuclear integrals must be provided")

        if qmol.nuc_eri is not None:
            integrals.append(
                #TwoBodyNuclearIntegrals(ElectronicBasis.MO, (qmol.mo_eri_ints, None, None, None), shift = 2*len(qmol.mo_coeff))
                TwoBodyNuclearIntegrals(ElectronicBasis.MO, (qmol.nuc_eri, None, None, None), shift = 2*len(qmol.el_kinetic))
            )
        else:
            raise ValueError("Coulomb nuclear integrals must be provided")

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
