"""The TotalDriverResult class."""

from typing import List, Tuple, cast, Union

from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import QMolecule
from .myqmolecule import myQMolecule
from qiskit_nature.operators.second_quantization import FermionicOp

from qiskit_nature.properties.second_quantization.driver_metadata import DriverMetadata
from qiskit_nature.properties.second_quantization.second_quantized_property import LegacyDriverResult, LegacyElectronicStructureDriverResult
from qiskit_nature.properties.second_quantization.electronic.angular_momentum import AngularMomentum
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.properties.second_quantization.electronic.dipole_moment import ElectronicDipoleMoment
from qiskit_nature.properties.second_quantization.electronic.electronic_energy import ElectronicEnergy
from qiskit_nature.properties.second_quantization.electronic.magnetization import Magnetization
from qiskit_nature.properties.second_quantization.electronic.particle_number import ParticleNumber
from .kinetic_coulomb import ElectronicKineticCoulomb
from .mixed_coulomb import ElectronicNuclearCoulomb
from .nuc_kinetic_coulomb import NuclearKineticCoulomb
from qiskit_nature.properties.second_quantization.electronic.types import GroupedElectronicProperty

LegacyTotalDriverResult = Union[LegacyDriverResult, myQMolecule]

class TotalDriverResult(GroupedElectronicProperty):
    """The TotalDriverResult class.

    This is a :class:~qiskit_nature.properties.GroupedProperty gathering all property objects
    previously stored in Qiskit Nature's `myQMolecule` object.
    """

    def __init__(self) -> None:
        """
        Property objects should be added via `add_property` rather than via the initializer.
        """
        super().__init__(self.__class__.__name__)
        self.molecule: Molecule = None

    @classmethod
    def from_legacy_driver_result(
        cls, result: LegacyTotalDriverResult
    ) -> "TotalDriverResult":
        """Converts a myQMolecule into an `TotalDriverResult`.

        Args:
            result: the myQMolecule to convert.

        Returns:
            An instance of this property.

        Raises:
            QiskitNatureError: if a WatsonHamiltonian is provided.
        """
        cls._validate_input_type(result, LegacyTotalDriverResult)

        ret = cls()

        qmol = cast(myQMolecule, result)

        ret.add_property(ElectronicEnergy.from_legacy_driver_result(qmol))
        ret.add_property(ParticleNumber.from_legacy_driver_result(qmol))
        ret.add_property(AngularMomentum.from_legacy_driver_result(qmol))
        ret.add_property(Magnetization.from_legacy_driver_result(qmol))
        ret.add_property(ElectronicDipoleMoment.from_legacy_driver_result(qmol))
        ret.add_property(ElectronicKineticCoulomb.from_legacy_driver_result(qmol))
        ret.add_property(ElectronicNuclearCoulomb.from_legacy_driver_result(qmol))
        ret.add_property(NuclearKineticCoulomb.from_legacy_driver_result(qmol))

        ret.add_property(
            ElectronicBasisTransform(
                ElectronicBasis.AO, ElectronicBasis.MO, qmol.mo_coeff, qmol.mo_coeff_b
            )
        )

        geometry: List[Tuple[str, List[float]]] = []
        for atom, xyz in zip(qmol.atom_symbol, qmol.atom_xyz):
            geometry.append((atom, xyz))

        ret.molecule = Molecule(geometry, qmol.multiplicity, qmol.molecular_charge)

        ret.add_property(
            DriverMetadata(
                qmol.origin_driver_name,
                qmol.origin_driver_version,
                qmol.origin_driver_config,
            )
        )

        return ret

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns the list of `FermioncOp`s given by the properties contained in this one."""
        ops: List[FermionicOp] = []
        # TODO: make aux_ops a Dict? Then we don't need to hard-code the order of these properties.
        # NOTE: this will also get rid of the hard-coded aux_operator eigenvalue indices in the
        # `interpret` methods of all of these properties

        total=FermionicOp.zero(1)
        for cls in [
            ElectronicEnergy,
            ParticleNumber,
            AngularMomentum,
            Magnetization,
            ElectronicDipoleMoment,
            ElectronicKineticCoulomb,
            NuclearKineticCoulomb,
            ElectronicNuclearCoulomb,
        ]:
            prop = self.get_property(cls)  # type: ignore
            if prop is None:
                continue

            print("detected=> ",cls.__name__)
            op = prop.second_q_ops()
            if cls in [ElectronicKineticCoulomb, NuclearKineticCoulomb, ElectronicNuclearCoulomb]:
               print("   accumulating=> ",cls.__name__)
               print(op[0])
               total = sum(op, total)

            ops.extend(op)
        ops.append(total)
        print("extended by=> TotalEnergy")
        return ops
