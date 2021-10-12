"""The total result."""

from functools import reduce
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

from qiskit_nature.constants import DEBYE
from qiskit_nature.results.electronic_structure_result import ElectronicStructureResult

# A dipole moment, when present as X, Y and Z components will normally have float values for all
# the components. However when using Z2Symmetries, if the dipole component operator does not
# commute with the symmetry then no evaluation is done and None will be used as the 'value'
# indicating no measurement of the observable took place
DipoleTuple = Tuple[Optional[float], Optional[float], Optional[float]]


class TotalResult(ElectronicStructureResult):
    """The total result."""

    def formatted(self) -> List[str]:
        """Formatted result as a list of strings"""
        lines = []
        lines.append("=== GROUND STATE ENERGY ===")
        lines.append(" ")
        lines.append(
            "* Total ground state energy (Hartree): {}".format(
                round(self.electronic_energies[0], 12)
            )
        )
        lines.append("  - computed part:      {}".format(round(self.computed_energies[0], 12)))
        for name, value in self.extracted_transformer_energies.items():
            lines.append("  - {} extracted energy part: {}".format(name, round(value, 12)))
        if self.nuclear_repulsion_energy is not None:
            lines.append(
                "~ Nuclear repulsion energy (Hartree): {}".format(
                    round(self.nuclear_repulsion_energy, 12)
                )
            )
            lines.append(
                "> Total ground state energy (Hartree): {}".format(
                    round(self.total_energies[0], 12)
                )
            )

        if len(self.computed_energies) > 1:
            lines.append(" ")
            lines.append("=== EXCITED STATE ENERGIES ===")
            lines.append(" ")
            for idx, (elec_energy, total_energy) in enumerate(
                zip(self.electronic_energies[1:], self.total_energies[1:])
            ):
                lines.append("{: 3d}: ".format(idx + 1))
                lines.append(
                    "* Electronic excited state energy (Hartree): {}".format(round(elec_energy, 12))
                )
                lines.append(
                    "> Total excited state energy (Hartree): {}".format(round(total_energy, 12))
                )

        if self.has_observables():
            lines.append(" ")
            lines.append("=== MEASURED OBSERVABLES ===")
            lines.append(" ")
            for idx, (num_particles, spin, total_angular_momentum, magnetization,) in enumerate(
                zip(
                    self.num_particles,
                    self.spin,
                    self.total_angular_momentum,
                    self.magnetization,
                )
            ):
                line = "{: 3d}: ".format(idx)
                if num_particles is not None:
                    line += " # Particles: {:.3f}".format(num_particles)
                if spin is not None:
                    line += " S: {:.3f}".format(spin)
                if total_angular_momentum is not None:
                    line += " S^2: {:.3f}".format(total_angular_momentum)
                if magnetization is not None:
                    line += " M: {:.3f}".format(magnetization)
                lines.append(line)

        if self.has_dipole():
            lines.append(" ")
            lines.append("=== DIPOLE MOMENTS ===")
            lines.append(" ")
            if self.nuclear_dipole_moment is not None:
                lines.append(
                    "~ Nuclear dipole moment (a.u.): {}".format(
                        _dipole_to_string(self.nuclear_dipole_moment)
                    )
                )
                lines.append(" ")
            for idx, (elec_dip, comp_dip, extr_dip, dip, tot_dip, dip_db, tot_dip_db,) in enumerate(
                zip(
                    self.electronic_dipole_moment,
                    self.computed_dipole_moment,
                    self.extracted_transformer_dipoles,
                    self.dipole_moment,
                    self.total_dipole_moment,
                    self.dipole_moment_in_debye,
                    self.total_dipole_moment_in_debye,
                )
            ):
                lines.append("{: 3d}: ".format(idx))
                lines.append(
                    "  * Electronic dipole moment (a.u.): {}".format(_dipole_to_string(elec_dip))
                )
                lines.append("    - computed part:      {}".format(_dipole_to_string(comp_dip)))
                for name, ex_dip in extr_dip.items():
                    lines.append(
                        "    - {} extracted energy part: {}".format(name, _dipole_to_string(ex_dip))
                    )
                if self.nuclear_dipole_moment is not None:
                    lines.append(
                        "  > Dipole moment (a.u.): {}  Total: {}".format(
                            _dipole_to_string(dip), _float_to_string(tot_dip)
                        )
                    )
                    lines.append(
                        "                 (debye): {}  Total: {}".format(
                            _dipole_to_string(dip_db), _float_to_string(tot_dip_db)
                        )
                    )
                lines.append(" ")

        return lines
