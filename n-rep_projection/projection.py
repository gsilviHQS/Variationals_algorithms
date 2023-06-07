import numpy as np
from typing import Tuple
from transformations import (get_q1_from_d1, get_q2_from_d1_d2, get_g2_from_d1_d2,
                             get_d1_from_q1, get_d2_from_q1_q2, get_d2_from_d1_g2)
from openfermion import fixed_trace_positive_projection
from itertools import product


def switch_indicies(rdm2: np.ndarray) -> np.ndarray:
    r"""Switch the last two indicies of a 2-RDM.
    
    OpenFermion uses a different definition of the 2-RDM and in order to utilize its
    `fixed_trace_positive_projection` funtion, one needs to switch the last two indecies
    of the 2-RDM:
    
    .. math:
        ^2D_{pqrs} \to ^2D_{pqsr}
        
    Args:
        rdm2 (np.ndarray): The 2-RDM to transform
        
    Returns:
        np.ndarray: The 2-RDM with the last two indicies switched
    """
    dim = rdm2.shape[0]
    rdm2_new = np.zeros_like(rdm2, dtype='complex128')
    for p, q, r, s in product(range(dim), repeat=4):
        rdm2_new[p][q][r][s] = rdm2[p][q][s][r]
    return rdm2_new


def get_energy(d1_rdm: np.ndarray, d2_rdm: np.ndarray, energy_offset: float,
               one_electron_integrals: np.ndarray, two_electron_integrals: np.ndarray) -> float:
    r"""Get the energy from 1- and 2-particle RDM, as well as the Hamiltonian prefactors.
    
    The energy is given by the expectation value of the Hamiltonian,

    .. math:
        \langle H \rangle = E_0 + \sum_{ij} t_{ij} ^D_{ij} + \sum_{ijkl} V_{ijkl} ^2D_{ijkl},
    
    with the energy offset :math:`E_0`, the one-elctron integrals :math:`t_{ij}`,
    thetwo-electron integrals :math:`V_{ijkl}`, as well as the one-partile RDM
    :math:`^1D_{ij} = \langle c^\dagger_i c_j \rangle`, and the two-particle RDM
    :math:`^2D_{ijkl} = \langle c^\dagger_i c^\dagger_j c_l c_k \rangle`.

    Args:
        d1_rdm (np.ndarray): The 1-particle RDM
        d2_rdm (np.ndarray): The 2-particle RDM
        energy_offset (float): The energy offset of the Hamiltonian
        one_electron_integrals (np.ndarray): The one-electron integrals of the Hamiltonian
        two_electron_integrals (np.ndarray): The two-electron integrals of the Hamiltonian
    
    Returns:
        energy (float): The energy
    """
    energy = energy_offset
    energy += np.einsum('ij, ij -> ', one_electron_integrals, d1_rdm)
    energy += np.einsum('pqrs, pqrs -> ', two_electron_integrals, d2_rdm)
    return energy


def best_fixed_trace_positive_projection(
        d1_measured: np.ndarray,
        d2_measured: np.ndarray,
        number_electrons: int,
        energy_offset: float,
        one_electron_integrals: np.ndarray,
        two_electron_integrals: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""Project the 1- and 2-RDM to have a fixed trace and be positive semi-definite, in either the
    particle, hole, or particle-hole sector, depending on which projection energetically yields the
    best result.

    The energy is given by the expectation value of the Hamiltonian,

    .. math:
        \langle H \rangle = E_0 + \sum_{ij} t_{ij} ^D_{ij} + \sum_{ijkl} V_{ijkl} ^2D_{ijkl},
    
    with the energy offset :math:`E_0`, the one-elctron integrals :math:`t_{ij}`,
    thetwo-electron integrals :math:`V_{ijkl}`, as well as the one-partile RDM

    .. math:
        ^1D_{ij} = \langle c^\dagger_i c_j \rangle,
    
    and the two-particle RDM

    .. math:
        ^2D_{ijkl} = \langle c^\dagger_i c^\dagger_j c_l c_k \rangle.

    If :math:`^1D` and :math:`^2D` of the ground state are obtained from a quantum computation
    on a NISQ device, they are obscurred by decoherence and shot noise. We assume, that the
    dominating noise source is decoherence, and in this case, the calculated energy is higher
    then the actual ground state energy.

    One can mitigate this error and the statistical variance from shot noise by imposing
    constraints that the RDMs need to fulfill: They need to have a particular trace (depending on
    the number of electrons/holes in the system) and need to be positive semi-definite. We impose
    this by projecting on the closest RDMs obeying these conditions.

    This projection can be done in the hole and particle-hole sector as well, projecting the 1- and
    2- hole RDMs,

    .. math:
        ^1Q_{ij} = \langle c_i c^\dagger_j \rangle,
        ^2Q_{ijkl} = \langle c_i c_j c^\dagger_l c^\dagger_k \rangle

    or the particle-hole RDM

    .. math:
        ^2G_{ijkl} = \langle c^\dagger_i c_j c^\dagger_l c_k \rangle.
    
    This function performs all three projections and returns the energetically best result,
    transformed back into the particle sector.

    For a more detailed description and details about the physics and reasoning, as well as an
    investigation of the performance of the method see:

    Tomislav Piskor, Florian G. Eich, Michael Marthaler, Frank K. Wilhelm, Jan-Michael Reiner,
    Post-processing noisy quantum computations utilizing N-representability constraints,
    arXiv: 2304.13401 [quant-ph] (2023), https://arxiv.org/abs/2304.13401 

    Args:
        d1_measured (np.ndarray): The 1-particle RDM, as measured on the quantum computer
        d2_measured (np.ndarray): The 2-particle RDM, as measured on the quantum computer
        number_electrons (int): The number of electrons in the system
        energy_offset (float): The energy offset of the Hamiltonian
        one_electron_integrals (np.ndarray): The one-electron integrals of the Hamiltonian
        two_electron_integrals (np.ndarray): The two-electron integrals of the Hamiltonian
 
    Retruns:
        Tuple: The best energy of the three projections, the corresponding projected 1-RDM, and the
        correponding projected 2-RDM, each transformed back into the particle sector (if necessary)
    """

    # Get the number of holes in the system
    number_orbitals = d1_measured.shape[0]
    number_holes = number_orbitals - number_electrons

    # Get RDMs of hole and particle-hole sectors by transformation
    q1_measured = get_q1_from_d1(d1_measured)
    q2_measured = get_q2_from_d1_d2(d1_measured, d2_measured)
    g2_measured = get_g2_from_d1_d2(d1_measured, d2_measured)
    
    # Perform projection in each sector
    d1_projected = fixed_trace_positive_projection(d1_measured, number_electrons)
    d2_projected = fixed_trace_positive_projection(switch_indicies(d2_measured),
                                                   number_electrons * (number_electrons - 1))
    d2_projected = switch_indicies(d2_projected)
    q1_projected = fixed_trace_positive_projection(q1_measured, number_holes)
    q2_projected = fixed_trace_positive_projection(switch_indicies(q2_measured),
                                                   number_holes * (number_holes - 1))
    q2_projected = switch_indicies(q2_projected)
    g2_projected = fixed_trace_positive_projection(switch_indicies(g2_measured),
                                                   number_electrons * (number_holes + 1))
    g2_projected = switch_indicies(g2_projected)
    
    # Transform back to the particle sector after projection.
    d1_from_q1_projected = get_d1_from_q1(q1_projected)
    d2_from_q1_q2_projected = get_d2_from_q1_q2(q1_projected, q2_projected)
    d2_from_d1_g2_projected = get_d2_from_d1_g2(d1_projected, g2_projected)

    # Get the resulting energies after projecting in each sector
    energy_d = get_energy(d1_projected, d2_projected,
                          energy_offset, one_electron_integrals, two_electron_integrals)
    energy_q = get_energy(d1_from_q1_projected, d2_from_q1_q2_projected,
                          energy_offset, one_electron_integrals, two_electron_integrals)
    energy_g = get_energy(d1_projected, d2_from_d1_g2_projected,
                          energy_offset, one_electron_integrals, two_electron_integrals)
    
    # Return best energy and corresponding RDMs in the particle sector
    if energy_d <= energy_q and energy_d <= energy_g:
        return energy_d, d1_projected, d2_projected
    elif energy_q <= energy_g:
        return energy_q, d1_from_q1_projected, d2_from_q1_q2_projected
    else:
        return energy_g, d1_projected, d2_from_d1_g2_projected
