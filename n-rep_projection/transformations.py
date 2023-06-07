"""Functions to transform 1- and 2-RDMs between the particle, hole, and particle-hole sectors."""
import numpy as np
from itertools import product


def get_d1_from_q1(q1: np.ndarray) -> np.ndarray:
    r"""Transform the 1-RDM from the hole to the particle sector.
    
    Yields the elements of the 1-particle RDM
    
    .. math:
        ^1D_{ij} = \langle c^\dagger_i c_j \rangle,
    
    given the 1-hole RDM

    .. math:
        ^1Q_{ij} = \langle c_i c^\dagger_j \rangle.

    Args:
        q1 (np.ndarray): The 1-hole RDM

    Returns:
        d1 (np.ndarray): The 1-particle RDM
    """
    dim = q1.shape[0]
    d1 = np.zeros_like(q1, dtype='complex128')
    delta = np.eye(dim)
    for p, q in product(range(dim), repeat=2):
        d1[q][p] = delta[p][q] - q1[p][q]
    return d1


def get_q1_from_d1(d1: np.ndarray) -> np.ndarray:
    r"""Transform the 1-RDM from the particle to the hole sector.
    
    Yields the elements of the 1-hole RDM
    
    .. math:
        ^1Q_{ij} = \langle c_i c^\dagger_j \rangle,
    
    given the 1-particle RDM

    .. math:
        ^1D_{ij} = \langle c^\dagger_i c_j \rangle.

    Args:
        d1 (np.ndarray): The 1-particle RDM

    Returns:
        q1 (np.ndarray): The 1-hole RDM
    """
    dim = d1.shape[0]
    q1 = np.zeros_like(d1, dtype='complex128')
    delta = np.eye(dim)
    for p, q in product(range(dim), repeat=2):
        q1[q][p] = delta[p][q] - d1[p][q]
    return q1


def get_d2_from_q1_q2(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r"""Transform the 2-RDM from the hole to the particle sector.
    
    Yields the elements of the 2-particle RDM
    
    .. math:
        ^2D_{ijkl} = \langle c^\dagger_i c^\dagger_j c_l c_k \rangle,
    
    given the 1-hole and 2-hole RDMs

    .. math:
        ^1Q_{ij} = \langle c_i c^\dagger_j \rangle,
        ^2Q_{ijkl} = \langle c_i c_j c^\dagger_l c^\dagger_k \rangle.

    Args:
        q1 (np.ndarray): The 1-hole RDM
        q2 (np.ndarray): The 2-hole RDM

    Returns:
        d2 (np.ndarray): The 2-particle RDM
    """
    dim = q2.shape[0]
    delta = np.eye(dim)
    d2 = np.zeros_like(q2, dtype='complex128')
    for p, q, r, s in product(range(dim), repeat=4):
        d2[p][q][r][s] = q2[p][q][r][s] \
            + delta[q][s] * q1[p][r] - delta[p][s] * q1[q][r] \
            - delta[q][r] * q1[p][s] + delta[p][r] * q1[q][s] \
            - delta[q][s] * delta[p][r] + delta[p][s] * delta[q][r]
    return d2


def get_d2_from_d1_g2(d1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    r"""Transform the 2-RDM from the particle-hole to the particle sector.
    
    Yields the elements of the 2-particle RDM
    
    .. math:
        ^2D_{ijkl} = \langle c^\dagger_i c^\dagger_j c_l c_k \rangle,
    
    given the 1-particle and particle-hole RDMs

    .. math:
        ^1D_{ij} = \langle c^\dagger_i c_j \rangle,
        ^2G_{ijkl} = \langle c^\dagger_i c_j c^\dagger_l c_k \rangle.

    Args:
        d1 (np.ndarray): The 1-particle RDM
        g2 (np.ndarray): The particle-hole RDM

    Returns:
        d2 (np.ndarray): The 2-particle RDM
    """
    dim = g2.shape[0]
    delta = np.eye(dim)
    d2 = np.zeros_like(g2, dtype='complex128')
    for p, q, r, s in product(range(dim), repeat=4):
        d2[p][q][r][s] = delta[q][r] * d1[p][s] - g2[p][r][q][s]
    return d2


def get_q2_from_d1_d2(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    r"""Transform the 2-RDM from the particle to the hole sector.
    
    Yields the elements of the 2-hole RDM
    
    .. math:
        ^2Q_{ijkl} = \langle c_i c_j c^\dagger_l c^\dagger_k \rangle,
    
    given the 1-particle and 2-particle RDMs

    .. math:
        ^1D_{ij} = \langle c^\dagger_i c_j \rangle,
        ^2D_{ijkl} = \langle c^\dagger_i c^\dagger_j c_l c_k \rangle.

    Args:
        d1 (np.ndarray): The 1-particle RDM
        d2 (np.ndarray): The 2-particle RDM

    Returns:
        q2 (np.ndarray): The 2-hole RDM
    """
    dim = d2.shape[0]
    q2 = np.zeros_like(d2, dtype='complex128')
    delta = np.eye(dim)
    for p, q, r, s in product(range(dim), repeat=4):
        q2[p][q][r][s] = (d2[p][q][r][s]
                          + delta[q][s] * d1[p][r] - delta[p][s] * d1[q][r]
                          - delta[q][r] * d1[p][s] + delta[p][r] * d1[q][s]
                          - delta[q][s] * delta[p][r] + delta[p][s] * delta[q][r])
    return q2


def get_g2_from_d1_d2(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    r"""Transform the 2-RDM from the particle to the particle-hole sector.
    
    Yields the elements of the particle-hole RDM
    
    .. math:
        ^2G_{ijkl} = \langle c^\dagger_i c_j c^\dagger_l c_k \rangle,
    
    given the 1-particle and 2-particle RDMs

    .. math:
        ^1D_{ij} = \langle c^\dagger_i c_j \rangle,
        ^2D_{ijkl} = \langle c^\dagger_i c^\dagger_j c_l c_k \rangle.

    Args:
        d1 (np.ndarray): The 1-particle RDM
        d2 (np.ndarray): The 2-particle RDM

    Returns:
        g2 (np.ndarray): The particle-hole RDM
    """
    dim = d2.shape[0]
    g2 = np.zeros_like(d2, dtype='complex128')
    delta = np.eye(dim)
    for p, q, r, s in product(range(dim), repeat=4):
        g2[p][q][s][r] = delta[q][s] * d1[p][r] - d2[p][s][q][r]
    return g2
