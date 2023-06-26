#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pyscf import gto, scf
from .Github_calc_Hamilt import save_H_into_dict
from .Github_calc_Energy import save_E_into_dict

from qat.lang.AQASM import Program
from qat.interop.qiskit import qlm_to_qiskit

from .Github_calc_Energy import HE_circuit_for_ansatz


def build_benz_dist_1_co2(alpha, d_benz_co2, basis='sto-3g'):
    """
    Input : alpha (varying R_C-C of benzene), d_benz_co2 (distance center of benzene-O of CO2), basis set
    Ouput : benzene (distortion 1) + CO2 PySCF molecule (according to paper https://link.aps.org/doi/10.1103/PhysRevA.107.012416)
    """
    mol_benz = gto.Mole()
    mol_benz.verbose = 5
    mol_benz.output = 'benz_co2.log'
    R_CC = 1.39*alpha
    R_CH = 1.09
    c_h = ''
    for i in range(6):
        angle = np.pi/6 + i*np.pi/3
        x, y = R_CC*np.cos(angle), R_CC*np.sin(angle)
        x_H, y_H = (R_CC+R_CH)*np.cos(angle), (R_CC+R_CH)*np.sin(angle)
        c_h += f'C {x} {y} 0; H {x_H} {y_H} 0;'
    R_CO = 1.16
    c_h += f'C 0 0 {d_benz_co2+R_CO}; O 0 0 {d_benz_co2}; O 0 0 {d_benz_co2+2*R_CO}'
    mol_benz.atom = c_h
    mol_benz.basis = basis
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz


def calc_benz_co2(d_benz_co2, alpha=1):
    """
    This function calculate the ground state energy of a benzene + CO2 system :
        - Benzene can be distorted by varying the alpha parameter (see https://link.aps.org/doi/10.1103/PhysRevA.107.012416)
        - d_benz_co2 controls the distance between center of benzene and one O of CO2.
    """

    mol, m_mol = build_benz_dist_1_co2(alpha, d_benz_co2, 'sto-3g')

    #############################################

    save_filename = 'test_github_neasqc.benzene_co2'
    nb_lumo = 4
    nb_homo = 4
    dic_H_save = save_H_into_dict(d_benz_co2, save_filename, mol, m_mol, nb_homo, nb_lumo)

    #############################################

    hamilt_filename = 'test_github_neasqc.benzene_co2.H.pickle'
    save_filename = 'test_github_neasqc.benzene_co2'


    ### qUCC calculation 
    dic_E_save = save_E_into_dict(d_benz_co2, hamilt_filename, save_filename, mol, m_mol, nb_homo, nb_lumo, ansatz="qUCC", nbshots=0, N_trials=1)

    ### HE calculation
    dic_E_save = save_E_into_dict(d_benz_co2, hamilt_filename, save_filename, mol, m_mol, nb_homo, nb_lumo, ansatz="HE", nbshots=0, d=1, N_trials=1)


    return dic_E_save