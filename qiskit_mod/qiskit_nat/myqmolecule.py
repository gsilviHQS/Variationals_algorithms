""" myQMolecule """

import logging
import os
import tempfile
import warnings
from typing import List
from qiskit_nature.drivers.second_quantization.qmolecule import QMolecule

import numpy

TWOE_TO_SPIN_SUBSCRIPT = "ijkl->ljik"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)


class myQMolecule(QMolecule):
    """
    Molecule data class containing driver result.

    When one of the chemistry :mod:`~qiskit_nature.drivers.second_quantization` is run and instance
    of this class is returned. This contains various properties that are made available in
    a consistent manner across the various drivers.

    Note that values here, for the same input molecule to each driver, may be vary across
    the drivers underlying code implementation. Also some drivers may not provide certain fields
    such as dipole integrals in the case of
    :class:`~qiskit_nature.drivers.second_quantization.PyQuanteDriver`.

    This class provides methods to save it and load it again from an HDF5 file
    """

    QMOLECULE_VERSION = 3

    def __init__(self, filename=None):
        super().__init__(filename)

        self.nuc_kinetic = None
        self.el_kinetic = None
        self.el_eri = None
        self.nuc_eri = None
        self.mix_eri = None

        dim = 2

        self.num_molecular_orbitals = dim*2

        #STATE=  1 ENERGY =        -1.0537650432
        # nuclear kinetic ints
        #         1           2
        # 1    0.0193113
        # 2    0.0000000   0.0193113
        self.nuc_kinetic = numpy.zeros((dim, dim))
        self.nuc_kinetic[0,0] = 0.0193113
        self.nuc_kinetic[1,1] = 0.0193113

        # electronic kinetic ints
        #         1           2
        # 1    0.5141493
        # 2    0.0000000   0.4429029
        self.el_kinetic = numpy.zeros((dim, dim))
        self.el_kinetic[0,0] = 0.5141493
        self.el_kinetic[1,1] = 0.4429029

        # 2-nuclear ints
        #  1  1  1  1    1  0.308983329105E+01  1  1  2  2    2  0.308983328889E+01  2  1  2  1    3  0.240662830981E+01
        #  2  2  2  2    4  0.308983328673E+01
        self.nuc_eri = numpy.zeros((dim, dim, dim, dim))
        self.nuc_eri[0,0,0,0] = 0.308983329105e+01
        self.nuc_eri[0,0,1,1] = 0.308983328889e+01
        self.nuc_eri[1,0,1,0] = 0.240662830981e+01
        self.nuc_eri[1,1,1,1] = 0.308983328673e+01

        self.nuc_eri[1,1,0,0] = 0.308983328889e+01
        self.nuc_eri[0,1,1,0] = 0.240662830981e+01
        self.nuc_eri[0,1,0,1] = 0.240662830981e+01
        self.nuc_eri[1,0,0,1] = 0.240662830981e+01
        # 2-electronic ints
        #  1  1  1  1    1  0.628234836549E+00  1  1  2  2    2  0.427139958018E+00  2  1  2  1    3  0.781765317968E-01
        #  2  2  2  2    4  0.384791185319E+00
        self.el_eri = numpy.zeros((dim, dim, dim, dim))
        self.el_eri[0,0,0,0] = 0.628234836549e+00
        self.el_eri[0,0,1,1] = 0.427139958018e+00
        self.el_eri[1,0,1,0] = 0.781765317968e-01
        self.el_eri[1,1,1,1] = 0.384791185319e+00

        self.el_eri[1,1,0,0] = 0.427139958018e+00
        self.el_eri[0,1,1,0] = 0.781765317968e-01
        self.el_eri[1,0,0,1] = 0.781765317968e-01
        self.el_eri[0,1,0,1] = 0.781765317968e-01
        # 2-el-nuc ints
        #  1  1  1  1    1  0.856601851395E+00  1  1  2  2    2  0.856601850720E+00  2  1  2  1    3 -0.168555566762E+00
        #  2  2  1  1    4  0.494370591475E+00  2  2  2  2    5  0.494370591777E+00
        self.mix_eri = numpy.zeros((dim, dim, dim, dim))
        self.mix_eri[0,0,0,0] = 0.856601851395e+00
        self.mix_eri[0,0,1,1] = 0.856601850720e+00
        self.mix_eri[1,0,1,0] = -0.168555566762e+00
        self.mix_eri[1,1,0,0] = 0.494370591475e+00
        self.mix_eri[1,1,1,1] = 0.494370591777E+00

        self.mix_eri[0,1,0,1] = -0.168555566762e+00
        self.mix_eri[0,1,1,0] = -0.168555566762e+00
        self.mix_eri[1,0,0,1] = -0.168555566762e+00
