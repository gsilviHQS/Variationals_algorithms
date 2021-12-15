#!/bin/sh
conda env create --force -f environment.yml 
conda activate NEASQC4featMYQLM
pip install git+https://github.com/Qiskit/qiskit-nature.git@c61fcf0afe098d5d133cfff584971b6d39f9a75c
##pip install git+https://github.com/Qiskit/qiskit-terra.git@0da9b6bced12af9589a35b574a550a2c7b12c614
pip install myqlm
#pip install myqlm-interop[qiskit_binder]
pip install -e myqlm-interop/
pip install -e ./
