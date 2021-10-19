#!/bin/sh
conda env create --force -f environment.yml 
conda activate NEASQC4
##pip install  git+ssh://git@github.com/NEASQC/hqs_qs_nature.git
##pip install  git+ssh://git@github.com/NEASQC/az_qs_terra.git
pip install git+https://github.com/Qiskit/qiskit-nature.git@c61fcf0afe098d5d133cfff584971b6d39f9a75c
pip install git+https://github.com/Qiskit/qiskit-terra.git@0da9b6bced12af9589a35b574a550a2c7b12c614

