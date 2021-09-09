#!/bin/sh
conda env create -f environment.yml 
conda activate NEASQC4
#pip install  git+ssh://git@github.com/NEASQC/hqs_qs_nature.git
pip install git+https://github.com/Qiskit/qiskit-nature.git@c61fcf0afe098d5d133cfff584971b6d39f9a75c
pip install  git+ssh://git@github.com/NEASQC/az_qs_terra.git
