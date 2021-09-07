#!/bin/sh
#conda env create -f environment.yml #doesn`work
conda create -n NEASQC4 python=3.9
conda activate NEASQC4
pip install qiskit-aer==0.8.2 matplotlib ipykernel
pip install  git+ssh://git@github.com/NEASQC/hqs_qs_nature.git
pip install  git+ssh://git@github.com/NEASQC/az_qs_terra.git
