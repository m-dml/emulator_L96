# Code for Deep Emulators for Differentiation, Forecasting and Parametrization in Earth Science Simulators

This repository contains pytorch code for defininig and training emulators on the [Lorenz-96 model](https://en.wikipedia.org/wiki/Lorenz_96_model) commonly used as a simple model for chaotic atmospheric dynamics.

The main purpose of this repository is to explore the use of the derivatives and gradients provided by autodifferentiation as surrogate for the derivates of the Lorenz-96 model. To this end we train neural networks in pytorch to emulate the Lorenz-96 model and use their pytorch-generated derivatives (input-output Jacobians) for downstream applications such as data assimilation (DA) and learning parameterizations of the Lorenz-96 dynamics.

## Data generation

This repository uses our [Lorenz-96 simulation package](https://github.com/m-dml/L96sim) to generate the Lorenz-96 data for training the neural networks, and to compare the output of the trained networks with ground-truth Lorenz-96 dynamics. See the [README](https://github.com/m-dml/L96sim/blob/master/readme.md) for installation instructions.

## Network training

This repository contains files describing different numerical experiments we conducted. Experiments for training the networks to emulate the one-level Lorenz-96 model are found in [/experiments](https://github.com/m-dml/emulator_L96/tree/master/experiments). Experiments included here were for the commonly studied  40-
dimensional system with forcing parameter F=8, resulting in chaotic dynamics.

To start a network-training experiment, use ```python main_train.py -c experiments/XYZ.yml``` where XYZ is an experiment filename.

## Data assimilation

Experiments on using these trained networks for data assimilation on the Lorenz-96 model (in the form of 4D-Var) are found in [/experiments_DA](https://github.com/m-dml/emulator_L96/tree/master/experiments_DA). We re-implemented the experimental setup as used in [Fertig et al. (2006)](https://doi.org/10.1111/j.1600-0870.2006.00205.x). 

To start a DA experiment, use ```python main_4DVar.py -c experiments_DA/XYZ.yml``` where XYZ is an experiment filename.

## Parametrization learning

Experiments on using these trained networks to learn parametrizations of the two-level Lorenz-96 model are found in [/experiments_parametrization](https://github.com/m-dml/emulator_L96/tree/master/experiments_parametrization). We used the two-level Lorenz-96 model and simple local parametrization as described in [Rasp et al. (2020)](https://doi.org/10.5194/gmd-13-2185-2020). 

To start a parametrization experiment, use ```python main_parametrization.py -c experiments_parametrization/XYZ.yml``` where XYZ is an experiment filename.

# Figure generation

This repository furthermore contains notebooks for the generation of figures for our manuscript, found as figure_*.ipynb
