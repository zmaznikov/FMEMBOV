# Code for the master thesis "Flexible Mixed Effects Models for Binary Outcome Variables"

by Zafir Maznikov

## Notes
#### MATLAB
+ Computations in MATLAB were performed with MATLAB version 24.2.0.2740171 (R2024b) Update 1
+ Required toolbox: Statistics and Machine Learning Toolbox Version 24.2 (R2024b)
+ Required packages: `deepGLMM` (https://github.com/VBayesLab/deepGLMM) and `deepGLM` (https://github.com/VBayesLab/deepGLM)
#### Python
+ Computations in Python were performed with Python version 3.7.17
+ Required package: `os`, `sys`, `numpy`, `pandas`, `matplotlib`, `lmmnn` (https://github.com/gsimchoni/lmmnn)
#### R
+ Computations in R were performed with R version 4.3.3 (2024-02-29 ucrt)
+ Required package: `saemix`

## Contents
#### `/Data`
+ The data simulated in `.csv` format.
+ `/Data/TrueAndPredictedTest` contains a comparison of true and predicted probabilities by method.

#### `/DeepGLMM`
+ `deepGLMMpredict.m` is a modification of the original predict function in https://github.com/VBayesLab/deepGLMM with a dimension correction.
+ `reproduction_*.m` files include the reproduction of simulation studies from the literature, including data simulation, application of DeepGL(M)M and classification trees
+ `simulation_linear.m` and `simulation_nonlin.m` files include the simulation studies with linear and nonlinear fixed effects accordingly and single-level random effects, including data simulation, application of DeepGL(M)M and classification trees
+ `simulation_levels*.m` includes the simulation studies with linear and nonlinear fixed effects and two-level random effects, including data simulation, application of DeepGL(M)M and classification trees

#### `/LMMNN`
+ `LMMNN.ipynb` includes the application of the LMMNN method on the simulated data

#### `/SAEM`
+ `saemix_*.r` files include the application of the SAEM method on the simulated data, with the listed specification - linear with 5 or 10 covariates and four different nonlinear specifications - as in the DeepGLMM reproduction, as in the LMMNN reproduction, and two correct specifications in the simulation study, differentiated by the true intercept value.

#### `/Plots`
+ `plotting_probabilities.m` contains code for the visualizations in folders `/Plots/TrueP` - histograms of the true probabilities in each simulation, and `/Plots/TrueAndPredictedTest` - point cloud comparison of true and predicted p by method and simulation.
