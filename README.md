# MassConservingPerceptron

This repository contains all the necessary scripts and data for reproducing the results presented in the publication: 

Wang, Y.H. and Gupta, H.V., 2024. A mass‐conserving‐perceptron for machine‐learning‐based modeling of geoscientific systems. *Water Resources Research, 60(4), p.e2023WR036461*. https://doi.org/10.1029/2023WR036461

## Get Started

The environment we used for the manuscript above is provided in the file environment.yml. To create the same conda environment, please ensure that conda is installed, then run:

```
conda env create -f environment.yml
```

The code will work with other Python environments that have different package versions, such as newer PyTorch versions, as long as all necessary packages are installed.

## Introduction to Each Folder
* 20220527-MDUPLEX-LeafRiver
  Contains the 40-year Leaf River data and the associated flags for training, selection, and testing sets.
