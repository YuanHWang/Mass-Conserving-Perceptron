# Mass-Conserving-Perceptron (MCP)

This repository contains all the necessary scripts and data for reproducing the results presented in the publication: 

Wang, Y.H. and Gupta, H.V., 2024. A mass‐conserving‐perceptron for machine‐learning‐based modeling of geoscientific systems. *Water Resources Research, 60(4), p.e2023WR036461*. https://doi.org/10.1029/2023WR036461

## Get Started

The environment we used for the manuscript above is provided in the file environment.yml. To create the same conda environment, please ensure that conda is installed, then run:

```
conda env create -f environment.yml
```

The code will work with other Python environments that have different package versions, such as newer PyTorch versions, as long as all necessary packages are installed.

## Introduction to Each Folder

* **20220527-MDUPLEX-LeafRiver**:
  Contains the [40-year Leaf River data]([https://repository.arizona.edu/handle/10150/668421](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/WR019i001p00251))

  and the associated flags for training, selection, and testing sets.
* **MCPBRNN_lib_tools**:
  The primary library used to train the mass-conserving-perceptron, as well as various physics-based and data-driven benchmark models.
* **Training_Script**: 
  Each ```.py``` file in this folder is used to train a specific case presented in the paper.
* **Evaluation_Script**:
  Each ```.py``` file in this folder is used to evaluate a specific case presented in the paper.
* **BM_Script**:
  Each ```.py``` file in this folder is used to train and evaluate a specific benchmark case presented in this study.
  
## Run the Codes

The user will need to move one of the training scripts from the **Training_Script** folder back to the parent directory and then run the script. The user is expected to configure the following in the script:
* Set the job folder name by updating the ```CaseName``` variable.
* Set the running directory by updating the ```parent_dir``` variable.
* Specify the directory for parameter initialization by updating the relevant ```Ini_dir``` variables.

To output all the time series generated by MCP, the user should move the associated files from the Evaluation_Script folder back to the parent directory and then run the script. The user is expected to configure the following in the script:

* Set the running directory by updating the ```parent_dir``` variable.
* Specify the directory where the .pt file results are stored by updating the relevant ```Ini_dir``` variables.

Finally, to train/evaluate the benchmark case, the user should move one of the training scripts from the BM_Script folder back to the parent directory and then run the script. The user is expected to set up the running directory by updating the ```parent_dir``` variable in the script.

## Notes

Please note the following:
* The current version of the code can only run certain cases based on the input variables used. For ARX, ANN, and RNN, the only case implemented uses current-day precipitation, potential evapotranspiration, and simulated discharge (or cell state) from the previous timestep.
* The code is designed to select the best epoch based on the highest KGE score for the selection set. Minor adjustments may be required in the script for model selection.
* The default number of epochs used in the training scripts may not match the number listed in Table S1 of the paper. Be sure to use the ```--epoch_no``` parser to enter the correct epoch number when reproducing the results.

## Concluding Remark

* Please refer to the ```Instructions_README.pdf``` file for more information on how each training and evaluation script corresponds to the test cases presented in the paper.
