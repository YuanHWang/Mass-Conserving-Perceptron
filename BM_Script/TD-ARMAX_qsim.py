import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch import Tensor
from torch.nn.parameter import Parameter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
from typing import Tuple
import numpy as np
import pandas as pd
import os
import math
import datetime
from tqdm import tqdm
import sys
from sklearn.metrics import mean_squared_error
import csv
import glob 
# import class
from MCPBRNN_lib_tools.BMZoo import ARX_LeafRiver_Qsim
from MCPBRNN_lib_tools.Eval_Metric import ANLL_out, correlation, NS, KGE
from MCPBRNN_lib_tools.Loss_Function import  KGELoss, ANLL_type1, ANLL_type2, MSELoss

parser = argparse.ArgumentParser()

parser.add_argument('--case_no',
                        type=int,
                        default=0,
                        help="Case Number: Initial values for the parameters of Hydro-MC-simple-LSTM")

parser.add_argument('--epoch_no',
                        type=int,
                        default=2,
                        help="number of epoch to train the network")

parser.add_argument('--depth_size',
                        type=int,
                        default=1,
                        help="number of hidden nodes")

parser.add_argument('--time_lag',
                        type=int,
                        default=0,
                        help="number of input time lag for the gate")

parser.add_argument('--gate_dim',
                        type=int,
                        default=1,
                        help="number of nodes for ANN functions")

parser.add_argument('--gate_dim_ucorr',
                        type=int,
                        default=3,
                        help="number of nodes for ANN functions at input gate for correcting precipitation")

parser.add_argument('--seed_no',
                        type=int,
                        default=2925,
                        help="specify torch random seed")

parser.add_argument('--Q_max',
                        type=float,
                        default=64.014774,
                        help="")

parser.add_argument('--PP_input',
                        type=int,
                        default=1,
                        help="")

parser.add_argument('--Q_input',
                        type=int,
                        default=0,
                        help="")

parser.add_argument('--PET_input',
                        type=int,
                        default=1,
                        help="")

parser.add_argument('--MRS_input',
                        type=int,
                        default=1,
                        help="")

cfg = vars(parser.parse_args())

# setup random seed
seed_no = cfg["seed_no"]
np.random.seed(seed_no)
torch.manual_seed(seed_no)
Q_max = cfg["Q_max"]

PP_input = cfg["PP_input"]
Q_input = cfg["Q_input"]
PET_input = cfg["PET_input"]
MRS_input = cfg["MRS_input"]
# Define Matrix size & Hyperparameters
input_size = 1
depth_size = cfg["depth_size"]
gate_dim = cfg["gate_dim"]
gate_dim_ucorr = cfg["gate_dim_ucorr"]
seq_length = 1
input_size_dyn = PP_input + Q_input + PET_input

if (PP_input-1)>=Q_input:
   if (PP_input-1)>=(PET_input-1):
      valid_data_str = (PP_input-1) 
   else:
      valid_data_str = (PET_input-1)
else:
   if (Q_input)>=(PET_input-1):
      valid_data_str = Q_input
   else:
      valid_data_str = (PET_input-1)

#if valid_data_str < MRS_input:
#   valid_data_str = MRS_input

input_size_stat = 0
num_output = 1
num_epochs = cfg["epoch_no"]
learning_rate = 0.0125
learning_rates = {300: 0.0125, 600: 0.0125}

# Define case & directory
CaseName = 'TD-ARMAX-caseno' + str(cfg["case_no"]) + "_" + "PP_Lag_"  +  str(PP_input) + "Q_Lag_" + str(Q_input) + "PET_Lag_" + str(PET_input)+ "MRS_Lag_" + str(MRS_input)
directory = CaseName
parent_dir = "/home/u9/yhwang0730/PB-LSTM-Papers/20230925-BM-New"
# HPC
###Ini_dir = "/Users/yhwang/Desktop/HPC_DownloadTemp/2022-Fall/20221019-SingleNode-UpdateNorm/result_temp_PET_NormUpdate/PET-constraint-iter2/MCPBRNN_PETconstraint_Generic_NormFUpdate_1Layer_1node_1/model_epoch10.pt"

path = os.path.join(parent_dir, directory)
isExist = os.path.exists(path)
if not isExist:
  os.mkdir(path)
  print("The new directory is created!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Import Data
text = "../20230129-ML-BM-MDUPLEX-LeafRiver/data_final/LeafRiverDaily_PP_lag" + str(PP_input) + "_Q_lag_" + str(Q_input) + "_PET_lag_" + str(PET_input) + "_40YR.csv"
F_data = pd.read_csv(text, header=None, sep= ',')
sizedim = F_data.shape[1]
F_data = F_data.to_numpy()
x_data = np.zeros((F_data.shape[0],F_data.shape[1]-1))
y_data = np.zeros((F_data.shape[0],1))
x_data = F_data[:,0:F_data.shape[1]-1]
y_data = F_data[:,F_data.shape[1]-1]
y_data = y_data.reshape(F_data.shape[0],-1)
x_data_qsim = np.zeros((F_data.shape[0],1))

# Skill Flag Training
if valid_data_str==0:
   SkillFlag = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR_Flag.txt', header=None, delimiter=r"\s+")
   print("0")
elif valid_data_str==1:
   SkillFlag = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR_Flag_Lagsum1.txt', header=None, delimiter=r"\s+")
   print("1")
elif valid_data_str==2:
   SkillFlag = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR_Flag_Lagsum2.txt', header=None, delimiter=r"\s+")
   print("2")
elif valid_data_str==3:
   SkillFlag = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR_Flag_Lagsum3.txt', header=None, delimiter=r"\s+")
   print("3")
elif valid_data_str==4:
   SkillFlag = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR_Flag_Lagsum4.txt', header=None, delimiter=r"\s+")
   print("4")
elif valid_data_str==5:
   SkillFlag = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR_Flag_Lagsum5.txt', header=None, delimiter=r"\s+")
   print("5")

SkillFlag = SkillFlag.rename(columns={0: 'Flag'})
SF = torch.tensor(SkillFlag['Flag'])
Mask_Test = SF.eq(1).unsqueeze(1)
Mask_Select = SF.eq(0).unsqueeze(1)
Mask_Train = SF.eq(-1).unsqueeze(1)

# Define output matrix
par_no = 0 # Temporarily not save the parameter value
skillmetric = 7
OutMatrix = np.zeros((num_epochs,par_no+skillmetric*4+3))

train_x = x_data
train_y = y_data

num_samples, num_features = train_x.shape
x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
y_new = np.zeros((num_samples - seq_length + 1, 1))

for i in range(0, x_new.shape[0]):
    x_new[i, :, :num_features] = train_x[i:i + seq_length, :]
    y_new[i, :] = train_y[i + seq_length - 1, 0]
       
x_new = torch.tensor(x_new)
y_new = torch.tensor(y_new)
x_new = x_new.float()
y_new = y_new.float()
trainx, trainy = Variable(x_new), Variable(y_new)

class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):
        """Initialize model.
        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat 
        self.MCPBRNNNode = ARX_LeafRiver_Qsim(input_size=input_size_dyn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_d):
        """Run forward pass through the model.
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """

        h_t = self.MCPBRNNNode(x_d)    
        last_h = self.dropout(h_t)      
        out = last_h       
        return out

model = Model(input_size_dyn=input_size_dyn,
              input_size_stat=input_size_stat,          
              dropout=0).to(device)

loss_func = KGELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = Data.TensorDataset(trainx, trainy)
loader = Data.DataLoader(
         dataset=train_dataset, 
         batch_size=F_data.shape[0],
         shuffle=False, num_workers=0)

# Set up initial value
#rand_num = np.random.rand(1,2)
#rand_num = np.random.normal(0,1,1)
# Save the initial value of parameters
for name, param in model.state_dict().items():
   if model.state_dict()[name][:].dim() == 1:
      YHW=np.random.rand(model.state_dict()[name][:].shape[0])
      model.state_dict()[name][:]=torch.Tensor(YHW)
   elif model.state_dict()[name][:].dim() == 2:
      YHW=np.random.rand(model.state_dict()[name][:].shape[0],model.state_dict()[name][:].shape[1])
      model.state_dict()[name][:]=torch.Tensor(YHW)
   elif model.state_dict()[name][:].dim() == 3:
      YHW=np.random.rand(model.state_dict()[name][:].shape[0],model.state_dict()[name][:].shape[1],model.state_dict()[name][:].shape[2])
      model.state_dict()[name][:]=torch.Tensor(YHW)

for name, param in model.state_dict().items():
    print(name)
    print(param)

savetext = CaseName + '/' +'model_epoch0.pt'
torch.save(model.state_dict(), savetext)

for epoch in range(1,num_epochs+1): 
    model.train()
    if epoch in learning_rates.keys():
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rates[epoch]

    pbar = tqdm(loader, file=sys.stdout)  
    pbar.set_description(f'# Epoch {epoch}')
    savetext = CaseName + '/' +'model_epoch' + str(epoch) + '.pt'
    pbar.set_description(f'# Epoch {epoch}')    

    for data in pbar:
        optimizer.zero_grad()
        x, y,= data 
        predictions = model(x)

        sim = torch.masked_select(predictions, Mask_Train).unsqueeze(1)       
        obs =  torch.masked_select(y, Mask_Train).unsqueeze(1)      
        loss = loss_func(sim, obs)        
        loss.backward()
        optimizer.step()

        predictions = model(x)
        sim = torch.masked_select(predictions, Mask_Train).unsqueeze(1)
        obs =  torch.masked_select(y, Mask_Train).unsqueeze(1)
        sim_select = torch.masked_select(predictions, Mask_Select).unsqueeze(1)
        obs_select =  torch.masked_select(y, Mask_Select).unsqueeze(1)
        sim_test = torch.masked_select(predictions, Mask_Test).unsqueeze(1)
        obs_test =  torch.masked_select(y, Mask_Test).unsqueeze(1)

    Result = model(x)
    hidden_eval = Result

    yout_eval = y.detach().cpu().numpy()    
    pout_eval = predictions.detach().cpu().numpy()   
    hidden_eval = hidden_eval.detach().cpu().numpy()            

    #calculate the skill
    sim = sim.detach().cpu().numpy()  
    obs = obs.detach().cpu().numpy()   
    sim_select = sim_select.detach().cpu().numpy()   
    obs_select = obs_select.detach().cpu().numpy()  
    sim_test = sim_test.detach().cpu().numpy()        
    obs_test = obs_test.detach().cpu().numpy()  

    hidden_eval =  hidden_eval*Q_max 
    pout_eval =  pout_eval*Q_max 
    sim = sim *Q_max 
    obs = obs *Q_max 
    sim_select = sim_select *Q_max 
    obs_select = obs_select *Q_max  
    sim_test = sim_test *Q_max 
    obs_test = obs_test *Q_max 

    pout_eval[pout_eval < 0] = 0
    hidden_eval[hidden_eval < 0] = 0
    
    [B, E, C, D, G] =KGE(sim, obs)
    A = NS(sim, obs)
    F = mean_squared_error(obs, sim)

    sz = obs.shape[0]
    [KGEtimelag_1, X1, X2, X3, X4] =KGE(sim[0+1:sz], obs[0:sz-1])   
    [KGEtimelag_2, X1, X2, X3, X4] =KGE(sim[0+2:sz], obs[0:sz-2])   
    [KGEtimelag_3, X1, X2, X3, X4] =KGE(sim[0+3:sz], obs[0:sz-3])   
    OutMatrix[epoch-1,par_no+28] = KGEtimelag_1
    OutMatrix[epoch-1,par_no+29] = KGEtimelag_2
    OutMatrix[epoch-1,par_no+30] = KGEtimelag_3

    [B2, E2, C2, D2, G2] =KGE(sim_select, obs_select)
    A2 = NS(sim_select, obs_select)
    F2 = mean_squared_error(obs_select, sim_select)

    [B3, E3, C3, D3, G3] =KGE(sim_test, obs_test)
    A3 = NS(sim_test, obs_test)
    F3 = mean_squared_error(obs_test, sim_test)    

    #[B4, E4, C4, D4, G4] =-99999#KGE(sim_spinup, obs_spinup)
    B4 = -99999
    E4 = -99999
    C4 = -99999
    D4 = -99999
    G4 = -99999
    A4 = -99999#NS(sim_spinup, obs_spinup)
    F4 = -99999#mean_squared_error(obs_spinup, sim_spinup)

    OutMatrix[epoch-1,par_no] = A
    OutMatrix[epoch-1,par_no+7] = A2    
    OutMatrix[epoch-1,par_no+14] = A3 
    OutMatrix[epoch-1,par_no+21] = -99999#A4 

    if np.isnan(B):
       OutMatrix[epoch-1,par_no+1] = -99999
    else:
       OutMatrix[epoch-1,par_no+1] = B

    if np.isnan(B2):
       OutMatrix[epoch-1,par_no+8] = -99999
    else:
       OutMatrix[epoch-1,par_no+8] = B2

    if np.isnan(B3):
       OutMatrix[epoch-1,par_no+15] = -99999
    else:
       OutMatrix[epoch-1,par_no+15] = B3

    if np.isnan(B4):
       OutMatrix[epoch-1,par_no+22] = -99999
    else:
       OutMatrix[epoch-1,par_no+22] = -99999#B4

    OutMatrix[epoch-1,par_no+2] = C
    OutMatrix[epoch-1,par_no+3] = D 
    OutMatrix[epoch-1,par_no+9] = C2
    OutMatrix[epoch-1,par_no+10] = D2     
    OutMatrix[epoch-1,par_no+16] = C3
    OutMatrix[epoch-1,par_no+17] = D3 
    OutMatrix[epoch-1,par_no+23] = -99999#C4
    OutMatrix[epoch-1,par_no+24] = -99999#D4 

    if np.isnan(E):  
       OutMatrix[epoch-1,par_no+4] = -99999
    else:  
       OutMatrix[epoch-1,par_no+4] = E

    if np.isnan(E2):
       OutMatrix[epoch-1,par_no+11] = -99999
    else:  
       OutMatrix[epoch-1,par_no+11] = E2       

    if np.isnan(E3):
       OutMatrix[epoch-1,par_no+18] = -99999
    else:  
       OutMatrix[epoch-1,par_no+18] = E3 

    if np.isnan(E4):
       OutMatrix[epoch-1,par_no+25] = -99999
    else:  
       OutMatrix[epoch-1,par_no+25] = -99999#E4

    OutMatrix[epoch-1,par_no+5] = F
    OutMatrix[epoch-1,par_no+12] = F2 
    OutMatrix[epoch-1,par_no+19] = F3 
    OutMatrix[epoch-1,par_no+26] = -99999#F4 

    if np.isnan(G):  
       OutMatrix[epoch-1,par_no+6] = -99999
    else:  
       OutMatrix[epoch-1,par_no+6] = G

    if np.isnan(G2):
       OutMatrix[epoch-1,par_no+13] = -99999
    else:  
       OutMatrix[epoch-1,par_no+13] = G2       

    if np.isnan(G3):
       OutMatrix[epoch-1,par_no+20] = -99999
    else:  
       OutMatrix[epoch-1,par_no+20] = G3 

    if np.isnan(G4):
       OutMatrix[epoch-1,par_no+27] = -99999
    else:  
       OutMatrix[epoch-1,par_no+27] = -99999#G4

    torch.save(model.state_dict(), savetext)
    print(B2)
# Save Results file
outtofile = pd.DataFrame(OutMatrix, columns=['NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])
outname = CaseName + '/' + 'IC_caseno_' + str(cfg["case_no"]) + '_summary.csv'
outtofile.to_csv(outname)

print(np.argmax(OutMatrix[:,8]))
indx = np.argmax(OutMatrix[:,8]) + 1
mpt = CaseName + '/model_epoch' + str(indx) + '.pt'
model.load_state_dict(torch.load(mpt))

for data in pbar:
    x, y,= data 
    predictions = model(x)

predictions = predictions.detach().cpu().numpy()   
Out_file = pd.DataFrame(predictions*Q_max)
outname = CaseName + '/' + 'Outhidden_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

outtext = CaseName + '/*.pt'
files = glob.glob(outtext, recursive=True)

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
