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
# import class 
from MCPBRNN_lib_tools.NodeZoo import MCPBRNN_Generic_PETconstraint_Scaling
from MCPBRNN_lib_tools.ComplexGateZoo import MCPBRNN_Generic_OutputANNGate_PETconstraint
from MCPBRNN_lib_tools.Eval_Metric import correlation, NS, KGE
from MCPBRNN_lib_tools.Loss_Function import  KGELoss

parser = argparse.ArgumentParser()

parser.add_argument('--case_no',
                        type=int,
                        default=0,
                        help="Case Number")

parser.add_argument('--epoch_no',
                        type=int,
                        default=3000,
                        help="number of epoch to train the network")

parser.add_argument('--depth_size',
                        type=int,
                        default=1,
                        help="layer depth")

parser.add_argument('--time_lag',
                        type=int,
                        default=0,
                        help="number of input time lag for the gate")

parser.add_argument('--gate_dim_o',
                        type=int,
                        default=2,
                        help="dim of output gate")

parser.add_argument('--gate_dim_l',
                        type=int,
                        default=1,
                        help="dim of loss gate")

parser.add_argument('--seed_no',
                        type=int,
                        default=2925,
                        help="specify torch random seed")

parser.add_argument('--c_mean',
                        type=float,
                        default=412.9139085,
                        help="40-year mean of cell state")

parser.add_argument('--c_std',
                        type=float,
                        default=77.49943867,
                        help="40-year std of cell state")

cfg = vars(parser.parse_args())

# setup random seed
seed_no = cfg["seed_no"]
np.random.seed(seed_no)
torch.manual_seed(seed_no)

# Define timesteps
lag = 0
time_lag = cfg["time_lag"]
spinLen = 1095 - time_lag - lag
traintimeLen = 8400 - time_lag - lag
timeLen = 15705 - lag 
timeLenR = 15705 - lag - spinLen

# Define Matrix size & Hyperparameters
input_size = 1
hidden_sizeM = 1#cfg["hidden_size"]
hidden_size = 1
depth_size = cfg["depth_size"]
gate_dim_o = cfg["gate_dim_o"]
gate_dim_l = cfg["gate_dim_l"]
c_mean = cfg["c_mean"]
c_std = cfg["c_std"]

num_features = 1
seq_length = 1
input_size_dyn = 1
input_size_stat = 0
num_output = 1
num_epochs = cfg["epoch_no"]
batch_size = timeLen #len_train+len_valid+len_test-seq_length+1
learning_rate = 0.025
learning_rates = {300: 0.0125, 600: 0.0125}
gate_dim=1

parent_dir = "/home/u9/yhwang0730/PB-LSTM-Papers/20230412-Single-Node-Cases"
#parent_dir = "/Users/yhwang/Desktop/HPC_DownloadTemp/2023-Spring-New/20230412-Single-Node-Cases" 

# Define case & directory
#CaseName = 'MCPBRNN_Generic_' + str(depth_size) + 'Layer_1node_' + 'outGatedim' + str(gate_dim_o) + '_lossGatedim' + str(gate_dim_l) + '_' + str(cfg["case_no"]) 
CaseName = 'MCPBRNN_Generic_' + str(depth_size) + 'Layer_1node_' + 'outGatedim' + str(gate_dim_o) + '_lossGate_sigmoid_dim' + str(gate_dim_l) + '_' + str(cfg["case_no"])
if gate_dim_l==1 and gate_dim_o==1:
   Ini_dir = "M5-case_Trial1/MCPBRNN_Generic_PETconstraint_Scaling_M5_1Layer_1node_1/model_epoch27.pt"
if gate_dim_l==1 and gate_dim_o==2:
   Ini_dir = "ComplexG-out-dim1/MCPBRNN_Generic_1Layer_1node_outGatedim1_lossGate_sigmoid_dim1_2_Training_M3/model_epoch5.pt"
elif gate_dim_l==1 and gate_dim_o==3: 
   Ini_dir = "ComplexG-out-dim2/MCPBRNN_Generic_1Layer_1node_outGatedim2_lossGate_sigmoid_dim1_4/model_epoch7.pt"
elif gate_dim_l==1 and gate_dim_o==4:
   Ini_dir = "ComplexG-out-dim3/MCPBRNN_Generic_1Layer_1node_outGatedim3_lossGate_sigmoid_dim1_6/model_epoch10.pt"
elif gate_dim_l==1 and gate_dim_o==5:
   Ini_dir = "ComplexG-out-dim4/MCPBRNN_Generic_1Layer_1node_outGatedim4_lossGate_sigmoid_dim1_2/model_epoch19.pt"

directory = CaseName
path = os.path.join(parent_dir, directory)
isExist = os.path.exists(path)
if not isExist:
  os.mkdir(path)
  print("The new directory is created!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Import Data
F_data = pd.read_csv('../20220527-MDUPLEX-LeafRiver/LeafRiverDaily_43YR.txt', header=None, delimiter=r"\s+")
F_data = F_data.rename(columns={0: 'P', 1: 'PET', 2: 'Q'})

# Skill Flag Training
SkillFlag = pd.read_csv('../20220527-MDUPLEX-LeafRiver/LeafRiverDaily_43YR_Flag.txt', header=None, delimiter=r"\s+")
SkillFlag = SkillFlag.rename(columns={0: 'Flag'})
SF = torch.tensor(SkillFlag['Flag'])
Mask_Test = SF.eq(1).unsqueeze(1)
Mask_Select = SF.eq(0).unsqueeze(1)
Mask_Train = SF.eq(-1).unsqueeze(1)
Mask_Spin = SF.eq(-99999).unsqueeze(1)

Xmean1=0.0
Xmean2=0.0
Xmean3=0.0
Xstd1=1.0
Xstd2=1.0
Xstd3=1.0
#print(F2_data['Qsim'])
F_data['P'][:] = (F_data['P'][:]-Xmean1)/Xstd1
F_data['Q'][:] = (F_data['Q'][:]-Xmean2)/Xstd2
F_data['PET'][:] = (F_data['PET'][:]-Xmean3)/Xstd3

# Define output matrix
skillmetric = 7
par_no = 6 + 2 * gate_dim_o
OutMatrix = np.zeros((num_epochs,par_no+skillmetric*4+3))

# Define output index
Out_hidden = np.zeros((timeLen,num_epochs))
Out_cell = np.zeros((timeLen,num_epochs))
Out_loss = np.zeros((timeLen,num_epochs))
Out_loss_c = np.zeros((timeLen,num_epochs))
Out_bypass = np.zeros((timeLen,num_epochs))
Out_i_gate = np.zeros((timeLen,num_epochs))
Out_o_gate = np.zeros((timeLen,num_epochs))
Out_l_gate = np.zeros((timeLen,num_epochs))
Out_lc_gate = np.zeros((timeLen,num_epochs))
Out_f_gate = np.zeros((timeLen,num_epochs))
#Teaching_Flag = np.zeros((timeLen,num_epochs))

x_data1 = np.array(F_data['P']).reshape(timeLen,-1)
x_data2 = np.array(F_data['PET']).reshape(timeLen,-1)
x_data = np.concatenate((x_data1, x_data2), axis=1)
y_data = np.array(F_data['Q']).reshape(timeLen,-1)

train_x = x_data[0:timeLen,:]
train_y = y_data[0:timeLen,:]

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

class Model_PET(nn.Module):

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 spinLen: int,
                 traintimeLen: int,
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):

        super(Model_PET, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_sizeM
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen        
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout       
        self.batch_size = batch_size        
        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_Scaling(input_size=input_size_dyn,
                         hidden_size=hidden_sizeM,
                         gate_dim=gate_dim,
                         spinLen=spinLen,
                         traintimeLen=traintimeLen,              
                         initial_forget_bias=initial_forget_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_d, epoch, time_lag, y_eval,c_mean, c_std):

        h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, olc, f, hn, obs_std = self.MCPBRNNNode(x_d, epoch, time_lag, y_eval, c_mean, c_std)    
        last_h = self.dropout(h_t)      
        out = last_h        
        return out, h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, olc, f

model_PET = Model_PET(input_size_dyn=input_size_dyn,
              input_size_stat=input_size_stat,
              hidden_size=hidden_size,
              spinLen=spinLen,
              traintimeLen=traintimeLen,              
              initial_forget_bias=0,
              dropout=0).to(device)

class Model_complex_previous(nn.Module):

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 spinLen: int,
                 traintimeLen: int,
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):

        super(Model_complex_previous, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen        
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout       
        self.batch_size = batch_size        
        self.MCPBRNNNode = MCPBRNN_Generic_OutputANNGate_PETconstraint(input_size=input_size_dyn,
                         hidden_size=hidden_size,
                         gate_dim_o=gate_dim_o-1,
                         gate_dim_l=gate_dim_l,                         
                         spinLen=spinLen,
                         traintimeLen=traintimeLen,              
                         initial_forget_bias=initial_forget_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_d, epoch, time_lag, y_eval, c_mean, c_std):

        h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, olc, f, h_nout, obs_std = self.MCPBRNNNode(x_d, epoch, time_lag, y_eval, c_mean, c_std)    
        last_h = self.dropout(h_t)      
        out = last_h        
        return out, h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, olc, f

model_cp = Model_complex_previous(input_size_dyn=input_size_dyn,
              input_size_stat=input_size_stat,
              hidden_size=hidden_size,
              spinLen=spinLen,
              traintimeLen=traintimeLen,              
              initial_forget_bias=0,
              dropout=0).to(device)

class Model(nn.Module):

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 spinLen: int,
                 traintimeLen: int,
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):

        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen        
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout       
        self.batch_size = batch_size        
        self.MCPBRNNNode = MCPBRNN_Generic_OutputANNGate_PETconstraint(input_size=input_size_dyn,
                         hidden_size=hidden_size,
                         gate_dim_o=gate_dim_o,
                         gate_dim_l=gate_dim_l,                         
                         spinLen=spinLen,
                         traintimeLen=traintimeLen,              
                         initial_forget_bias=initial_forget_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_d, epoch, time_lag, y_eval,c_mean, c_std):

        h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, olc, f, h_nout, obs_std = self.MCPBRNNNode(x_d, epoch, time_lag, y_eval, c_mean, c_std)    
        last_h = self.dropout(h_t)      
        out = last_h        
        return out, h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, olc, f

model = Model(input_size_dyn=input_size_dyn,
              input_size_stat=input_size_stat,
              hidden_size=hidden_size,
              spinLen=spinLen,
              traintimeLen=traintimeLen,              
              initial_forget_bias=0,
              dropout=0).to(device)

loss_func = KGELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = Data.TensorDataset(trainx, trainy)
loader = Data.DataLoader(
         dataset=train_dataset, 
         batch_size=batch_size, 
         shuffle=False, num_workers=0)

# Set up initial value
#rand_num = np.random.rand(1,2)
#rand_num = np.random.normal(0,1,1)
# Save the initial value of parameters
for name, param in model.state_dict().items():
   if model.state_dict()[name][:].dim() == 1:
      YHW=np.random.normal(0,1,size=(model.state_dict()[name][:].shape[0]))
      model.state_dict()[name][:]=torch.Tensor(YHW)
   elif model.state_dict()[name][:].dim() == 2:
      YHW=np.random.normal(0,1,size=(model.state_dict()[name][:].shape[0],model.state_dict()[name][:].shape[1]))
      model.state_dict()[name][:]=torch.Tensor(YHW)
   elif model.state_dict()[name][:].dim() == 3:
      YHW=np.random.normal(0,1,size=(model.state_dict()[name][:].shape[0],model.state_dict()[name][:].shape[1],model.state_dict()[name][:].shape[2]))
      model.state_dict()[name][:]=torch.Tensor(YHW)

name = "MCPBRNNNode.relu_bias_o"
YHW=np.random.normal(0,0.1,size=(model.state_dict()[name][:].shape[0],model.state_dict()[name][:].shape[1]))
model.state_dict()[name][:]=torch.Tensor(YHW)

weight_IC_file = Ini_dir
PET_file = "M5-case_Trial1/MCPBRNN_Generic_PETconstraint_Scaling_M5_1Layer_1node_1/model_epoch27.pt"
if gate_dim_o==2 and gate_dim_l==1:
#   model_PET.load_state_dict(torch.load(weight_IC_file))
#   model.state_dict()["MCPBRNNNode.weight_r_yom"][:]= model_PET.state_dict()["MCPBRNNNode.weight_r_yom"][:]
#   model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]= model_PET.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
#   model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]= model_PET.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
#   model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_yom"][:]
#   model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
#   model.state_dict()["MCPBRNNNode.weight_b2_ylm"][0][0] = model_PET.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
#   model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0] = model_PET.state_dict()["MCPBRNNNode.weight_b1_yom"][:]
   model_cp.load_state_dict(torch.load(weight_IC_file))
   model_PET.load_state_dict(torch.load(PET_file))
   model.state_dict()["MCPBRNNNode.weight_r_yom"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yom"][:]
   model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
   model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_yom"][:]
   model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
   model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:] = model_cp.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
   model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0] = model_cp.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]
elif gate_dim_o==3 and gate_dim_l==1:
   model_cp.load_state_dict(torch.load(weight_IC_file))     
   model_PET.load_state_dict(torch.load(PET_file))
   model.state_dict()["MCPBRNNNode.weight_r_yom"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yom"][:]
   model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
   model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_yom"][:]
   model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0]   
   model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:] = model_cp.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
   model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0] = model_cp.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]
elif gate_dim_o==4 and gate_dim_l==1:
   model_cp.load_state_dict(torch.load(weight_IC_file))
   model_PET.load_state_dict(torch.load(PET_file))
   model.state_dict()["MCPBRNNNode.weight_r_yom"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yom"][:]
   model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
   model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_yom"][:]
   model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][2][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][2][0]
   model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:] = model_cp.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
   model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0] = model_cp.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]
elif gate_dim_o==5 and gate_dim_l==1:
   model_cp.load_state_dict(torch.load(weight_IC_file))
   model_PET.load_state_dict(torch.load(PET_file))
   model.state_dict()["MCPBRNNNode.weight_r_yom"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yom"][:]
   model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]= model_cp.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
   model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_yom"][:]
   model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]= model_cp.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][2][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][2][0]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][3][0] = model_cp.state_dict()["MCPBRNNNode.weight_b1_yom"][3][0]
   model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:] = model_cp.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
   model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0] = model_cp.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]
elif gate_dim_o==1 and gate_dim_l==1:
   model_PET.load_state_dict(torch.load(weight_IC_file))
   model.state_dict()["MCPBRNNNode.weight_r_yom"][:]= model_PET.state_dict()["MCPBRNNNode.weight_r_yom"][:]
   model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]= model_PET.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]= model_PET.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
   model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_yom"][:]
   model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]= model_PET.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_b2_ylm"][0][0] = model_PET.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
   model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0] = model_PET.state_dict()["MCPBRNNNode.weight_b1_yom"][:]

savetext = CaseName + '/' +'model_epoch0.pt'
torch.save(model.state_dict(), savetext)

for name, param in model.state_dict().items():
    print(name)
    print(param)

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
        #predictions = model(x,epoch, time_lag, y)[0]  
        predictions = model(x,epoch, time_lag, y,c_mean, c_std)[1]      
        sim = torch.masked_select(predictions, Mask_Train).unsqueeze(1)       
        obs =  torch.masked_select(y, Mask_Train).unsqueeze(1)     
        loss = loss_func(sim, obs)        
        loss.backward()
        optimizer.step()
        model.eval()
        Result = model(x,epoch, time_lag, y,c_mean, c_std)
        predictions = Result[1]
        sim = torch.masked_select(predictions, Mask_Train).unsqueeze(1)
        obs =  torch.masked_select(y, Mask_Train).unsqueeze(1)
        sim_select = torch.masked_select(predictions, Mask_Select).unsqueeze(1)
        obs_select =  torch.masked_select(y, Mask_Select).unsqueeze(1)
        sim_test = torch.masked_select(predictions, Mask_Test).unsqueeze(1)
        obs_test =  torch.masked_select(y, Mask_Test).unsqueeze(1)
        sim_spinup = torch.masked_select(predictions, Mask_Spin).unsqueeze(1)
        obs_spinup =  torch.masked_select(y, Mask_Spin).unsqueeze(1)

    for name, param in model.state_dict().items():
        print(name)
        print(param)

    hidden_eval = Result[1]
    cell_eval = Result[2]
    loss_eval = Result[3]
    lossc_eval = Result[4]    
    bypass_eval = Result[5]
    i_gate = Result[6]
    o_gate = Result[7]
    l_gate = Result[8]
    lc_gate = Result[9]    
    f_gate = Result[10]

    yout_eval = y.detach().cpu().numpy()    
    pout_eval = predictions.detach().cpu().numpy()   
    hidden_eval = hidden_eval.detach().cpu().numpy()       
    cell_eval = cell_eval.detach().cpu().numpy()
    loss_eval = loss_eval.detach().cpu().numpy() 
    lossc_eval = lossc_eval.detach().cpu().numpy()     
    bypass_eval = bypass_eval.detach().cpu().numpy()    
    i_gate = i_gate.detach().cpu().numpy()
    o_gate = o_gate.detach().cpu().numpy()
    l_gate = l_gate.detach().cpu().numpy()
    lc_gate = lc_gate.detach().cpu().numpy()    
    f_gate = f_gate.detach().cpu().numpy()      

    #calculate the skill
    sim = sim.detach().cpu().numpy()  
    obs = obs.detach().cpu().numpy()  
    sim_spinup = sim_spinup.detach().cpu().numpy()   
    obs_spinup = obs_spinup.detach().cpu().numpy()  
    sim_select = sim_select.detach().cpu().numpy()   
    obs_select = obs_select.detach().cpu().numpy()  
    sim_test = sim_test.detach().cpu().numpy()        
    obs_test = obs_test.detach().cpu().numpy()  

    pout_eval[pout_eval < 0] = 0
    hidden_eval[hidden_eval < 0] = 0

    [B, E, C, D, G] =KGE(sim, obs)
    A = NS(sim, obs)
    F = mean_squared_error(obs, sim)

    [B2, E2, C2, D2, G2] =KGE(sim_select, obs_select)
    A2 = NS(sim_select, obs_select)
    F2 = mean_squared_error(obs_select, sim_select)

    sz = obs.shape[0]
    [KGEtimelag_1, X1, X2, X3, X4] =KGE(sim[0+1:sz], obs[0:sz-1])   
    [KGEtimelag_2, X1, X2, X3, X4] =KGE(sim[0+2:sz], obs[0:sz-2])   
    [KGEtimelag_3, X1, X2, X3, X4] =KGE(sim[0+3:sz], obs[0:sz-3])   
    OutMatrix[epoch-1,par_no+28] = KGEtimelag_1
    OutMatrix[epoch-1,par_no+29] = KGEtimelag_2
    OutMatrix[epoch-1,par_no+30] = KGEtimelag_3

    [B3, E3, C3, D3, G3] =KGE(sim_test, obs_test)
    A3 = NS(sim_test, obs_test)
    F3 = mean_squared_error(obs_test, sim_test)    

    [B4, E4, C4, D4, G4] =KGE(sim_spinup, obs_spinup)
    A4 = NS(sim_spinup, obs_spinup)
    F4 = mean_squared_error(obs_spinup, sim_spinup)

    OutMatrix[epoch-1,par_no] = A
    OutMatrix[epoch-1,par_no+7] = A2    
    OutMatrix[epoch-1,par_no+14] = A3 
    OutMatrix[epoch-1,par_no+21] = A4 

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
       OutMatrix[epoch-1,par_no+22] = B4

    OutMatrix[epoch-1,par_no+2] = C
    OutMatrix[epoch-1,par_no+3] = D 
    OutMatrix[epoch-1,par_no+9] = C2
    OutMatrix[epoch-1,par_no+10] = D2     
    OutMatrix[epoch-1,par_no+16] = C3
    OutMatrix[epoch-1,par_no+17] = D3 
    OutMatrix[epoch-1,par_no+23] = C4
    OutMatrix[epoch-1,par_no+24] = D4 

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
       OutMatrix[epoch-1,par_no+25] = E4

    OutMatrix[epoch-1,par_no+5] = F
    OutMatrix[epoch-1,par_no+12] = F2 
    OutMatrix[epoch-1,par_no+19] = F3 
    OutMatrix[epoch-1,par_no+26] = F4 

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
       OutMatrix[epoch-1,par_no+27] = G4

    if gate_dim_o==2:
      OutMatrix[epoch-1,0]= model.state_dict()["MCPBRNNNode.weight_r_yom"][:]
      OutMatrix[epoch-1,1]= model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
      OutMatrix[epoch-1,2]= model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
      OutMatrix[epoch-1,3]= model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
      OutMatrix[epoch-1,4]= model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
      OutMatrix[epoch-1,5]= model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]
      OutMatrix[epoch-1,6]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
      OutMatrix[epoch-1,7]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0]      
      OutMatrix[epoch-1,8]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]   
      OutMatrix[epoch-1,9]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][1]        
    elif gate_dim_o==3:
      OutMatrix[epoch-1,0]= model.state_dict()["MCPBRNNNode.weight_r_yom"][:]
      OutMatrix[epoch-1,1]= model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
      OutMatrix[epoch-1,2]= model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
      OutMatrix[epoch-1,3]= model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
      OutMatrix[epoch-1,4]= model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
      OutMatrix[epoch-1,5]= model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]
      OutMatrix[epoch-1,6]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
      OutMatrix[epoch-1,7]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0] 
      OutMatrix[epoch-1,8]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][2][0]            
      OutMatrix[epoch-1,9]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]   
      OutMatrix[epoch-1,10]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][1]  
      OutMatrix[epoch-1,11]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][2] 
    elif gate_dim_o==4:
      OutMatrix[epoch-1,0]= model.state_dict()["MCPBRNNNode.weight_r_yom"][:]
      OutMatrix[epoch-1,1]= model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
      OutMatrix[epoch-1,2]= model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
      OutMatrix[epoch-1,3]= model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
      OutMatrix[epoch-1,4]= model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
      OutMatrix[epoch-1,5]= model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]
      OutMatrix[epoch-1,6]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
      OutMatrix[epoch-1,7]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0]
      OutMatrix[epoch-1,8]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][2][0]
      OutMatrix[epoch-1,9]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][3][0]
      OutMatrix[epoch-1,10]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]
      OutMatrix[epoch-1,11]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][1]
      OutMatrix[epoch-1,12]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][2]
      OutMatrix[epoch-1,13]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][3]
    elif gate_dim_o==5:
      OutMatrix[epoch-1,0]= model.state_dict()["MCPBRNNNode.weight_r_yom"][:]
      OutMatrix[epoch-1,1]= model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
      OutMatrix[epoch-1,2]= model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
      OutMatrix[epoch-1,3]= model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
      OutMatrix[epoch-1,4]= model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
      OutMatrix[epoch-1,5]= model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]
      OutMatrix[epoch-1,6]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
      OutMatrix[epoch-1,7]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][1][0]
      OutMatrix[epoch-1,8]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][2][0]
      OutMatrix[epoch-1,9]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][3][0]
      OutMatrix[epoch-1,10]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][4][0]
      OutMatrix[epoch-1,11]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]
      OutMatrix[epoch-1,12]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][1]
      OutMatrix[epoch-1,13]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][2]
      OutMatrix[epoch-1,14]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][3]
      OutMatrix[epoch-1,15]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][4]
    elif gate_dim_o==1:
      OutMatrix[epoch-1,0]= model.state_dict()["MCPBRNNNode.weight_r_yom"][:]
      OutMatrix[epoch-1,1]= model.state_dict()["MCPBRNNNode.weight_r_ylm"][:]
      OutMatrix[epoch-1,2]= model.state_dict()["MCPBRNNNode.weight_r_yfm"][:]
      OutMatrix[epoch-1,3]= model.state_dict()["MCPBRNNNode.bias_b0_ylm"][:]
      OutMatrix[epoch-1,4]= model.state_dict()["MCPBRNNNode.weight_b2_ylm"][:]
      OutMatrix[epoch-1,5]= model.state_dict()["MCPBRNNNode.bias_ln_yom"][:]
      OutMatrix[epoch-1,6]= model.state_dict()["MCPBRNNNode.weight_b1_yom"][0][0]
      OutMatrix[epoch-1,7]= model.state_dict()["MCPBRNNNode.relu_bias_o"][0][0]

# Output time series
    #Out_hidden[0:timeLen,epoch-1] = hidden_eval[:,0]       
    #Out_cell[0:timeLen,epoch-1] = cell_eval[:,0]
    #Out_loss[0:timeLen,epoch-1] = loss_eval[:,0]
    #Out_bypass[0:timeLen,epoch-1] = bypass_eval[:,0]
    #Out_i_gate [0:timeLen,epoch-1] = i_gate[:,0]
    #Out_o_gate [0:timeLen,epoch-1] = o_gate[:,0]
    #Out_l_gate [0:timeLen,epoch-1] = l_gate[:,0]    
    #Out_f_gate [0:timeLen,epoch-1] = f_gate[:,0]

    print(B2)
    torch.save(model.state_dict(), savetext)

# Save Results file
if gate_dim_o==2:
   outtofile = pd.DataFrame(OutMatrix, columns=['MCPBRNNNode.weight_r_yom','MCPBRNNNode.weight_r_ylm','MCPBRNNNode.weight_r_yfm','MCPBRNNNode.bias_b0_ylm','MCPBRNNNode.weight_b1_ylm','MCPBRNNNode.bias_ln_yom','MCPBRNNNode.weight_b1_yom_0','MCPBRNNNode.weight_b1_yom_1','MCPBRNNNode.relu_bias_o_0','MCPBRNNNode.relu_bias_o_1','NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])
elif gate_dim_o==3:
   outtofile = pd.DataFrame(OutMatrix, columns=['MCPBRNNNode.weight_r_yom','MCPBRNNNode.weight_r_ylm','MCPBRNNNode.weight_r_yfm','MCPBRNNNode.bias_b0_ylm','MCPBRNNNode.weight_b1_ylm','MCPBRNNNode.bias_ln_yom','MCPBRNNNode.weight_b1_yom_0','MCPBRNNNode.weight_b1_yom_1','MCPBRNNNode.weight_b1_yom_2','MCPBRNNNode.relu_bias_o_0','MCPBRNNNode.relu_bias_o_1','MCPBRNNNode.relu_bias_o_2','NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])
elif gate_dim_o==4:
   outtofile = pd.DataFrame(OutMatrix, columns=['MCPBRNNNode.weight_r_yom','MCPBRNNNode.weight_r_ylm','MCPBRNNNode.weight_r_yfm','MCPBRNNNode.bias_b0_ylm','MCPBRNNNode.weight_b1_ylm','MCPBRNNNode.bias_ln_yom','MCPBRNNNode.weight_b1_yom_0','MCPBRNNNode.weight_b1_yom_1','MCPBRNNNode.weight_b1_yom_2','MCPBRNNNode.weight_b1_yom_3','MCPBRNNNode.relu_bias_o_0','MCPBRNNNode.relu_bias_o_1','MCPBRNNNode.relu_bias_o_2','MCPBRNNNode.relu_bias_o_3','NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])
elif gate_dim_o==5:
   outtofile = pd.DataFrame(OutMatrix, columns=['MCPBRNNNode.weight_r_yom','MCPBRNNNode.weight_r_ylm','MCPBRNNNode.weight_r_yfm','MCPBRNNNode.bias_b0_ylm','MCPBRNNNode.weight_b1_ylm','MCPBRNNNode.bias_ln_yom','MCPBRNNNode.weight_b1_yom_0','MCPBRNNNode.weight_b1_yom_1','MCPBRNNNode.weight_b1_yom_2','MCPBRNNNode.weight_b1_yom_3','MCPBRNNNode.weight_b1_yom_4','MCPBRNNNode.relu_bias_o_0','MCPBRNNNode.relu_bias_o_1','MCPBRNNNode.relu_bias_o_2','MCPBRNNNode.relu_bias_o_3','MCPBRNNNode.relu_bias_o_4','NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])
elif gate_dim_o==1:
   outtofile = pd.DataFrame(OutMatrix, columns=['MCPBRNNNode.weight_r_yom','MCPBRNNNode.weight_r_ylm','MCPBRNNNode.weight_r_yfm','MCPBRNNNode.bias_b0_ylm','MCPBRNNNode.weight_b1_ylm','MCPBRNNNode.bias_ln_yom','MCPBRNNNode.weight_b1_yom_0','MCPBRNNNode.relu_bias_o_0','NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])


#outtofile = pd.DataFrame(OutMatrix, columns=['NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])

outname = CaseName + '/' + 'IC_caseno_' + str(cfg["case_no"]) + '_summary.csv'
outtofile.to_csv(outname)
# Save timeseries
#Out_file = pd.DataFrame(Out_hidden)
#outname = CaseName + '/' + 'Outhidden_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)

#Out_file = pd.DataFrame(Out_cell)
#outname = CaseName + '/' + 'Outcell_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)

#Out_file = pd.DataFrame(Out_loss)
#outname = CaseName + '/' + 'Outloss_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)

#Out_file = pd.DataFrame(Out_bypass)
#outname = CaseName + '/' + 'Outbypass_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)

#Out_file = pd.DataFrame(Out_i_gate)
#outname = CaseName + '/' + 'Out_gatei_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)

#Out_file = pd.DataFrame(Out_o_gate)
#outname = CaseName + '/' + 'Out_gateo_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)

#Out_file = pd.DataFrame(Out_l_gate)
#outname = CaseName + '/' + 'Out_gatel_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)

#Out_file = pd.DataFrame(Out_f_gate)
#outname = CaseName + '/' + 'Out_gatef_' + str(cfg["case_no"]) + '_summary.csv'
#Out_file.to_csv(outname)
