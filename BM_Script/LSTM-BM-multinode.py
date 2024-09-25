# LSTM script uses here follows Kratzert et al., (2019) HESS
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
from MCPBRNN_lib_tools.Eval_Metric import ANLL_out, correlation, NS, KGE
from MCPBRNN_lib_tools.Loss_Function import  KGELoss, ANLL_type1, ANLL_type2, MSELoss

parser = argparse.ArgumentParser()

parser.add_argument('--case_no',
                        type=int,
                        default=0,
                        help="Case Number: Initial values for the parameters of Hydro-MC-simple-LSTM")

parser.add_argument('--epoch_no',
                        type=int,
                        default=2000,
                        help="number of epoch to train the network")

parser.add_argument('--seed_no',
                        type=int,
                        default=2925,
                        help="specify torch random seed")

parser.add_argument('--hidden_size',
                        type=int,
                        default=1,
                        help="specify the number of mcpbrnn Node")

parser.add_argument('--seq_length',
                        type=int,
                        default=365,
                        help="")

parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help="")

cfg = vars(parser.parse_args())

# setup random seed
seed_no = cfg["seed_no"]
np.random.seed(seed_no)
torch.manual_seed(seed_no)
input_size_dyn = 1
# Define timesteps
hidden_size = cfg["hidden_size"]
seq_length = cfg["seq_length"]
batch_size = cfg["batch_size"]
num_features = 2
num_output = 1
num_epochs = cfg["epoch_no"]
learning_rate = 0.0125
learning_rates = {300: 0.0125, 600: 0.0125}

# Define case & directory
CaseName = 'LSTM_BM_Nodeno_' + str(hidden_size) + '_batchsize_' + str(batch_size) + '_seq_length_' + str(seq_length) + '_caseno' + str(cfg["case_no"]) 
directory = CaseName
parent_dir = "/groups/maliaca/yhwang0730/PB-LSTM-temp/20230223-LSTM"
path = os.path.join(parent_dir, directory)
isExist = os.path.exists(path)
if not isExist:
  os.mkdir(path)
  print("The new directory is created!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Import Data
F_data = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR.txt', header=None, delimiter=r"\s+")
F_data = F_data.rename(columns={0: 'P', 1: 'PET', 2: 'Q'})

# Skill Flag Training
SkillFlag = pd.read_csv('../20230129-ML-BM-MDUPLEX-LeafRiver/LeafRiverDaily_40YR_Flag.txt', header=None, delimiter=r"\s+")
SkillFlag = SkillFlag.rename(columns={0: 'Flag'})

# Define output matrix
par_no = 0 # Temporarily not save the parameter value
skillmetric = 7
OutMatrix = np.zeros((num_epochs,par_no+skillmetric*4+3))
time_len = F_data['P'].shape[0]
x_data1 = np.array(F_data['P']).reshape(F_data['P'].shape[0],-1)
x_data2 = np.array(F_data['PET']).reshape(F_data['PET'].shape[0],-1)
Pmax = np.amax(x_data1)
PETmax = np.amax(x_data2)
x_data = np.concatenate((x_data1/Pmax, x_data2/PETmax), axis=1)
y_data = np.array(F_data['Q']).reshape(F_data['Q'].shape[0],-1)
Qmax = np.amax(y_data)
y_data = y_data/Qmax
DataFlag = np.array(SkillFlag['Flag']).reshape(SkillFlag['Flag'].shape[0],-1)

if seq_length>1:
   DataFlag[0:seq_length-1] = -99999

#####
input_size_dyn = x_data.shape[1]
input_size  = input_size_dyn
batchno_train = np.count_nonzero(DataFlag == -1)
batchno_select = np.count_nonzero(DataFlag == 0)
batchno_test = np.count_nonzero(DataFlag == 1)
#####
x_train_new = np.zeros((batchno_train, seq_length, num_features))
y_train_new = np.zeros((batchno_train, 1))
x_select_new = np.zeros((batchno_select, seq_length, num_features))
y_select_new = np.zeros((batchno_select, 1))
x_test_new = np.zeros((batchno_test, seq_length, num_features))
y_test_new = np.zeros((batchno_test, 1))
x_full_new = np.zeros((batchno_train + batchno_select + batchno_test, seq_length, num_features))
y_full_new = np.zeros((batchno_train + batchno_select + batchno_test, 1))

count1=-1
count2=-1
count3=-1
count4=-1
#####

for tloc in range(0, time_len):
        
    if DataFlag[tloc] == -1:
       count1 = count1 + 1
       x_train_new[count1,0:seq_length,:] = x_data[tloc-seq_length+1:tloc+1,:]
       y_train_new[count1,0] = y_data[tloc,0]
       count4 = count4 + 1
       x_full_new[count4,0:seq_length,:] = x_data[tloc-seq_length+1:tloc+1,:]
       y_full_new[count4,0] = y_data[tloc,0]

    if DataFlag[tloc] == 0:
       count2 = count2 + 1
       x_select_new[count2,0:seq_length,:] = x_data[tloc-seq_length+1:tloc+1,:]
       y_select_new[count2,0] = y_data[tloc,0]
       count4 = count4 + 1
       x_full_new[count4,0:seq_length,:] = x_data[tloc-seq_length+1:tloc+1,:]
       y_full_new[count4,0] = y_data[tloc,0]
       
    if DataFlag[tloc] == 1:
       count3 = count3 + 1
       x_test_new[count3,0:seq_length,:] = x_data[tloc-seq_length+1:tloc+1,:]
       y_test_new[count3,0] = y_data[tloc,0]
       count4 = count4 + 1
       x_full_new[count4,0:seq_length,:] = x_data[tloc-seq_length+1:tloc+1,:]
       y_full_new[count4,0] = y_data[tloc,0]

x_train_new = torch.tensor(x_train_new)
y_train_new = torch.tensor(y_train_new)
x_select_new = torch.tensor(x_select_new)
y_select_new = torch.tensor(y_select_new)
x_test_new = torch.tensor(x_test_new)
y_test_new = torch.tensor(y_test_new)
x_full_new = torch.tensor(x_full_new)
y_full_new = torch.tensor(y_full_new)

x_train_new = x_train_new.float()
y_train_new = y_train_new.float()
x_select_new = x_select_new.float()
y_select_new = y_select_new.float()
x_test_new = x_test_new.float()
y_test_new = y_test_new.float()
x_full_new = x_full_new.float()
y_full_new = y_full_new.float()

trainx, trainy = Variable(x_train_new), Variable(y_train_new)
selectx, selecty = Variable(x_select_new), Variable(y_select_new)
testx, testy = Variable(x_test_new), Variable(y_test_new)
fullx, fully = Variable(x_full_new), Variable(y_full_new)

class LSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

    def forward(self, x):

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        h_0 = x.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x
            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x[t], self.weight_ih))         
            f, i, o, g = gates.chunk(4, 1)
            
            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
            
            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        #print(h_1.shape)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        #print(h_n.shape)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)
        
        return h_n, c_n

class Model(nn.Module):

    def __init__(self,
                 input_size_dyn: int,
                 hidden_size: int,
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):

        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout

        self.lstm = LSTM(input_size=input_size_dyn,
                         hidden_size=hidden_size,
                         initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d):

        h_n, c_n = self.lstm(x_d)        
        last_h = self.dropout(h_n[:, -1, :])   
        out = self.fc(last_h)
        return out, h_n, c_n

model = Model(input_size_dyn=input_size_dyn,
              hidden_size=hidden_size,
              initial_forget_bias=5,
              dropout=0).to(device)

# Set up initial value
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

loss_func = KGELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = Data.TensorDataset(trainx, trainy)
loader_train = Data.DataLoader(
         dataset=train_dataset, 
         batch_size=batch_size, 
         shuffle=True, num_workers=0)

train_eval_dataset = Data.TensorDataset(trainx, trainy)
loader_train_eval = Data.DataLoader(
         dataset=train_dataset, 
         batch_size=batchno_train, 
         shuffle=False, num_workers=0)

select_dataset = Data.TensorDataset(selectx, selecty)
loader_select = Data.DataLoader(
         dataset=select_dataset, 
         batch_size=batchno_select, 
         shuffle=False, num_workers=0)

test_dataset = Data.TensorDataset(testx, testy)
loader_test = Data.DataLoader(
         dataset=test_dataset, 
         batch_size=batchno_test, 
         shuffle=False, num_workers=0)

full_dataset = Data.TensorDataset(fullx, fully)
loader_full = Data.DataLoader(
         dataset=full_dataset, 
         batch_size=14610-seq_length+1, 
         shuffle=False, num_workers=0)

#for name, param in model.state_dict().items():
#    print(name)
#    print(param)

savetext = CaseName + '/' +'model_epoch0.pt'
torch.save(model.state_dict(), savetext)

for epoch in range(1,num_epochs+1): 
    model.train()
    if epoch in learning_rates.keys():
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rates[epoch]

    pbar = tqdm(loader_train, file=sys.stdout)  
   # pbar.set_description(f'# Epoch {epoch}')
    pbar_traineval = tqdm(loader_train_eval, file=sys.stdout)  
   # pbar_traineval.set_description(f'# Epoch {epoch}')
    pbar_select = tqdm(loader_select, file=sys.stdout)  
   # pbar_select.set_description(f'# Epoch {epoch}')
    pbar_test = tqdm(loader_test, file=sys.stdout)  
   # pbar_test.set_description(f'# Epoch {epoch}')
    pbar_full = tqdm(loader_full, file=sys.stdout)  
   # pbar_full.set_description(f'# Epoch {epoch}')

    savetext = CaseName + '/' +'model_epoch' + str(epoch) + '.pt'
    pbar.set_description(f'# Epoch {epoch}')    

    for data in pbar:
        optimizer.zero_grad()
        x, y,= data 
        predictions = model(x)[0]         
        loss = loss_func(predictions, y)        
        loss.backward()
        optimizer.step()

    for data_traineval in pbar_traineval:
        x_traineval, y_traineval,= data_traineval 
        predictions_traineval = model(x_traineval)[0]         

    for data_select in pbar_select:
        x_select, y_select,= data_select
        predictions_select = model(x_select)[0]  

    for data_test in pbar_test:
        x_test, y_test,= data_test
        predictions_test = model(x_test)[0]  

    '''for data_full in pbar_full:
        x_full, y_full,= data_full
        predictions_full = model(x_full)[0] ''' 

    predictions_traineval = predictions_traineval.detach().cpu().numpy()  
    predictions_select = predictions_select.detach().cpu().numpy()  
    predictions_test = predictions_test.detach().cpu().numpy()  
    y_traineval = y_traineval.detach().cpu().numpy()  
    y_select = y_select.detach().cpu().numpy()  
    y_test = y_test.detach().cpu().numpy()

    predictions_traineval[predictions_traineval<0] = 0
    predictions_select[predictions_select<0] = 0
    predictions_test[predictions_test<0] = 0    

    predictions_traineval =  predictions_traineval*Qmax 
    predictions_select =  predictions_select*Qmax 
    predictions_test = predictions_test *Qmax 
    '''predictions_full = predictions_full *Qmax '''    

    y_traineval = y_traineval *Qmax 
    y_select = y_select *Qmax 
    y_test = y_test *Qmax  

    [B, E, C, D, G] =KGE(predictions_traineval, y_traineval)
    A = NS(predictions_traineval, y_traineval)
    F = mean_squared_error(y_traineval, predictions_traineval)

    sz = y_traineval.shape[0]
    [KGEtimelag_1, X1, X2, X3, X4] =KGE(predictions_traineval[0+1:sz], y_traineval[0:sz-1])   
    [KGEtimelag_2, X1, X2, X3, X4] =KGE(predictions_traineval[0+2:sz], y_traineval[0:sz-2])   
    [KGEtimelag_3, X1, X2, X3, X4] =KGE(predictions_traineval[0+3:sz], y_traineval[0:sz-3])   
    OutMatrix[epoch-1,par_no+28] = KGEtimelag_1
    OutMatrix[epoch-1,par_no+29] = KGEtimelag_2
    OutMatrix[epoch-1,par_no+30] = KGEtimelag_3

    [B2, E2, C2, D2, G2] =KGE(predictions_select, y_select)
    A2 = NS(predictions_select, y_select)
    F2 = mean_squared_error(y_select, predictions_select)

    [B3, E3, C3, D3, G3] =KGE(predictions_test, y_test)
    A3 = NS(predictions_test, y_test)
    F3 = mean_squared_error(y_test, predictions_test)    

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

    #print(B)
    #print(B2)
    #print(B3)

    torch.save(model.state_dict(), savetext)

# Save Results file
outtofile = pd.DataFrame(OutMatrix, columns=['NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup','KGEtimelag_1','KGEtimelag_2','KGEtimelag_3'])
outname = CaseName + '/' + 'IC_caseno_' + str(cfg["case_no"]) + '_summary.csv'
outtofile.to_csv(outname)

# output flowseries
indx = np.argmax(OutMatrix[:,8]) + 1
mpt = CaseName + '/model_epoch' + str(indx) + '.pt'
model.load_state_dict(torch.load(mpt))

for data_full in pbar_full:
    x_full, y_full,= data_full
    predictions_full = model(x_full)[0] 

predictions_full = predictions_full.detach().cpu().numpy()   
Out_file = pd.DataFrame(predictions_full*Qmax)
outname = CaseName + '/' + 'Outhidden_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

outtext = CaseName + '/*.pt'
files = glob.glob(outtext, recursive=True)

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
