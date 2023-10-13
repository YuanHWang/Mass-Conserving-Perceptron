import torch
import torch.nn as nn
from torch import Tensor

class ARX_LeafRiver_Qsim(nn.Module):

    def __init__(self,
                 input_size: int,
                 batch_first: bool = True):
        super(ARX_LeafRiver_Qsim, self).__init__()

        self.input_size = input_size    
        self.batch_first = batch_first                         
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size,1))
        self.weight_y = nn.Parameter(torch.FloatTensor(1,1))        
        self.bias = nn.Parameter(torch.FloatTensor(1))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters"""
        self.weight = nn.Parameter(torch.rand(self.input_size,1))
        self.weight_y = nn.Parameter(torch.rand(1,1))        
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, x):

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()
        h_n = torch.zeros([batch_size, 1])
        y_hs = x.data.new(1, 1).zero_()
        y_hs_x = y_hs
        bias = self.bias.unsqueeze(0).expand(1, 1)

        for t in range(batch_size):

            y_hs = y_hs_x

            y_h = torch.addmm(bias, x[0,t,:].unsqueeze(0).expand(1, x.shape[2]), self.weight) + torch.mm(y_hs, self.weight_y)
            h_n[t,:] = y_h 

            y_hs_x = y_hs

        return h_n

class ANN_LeafRiver_Sigmoid_qsim(nn.Module):

    def __init__(self,
                 input_size: int,
                 batch_first: bool = True,
                 hidden_size: int = 1):
        super(ANN_LeafRiver_Sigmoid_qsim, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()        
        # create tensors of learnable parameters                   
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size,self.hidden_size))
        self.weight_y = nn.Parameter(torch.FloatTensor(1,self.hidden_size))        
        self.bias = nn.Parameter(torch.FloatTensor(1))
        self.weight_ln = nn.Parameter(torch.FloatTensor(self.hidden_size,1))
        self.bias_ln = nn.Parameter(torch.FloatTensor(1))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters"""
        self.weight = nn.Parameter(torch.rand(self.input_size,self.hidden_size))
        self.weight_y = nn.Parameter(torch.rand(1,self.hidden_size))        
        self.bias = nn.Parameter(torch.rand(1))
        self.weight_ln = nn.Parameter(torch.rand(self.hidden_size,1))
        self.bias_ln = nn.Parameter(torch.rand(1))

    def forward(self, x):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        h_n = torch.zeros([batch_size, 1])
        y_hs = x.data.new(1, self.hidden_size).zero_()
        y_hs_x = y_hs
        bias = self.bias.unsqueeze(0).expand(1, self.hidden_size)

        for t in range(batch_size):

            y_hs = y_hs_x

            y_h = torch.addmm(bias, x[0,t,:].unsqueeze(0).expand(1, x.shape[2]), self.weight) + torch.mul(y_hs, self.weight_y)
            y_hs = self.Sigmoid(y_h)
            y_out = torch.addmm(self.bias_ln, y_hs, self.weight_ln)    
            h_n[t,:] = y_out 

            y_hs_x = y_hs

        return h_n,y_h,y_hs

class Vanilla_RNN_Sigmoid_Qsim_SingleNode(nn.Module):

    def __init__(self,
                 input_size: int,
                 batch_first: bool = True,
                 hidden_size: int = 1):
        super(Vanilla_RNN_Sigmoid_Qsim_SingleNode, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()        
        # create tensors of learnable parameters                   
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size, self.hidden_size))
        self.weight_h = nn.Parameter(torch.FloatTensor(1, self.hidden_size))        
        self.bias = nn.Parameter(torch.FloatTensor(1))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters"""
        self.weight = nn.Parameter(torch.rand(self.input_size,self.hidden_size))
        self.weight_h = nn.Parameter(torch.rand(1,self.hidden_size))        
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, x):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        h_n = torch.zeros([batch_size, 1])
        bias = self.bias.unsqueeze(0).expand(1, self.hidden_size)
        y_hs = x.data.new(1, self.hidden_size).zero_()
        y_hs_x = y_hs

        for t in range(batch_size):

            y_hs = y_hs_x

            y_h = torch.addmm(bias, x[0,t,:].unsqueeze(0).expand(1, x.shape[2]), self.weight) + torch.mul(y_hs, self.weight_h)
            y_hs = y_h#self.Sigmoid(y_h)
            y_out = y_h#torch.addmm(self.bias_ln, y_hs, self.weight_ln)    
            h_n[t,:] = y_out 

            y_hs_x = y_hs

        return h_n,y_h,y_hs

class Vanilla_RNN_Sigmoid_Qsim(nn.Module):

    def __init__(self,
                 input_size: int,
                 batch_first: bool = True,
                 hidden_size: int = 1):
        super(Vanilla_RNN_Sigmoid_Qsim, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()        
        # create tensors of learnable parameters                   
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size, self.hidden_size))
        self.weight_h = nn.Parameter(torch.FloatTensor(1, self.hidden_size))        
        self.bias = nn.Parameter(torch.FloatTensor(1))
        self.weight_ln = nn.Parameter(torch.FloatTensor(self.hidden_size,1))
        self.bias_ln = nn.Parameter(torch.FloatTensor(1))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters"""
        self.weight = nn.Parameter(torch.rand(self.input_size,self.hidden_size))
        self.weight_h = nn.Parameter(torch.rand(1,self.hidden_size))        
        self.bias = nn.Parameter(torch.rand(1))
        self.weight_ln = nn.Parameter(torch.rand(self.hidden_size,1))
        self.bias_ln = nn.Parameter(torch.rand(1))

    def forward(self, x):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        h_n = torch.zeros([batch_size, 1])
        bias = self.bias.unsqueeze(0).expand(1, self.hidden_size)
        y_hs = x.data.new(1, self.hidden_size).zero_()
        y_hs_x = y_hs

        for t in range(batch_size):

            y_hs = y_hs_x

            y_h = torch.addmm(bias, x[0,t,:].unsqueeze(0).expand(1, x.shape[2]), self.weight) + torch.mul(y_hs, self.weight_h)
            y_hs = self.Sigmoid(y_h)
            y_out = torch.addmm(self.bias_ln, y_hs, self.weight_ln)    
            h_n[t,:] = y_out 

            y_hs_x = y_hs

        return h_n,y_h,y_hs
