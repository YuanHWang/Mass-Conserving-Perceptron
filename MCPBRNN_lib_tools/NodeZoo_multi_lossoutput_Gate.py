import torch
import torch.nn as nn
from torch import Tensor

class MCPBRNN_Generic_PETconstraint_MIoutputloss(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim_o: int,
                 gate_dim_l: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_MIoutputloss, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size     
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l        
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen
        self.relu_o = nn.SELU()        
        self.relu_l = nn.SELU()
        self.relu = nn.ReLU()        
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))

        self.bias_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(2, self.gate_dim_o))        
        self.weight_b2_yom = nn.Parameter(torch.FloatTensor(self.gate_dim_o, self.hidden_size))
        self.relu_bias_o = nn.Parameter(torch.FloatTensor(1, self.gate_dim_o))
        self.bias_ln_yom= nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))

        self.bias_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.FloatTensor(2, self.gate_dim_l))
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.gate_dim_l, self.hidden_size))        
        self.relu_bias_l = nn.Parameter(torch.FloatTensor(1, self.gate_dim_l))
        self.bias_ln_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))

        self.bias_yom = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.rand(2, self.gate_dim_o))        
        self.weight_b2_yom = nn.Parameter(torch.rand(self.gate_dim_o, self.hidden_size))
        self.relu_bias_o = nn.Parameter(torch.rand(1, self.gate_dim_o))
        self.bias_ln_yom= nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))

        self.bias_ylm = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.rand(2, self.gate_dim_l))
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.gate_dim_l, self.hidden_size))        
        self.relu_bias_l = nn.Parameter(torch.rand(1, self.gate_dim_l))
        self.bias_ln_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size

        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()
        c_0p1 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)
        h_xp1 = c_0p1
        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])        
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size]) 

        # expand bias vectors to batch size
        bias_yom = (self.bias_yom.unsqueeze(0).expand(1, self.gate_dim_o))
        bias_ylm = (self.bias_ylm.unsqueeze(0).expand(1, self.gate_dim_l))

        #torch.set_printoptions(precision=20)
        mo = cmean
        ml = 2.9086
        so = cstd
        sl = 1.8980

        for b in range(0, batch_size):
            for t in range(seq_len):

                h_0, c_0 = h_x

                if b>=time_lag:
                   c_0p1 = h_xp1
                    
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                c_Gate_0 = (c_0.expand(1, 1))
                c_Gate_1 = (c_0p1.expand(1, 1))
                c_0_Gate = (c_0.expand(1, 1))
                u2_Gate = (u2.expand(1, 1))
 
                ib = 0 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.cat(((c_Gate_0-mo)/so, (c_Gate_1-mo)/so), 1)
                oo3 = torch.addmm(bias_yom, oo2, self.weight_b1_yom)
                oo4 = self.relu_o(oo3 - self.relu_bias_o)
                oo = oo1 * torch.sigmoid(torch.addmm(self.bias_ln_yom, oo4, self.weight_b2_yom)) 

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = torch.cat(((c_0_Gate-mo)/so, (u2_Gate-ml)/sl), 1)
                ol3 = torch.addmm(bias_ylm, ol2, self.weight_b1_ylm)
                ol4 = self.relu_l(ol3 - self.relu_bias_l)
                ol = ol1 * torch.sigmoid(torch.addmm(self.bias_ln_ylm, ol4, self.weight_b2_ylm)) 

                if c_0 > 0:                
                    ol_constraint = ol - self.relu(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g
                h_1 = oo * c_1
                l_1 = ol * c_1
                lc_1 = ol_constraint * c_1
                bp_0 = ib * g                
                h_0 = oo * c_0
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0                
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 

                h_x = (h_1, c_1)

                if b>=time_lag:
                   h_xp1 = c_0

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f

class MCPBRNN_Generic_PETconstraint_MIoutputloss_Sigmoid(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim_o: int,
                 gate_dim_l: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_MIoutputloss_Sigmoid, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size     
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim_o = gate_dim_o
        self.gate_dim_l = gate_dim_l        
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen
        self.relu_o = nn.SELU()        
        self.relu_l = nn.SELU()
        self.relu = nn.ReLU()        
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.gate_dim_o, self.hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.FloatTensor(self.gate_dim_l, self.gate_dim_l))
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.FloatTensor(self.gate_dim_o, self.hidden_size)) 
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.gate_dim_l, self.hidden_size))        

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.rand(self.gate_dim_o, self.hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.rand(self.gate_dim_l, self.gate_dim_l))
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.rand(self.gate_dim_o, self.hidden_size)) 
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.gate_dim_l, self.hidden_size))  

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size

        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()
        c_0p1 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)
        h_xp1 = c_0p1
        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])        
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size]) 

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, self.gate_dim_o))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, self.gate_dim_l))

        #torch.set_printoptions(precision=20)
        mo = cmean
        ml = 2.9086
        so = cstd
        sl = 1.8980

        for b in range(0, batch_size):
            for t in range(seq_len):

                h_0, c_0 = h_x

                if b>=time_lag:
                   c_0p1 = h_xp1
                    
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                c_Gate_0 = (c_0.expand(1, 1))
                c_Gate_1 = (c_0p1.expand(1, 1))
                c_0_Gate = (c_0.expand(1, 1))
                u2_Gate = (u2.expand(1, 1))
 
                ib = 0 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_Gate_0-mo)/so, self.weight_b1_yom) + torch.mm((c_Gate_1-mo)/so, self.weight_b2_yom)
                oo = oo1 * torch.sigmoid(oo2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = torch.addmm(bias_b0_ylm, (c_0_Gate-mo)/so, self.weight_b1_ylm) + torch.mm((u2_Gate-ml)/sl, self.weight_b2_ylm)
                ol = ol1 * torch.sigmoid(ol2) 

                if c_0 > 0:                
                    ol_constraint = ol - self.relu(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint) 
                g = u1

            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g
                h_1 = oo * c_1
                l_1 = ol * c_1
                lc_1 = ol_constraint * c_1
                bp_0 = ib * g                
                h_0 = oo * c_0
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0                
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0  
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 

                h_x = (h_1, c_1)

                if b>=time_lag:
                   h_xp1 = c_0

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f