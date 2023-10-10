import torch
import torch.nn as nn
from torch import Tensor

class MCPBRNN_PETconstraint_MassRelax_Regular(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 gate_dim_ucorr: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_PETconstraint_MassRelax_Regular, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim
        self.gate_dim_ucorr = gate_dim_ucorr
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen
        self.relu_l = nn.ReLU()
        self.relu_v = nn.ReLU()
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.rand(self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size

        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])        
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])
        mr_n = torch.zeros([batch_size, hidden_size])
        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ov = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yrm = (self.bias_b0_yrm.unsqueeze(0).expand(1, *self.bias_b0_yrm.size()))

        #torch.set_printoptions(precision=20)
        mo = p_mean
        so = p_std
        ml = 2.9086
        sl = 1.8980
        scale_mr = 500
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])

        for b in range(0+time_lag, batch_size):
            for t in range(seq_len):

                h_0, c_0 = h_x
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)

                ib = 0 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)

                ov0 = torch.mm(c_0/scale_mr - torch.exp(bias_b0_yrm), torch.exp(self.weight_s_yvm))
                ov1 = torch.sigmoid(self.weight_r_yvm) * torch.tanh(ov0)

                ov = ov1 - self.relu_v(ov1 - f)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g - ov * torch.abs(c_0 - torch.exp(bias_b0_yrm) * scale_mr)
                h_1 = oo * c_1
                l_1 = ol * c_1
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
                mr_n[b,:] = ov * torch.abs(c_0 - torch.exp(bias_b0_yrm) * scale_mr)
            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_ov[b,:] = ov
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd 

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f, h_nout, obs_std, Gate_ov, mr_n

class MCPBRNN_PETconstraint_MassRelax_Indepedent(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 gate_dim_ucorr: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_PETconstraint_MassRelax_Indepedent, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
#        self.hidden_sizeM = hidden_sizeM        
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim
        self.gate_dim_ucorr = gate_dim_ucorr
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen
        self.relu_l = nn.ReLU()
        self.relu_v = nn.ReLU()
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        #self.weight_s_yvm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.rand(self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size

        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

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
        Gate_ov = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yrm = (self.bias_b0_yrm.unsqueeze(0).expand(1, *self.bias_b0_yrm.size()))

        #torch.set_printoptions(precision=20)
        mo = p_mean
        so = p_std
        ml = 2.9086
        sl = 1.8980
        scale_mr = 500
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])

        for b in range(0+time_lag, batch_size):
            for t in range(seq_len):

                h_0, c_0 = h_x
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)

                ib = 0  

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0  
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0  
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)

                ov0 = torch.sign(c_0/scale_mr - torch.exp(bias_b0_yrm))  
                ov1 = torch.sigmoid(self.weight_r_yvm) * ov0

                ov = ov1 - self.relu_v(ov1 - f)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g - ov * torch.abs(c_0 - torch.exp(bias_b0_yrm) * scale_mr)
                h_1 = oo * c_1
                l_1 = ol * c_1
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0  
                l_n[b,:] = l_0
                bp_n[b,:] = bp_0
                lc_n[b,:] = lc_0       

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_ov[b,:] = ov
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd 

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f, h_nout, obs_std, Gate_ov

class MCPBRNN_PETconstraint_MassRelax_Regular_MI_outputloss_sigmoid(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 gate_dim_ucorr: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_PETconstraint_MassRelax_Regular_MI_outputloss_sigmoid, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim
        self.gate_dim_ucorr = gate_dim_ucorr
        self.spinLen = spinLen
        self.traintimeLen = traintimeLen
        self.relu_l = nn.ReLU()
        self.relu_v = nn.ReLU()
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))        
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.weight_r_yvm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))
        self.weight_b1_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.rand(self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):
 
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
        mr_n = torch.zeros([batch_size, hidden_size])

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ov = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yrm = (self.bias_b0_yrm.unsqueeze(0).expand(1, *self.bias_b0_yrm.size()))

        #torch.set_printoptions(precision=20)
        mo = p_mean
        so = p_std
        ml = 2.9086
        sl = 1.8980
        scale_mr = 500
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])

        for b in range(0+time_lag, batch_size):
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
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)

                ov0 = torch.mm(c_0/scale_mr - torch.exp(bias_b0_yrm), torch.exp(self.weight_s_yvm))
                ov1 = torch.sigmoid(self.weight_r_yvm) * torch.tanh(ov0)

                ov = ov1 - self.relu_v(ov1 - f)

                g = u1

            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g - ov * torch.abs(c_0 - torch.exp(bias_b0_yrm) * scale_mr)
                h_1 = oo * c_1
                l_1 = ol * c_1
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
                mr_n[b,:] = ov * torch.abs(c_0 - torch.exp(bias_b0_yrm) * scale_mr)
                
            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_ov[b,:] = ov
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f, h_nout, obs_std, Gate_ov, mr_n
