import torch
import torch.nn as nn
from torch import Tensor

class MCPBRNN_Generic_PETconstraint_Scaling(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Scaling, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
 
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
        Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = cmean
        ml = 2.9086
        so = cstd
        sl = 1.8980        
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
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                lc_1 = ol_constraint  * c_1                
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 
                lc_0 = ol_constraint  * c_0 
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
                Gate_olc[b,:] = ol_constraint              
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd 

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_Scaling(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_Scaling, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = cmean
        ml = 2.9086
        so = cstd
        sl = 1.8980        
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

                f = (1.0 -  oo - ol) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd 

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_NOscaling(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_NOscaling, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = 0
        ml = 0
        so = 1
        sl = 1        
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

                f = (1.0 -  oo - ol) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_constantoutput_variableLoss(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_constantoutput_variableLoss, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.relu_l = nn.ELU()
        self.relu = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))      
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs):
 
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
        Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = 0
        ml = 2.9086
        so = 1
        sl = 1.8980        
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
                oo2 = 0 
                oo = oo1 

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

                ol_constraint = self.relu(ol_constraint)
                f = self.relu(f)

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
                Gate_olc[b,:] = ol_constraint                
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_constantoutput_variableLoss_BYPASSM0(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):

        super(MCPBRNN_Generic_PETconstraint_constantoutput_variableLoss_BYPASSM0, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))     
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))  
        self.theltaC = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))  

        self.relu_l = nn.ELU()
        self.relu = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.theltaC = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))  

    def forward(self, x, epoch, time_lag, y_obs):

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
        Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = 0
        ml = 2.9086
        so = 1
        sl = 1.8980        
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  
        scale_factor = 1
        
        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

                px = self.relu(u1 + c_0 - torch.exp(self.theltaC) * scale_factor)

                if u1 == 0:
                    ib = self.relu(u1)
                else:
                    ib = (px / u1) 

                ib = 0 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = 0 
                oo = oo1 #* torch.sigmoid(oo2)

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

                ol_constraint = self.relu(ol_constraint)
                f = self.relu(f)

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
                Gate_olc[b,:] = ol_constraint                
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_constant_Out_variableLoss(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_constant_Out_variableLoss, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))     
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = 0
        ml = 2.9086
        so = 1
        sl = 1.8980        
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
                oo2 = 0 
                oo = oo1

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                f = (1.0 -  oo - ol) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0  
                l_n[b,:] = l_0
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd 

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_constant_Out_variableLoss_BYPASSM0(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_constant_Out_variableLoss_BYPASSM0, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.theltaC = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))  
        self.relu = nn.ReLU()
        self.relu_l = nn.ReLU()        
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))      
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.theltaC = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))  

    def forward(self, x, epoch, time_lag, y_obs):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = 0#236.6076
        ml = 2.9086
        so = 1#63.5897
        sl = 1.8980        
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  
        scale_factor = 1
        
        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

                px = self.relu(u1 + c_0 - torch.exp(self.theltaC) * scale_factor)

                if u1 == 0:
                    ib = self.relu(u1)
                else:
                    ib = (px / u1) 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = 0 
                oo = oo1

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                f = (1.0 -  oo - ol) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_variable_Out_constantLoss(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_variable_Out_constantLoss, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))            
        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std):
 
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
        Gate_olc = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))

        #torch.set_printoptions(precision=20)
        mo = c_mean
        ml = 2.9086
        so = c_std
        sl = 1.8980        
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
                ol3 = 0 
                ol4 = 0 
                ol = ol1 

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1
                lc_1 = ol_constraint * c_1 
                lc_0 = ol_constraint * c_0                                 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 

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
                Gate_olc[b,:] = ol_constraint               
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_variable_Out_constantLoss(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_variable_Out_constantLoss, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))        

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
 
    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))

        #torch.set_printoptions(precision=20)
        mo = c_mean
        ml = 2.9086
        so = c_std
        sl = 1.8980        
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
                ol3 = 0 
                ol4 = 0 
                ol = ol1 

                f = (1.0 -  oo - ol) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_f, h_nout, obs_std

class MCPBRNN_PETconstraint_constant_OutLoss(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_PETconstraint_constant_OutLoss, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))          
        self.relu_l = nn.ELU()
        self.relu = nn.ReLU()        
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))          
 
    def forward(self, x, epoch, time_lag, y_obs):

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
        Gate_olc = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        #torch.set_printoptions(precision=20)
        mo = 0#236.6076
        ml = 0#2.9086
        so = 1#63.5897
        sl = 1#1.8980        
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
                oo2 = 0
                oo = oo1 

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 
                ol3 = 0
                ol4 = 0 
                ol = ol1 

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)
                f = self.relu(f)
                ol_constraint = self.relu(ol_constraint)
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
                Gate_olc[b,:] = ol_constraint               
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std

class MCPBRNN_constant_OutLoss(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_constant_OutLoss, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))          

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))          
 
    def forward(self, x, epoch, time_lag, y_obs):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        #torch.set_printoptions(precision=20)
        mo = 0
        ml = 0
        so = 1
        sl = 1        
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
                oo2 = 0
                oo = oo1 

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 
                ol3 = 0
                ol4 = 0 
                ol = ol1 

                f = (1.0 -  oo - ol) 

                g = u1

            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM0_SC(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM0_SC, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))    
        self.theltaC = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))               
        self.relu_l = nn.ReLU()
        self.relu = nn.ReLU()        
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.theltaC = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
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
        Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = cmean
        ml = 2.9086
        so = cstd
        sl = 1.8980        
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  
        scale_factor = 1
        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     
                px = self.relu(u1 + c_0 - torch.exp(self.theltaC) * scale_factor)
 
                if u1 == 0:
                    ib = self.relu(u1)
                else:
                    ib = (px / u1) 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                lc_1 = ol_constraint  * c_1                
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 
                lc_0 = ol_constraint  * c_0 
            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_olc[b,:] = ol_constraint              
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM1_SC(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Scaling_BYPASSM1_SC, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.weight_b1_yum = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yum = nn.Parameter(torch.FloatTensor(self.hidden_size))  

        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))   
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.weight_b1_yum = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yum = nn.Parameter(torch.rand(self.hidden_size))  

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
 
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
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
        Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yum = (self.bias_b0_yum.unsqueeze(0).expand(1, *self.bias_b0_yum.size()))
        #torch.set_printoptions(precision=20)
        mo = cmean
        ml = 2.9086
        so = cstd
        sl = 1.8980        
        u1_max = 221.5190
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

                ib1 = torch.addmm(bias_b0_yum, (c_0-mo)/so, self.weight_b1_yum)
                ib2 = torch.mm(u1/u1_max, self.weight_b1_yum)                
                ib = torch.sigmoid(ib1+ib2)

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

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                lc_1 = ol_constraint  * c_1                
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 
                lc_0 = ol_constraint  * c_0 
            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_olc[b,:] = ol_constraint              
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, Gate_ib, Gate_oo, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std