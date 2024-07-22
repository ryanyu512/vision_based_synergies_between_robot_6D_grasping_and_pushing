import os
import numpy as np

import torch
import torchvision 
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal, Categorical

class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

        #initialise depth and color encoder
        self.depth_encoder = torchvision.models.densenet.densenet121(pretrained=True)
        self.color_encoder = torchvision.models.densenet.densenet121(pretrained=True)

        self.to(self.device)

    def forward(self, color_img, depth_img):

        depth_feat = self.depth_encoder.features(depth_img)
        color_feat = self.color_encoder.features(color_img)

        feat = torch.cat((depth_feat, color_feat), dim = 1)

        return feat

class Critic(nn.Module):

    def __init__(self,                                                         
                 max_action = [0.1, 0.1, 0.1, np.deg2rad(10.), np.deg2rad(10.), np.deg2rad(10.)], #action range 
                 input_dims     = [2048],                 #input dimension
                 FCL_dims       = [256, 256],             #network dimension
                 N_output       = 1,                      #output dimension
                 N_action       = 6,                      #action dimension
                 N_action_type  = 3,                      #action type dimension
                 lr             = 1e-3,                   #learning rate
                 name           = 'critic',               #define the network name
                 checkpt_dir    = 'logs/models'):
        
        super(Critic, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        #initialise network dimensions
        self.input_dims = input_dims
        self.FCL_dims   = FCL_dims
        
        #initialise action dimension
        self.N_action      = N_action
        self.N_action_type = N_action_type

        #initialise output dimention
        self.N_out = N_output

        #initialise network layers
        self.fc = nn.Sequential(
            nn.Linear(self.input_dims[0] + N_action_type + N_action, self.FCL_dims[0]),
            nn.ReLU(),
            nn.Linear(self.FCL_dims[0], self.FCL_dims[1]),
            nn.ReLU()
        )

        self.Q_value      = nn.Linear(self.FCL_dims[1], self.N_out)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.lr = lr
        self.optimiser = optim.Adam(self.parameters(), lr = self.lr)

        self.to(self.device)

    def foward(self, state, action, action_type):

        #concat all inputs 
        x = torch.cat([state, action, action_type], dim = 1)

        x = self.fc(x)

        #compute normal distribution mean
        x = self.Q_value(x)
        
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))

class Actor(nn.Module):
    
    def __init__(self,                                                         
                 max_action = [0.1, 0.1, 0.1, np.deg2rad(10.), np.deg2rad(10.), np.deg2rad(10.)], #action range 
                 input_dims   = [2048],                 #input dimension
                 FCL_dims     = [256, 256],             #network dimension
                 N_action       = 6,                    #action dimension
                 N_action_type  = 3,                    #action type dimension
                 lr             = 1e-3,                 #learning rate
                 name           = 'actor',              #define the network name
                 checkpt_dir    = 'logs/models'):     #define checkpoint directory
        
        super(Actor, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        #initialise network dimensions
        self.input_dims = input_dims
        self.FCL_dims   = FCL_dims
        
        #initialise action dimention
        self.N_action = N_action
        self.N_action_type = N_action_type

        #initialise network layers
        self.fc = nn.Sequential(
            nn.Linear(*self.input_dims, self.FCL_dims[0]),
            nn.ReLU(),
            nn.Linear(self.FCL_dims[0], self.FCL_dims[1]),
            nn.ReLU()
        )

        self.mean        = nn.Linear(self.FCL_dims[1], self.N_action)
        self.std         = nn.Linear(self.FCL_dims[1], self.N_action)
        self.action_type = nn.Linear(self.FCL_dims[1], self.N_action_type)

        #initialise max action range
        self.max_action = max_action

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.lr = lr
        self.optimiser = optim.Adam(self.parameters(), lr = self.lr)

        #used for preventing 0 value
        self.sm_c = 1e-6                 

        self.to(self.device)

    def forward(self, state):

        x = self.fc(state)

        #compute normal distribution mean
        mean = self.mean(x)

        #compute normal distribution std
        std = self.std(x)

        #clamp the std within range
        std = torch.clamp(std, 
                          min = self.sm_c, 
                          max = 1.)
        
        #compute action type
        action_type_probs = torch.softmax(self.action_type(x))

        return mean, std, action_type_probs

    def get_actions(self, 
                    state, 
                    is_reparametrerise = True):

        mean, std, action_type_probs = self.forward(state)

        normal = Normal(mean, std)

        if is_reparametrerise:
            z = normal.rsample()
        else:
            z = normal.sample()

        actions     = torch.tanh(z)*torch.tensor(self.max_action).to(self.dvice)
        action_type = Categorical(action_type_probs).sample()

        return actions, action_type, normal, action_type_probs

    def compute_log_prob(self, normal, a):

        log_probs  = normal.log_prob(a) - torch.log(1-a.pow(2) + self.sm_c)
        log_probs  = log_probs.sum(-1, keepdim = True)

        return log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))