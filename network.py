import os
import numpy as np

import torch
import torchvision 
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal, Categorical

class Encoder(nn.Module):
    def __init__(self, 
                 lr             = 1e-4,            #define learning rate
                 latent_dim     = 128,             #define the latent vector dimension
                 name           = 'encoder',       #define the network name
                 checkpt_dir    = 'logs/models'):
        
        super(Encoder, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     

        # Encoder
        # H_out ​= floor((H_in​− kernel_size[0] + 2*padding[0])/×stride[0]) + 1
        # W_out​ = floor((W_in​− kernel_size[1] + 2*padding[1])/×stride[1]) + 1
        self.encoder = nn.Sequential(
            #128 => 64
            nn.Conv2d( 1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            #64 => 32 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            #32 => 16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            #16 => 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, latent_dim)
        )
        # Decoder
        # H_out ​= (H_in​−1)*stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
        # W_out​ = (W_in​−1)×stride[1] − 2×padding[1] + dilation[1]×(kernel_size[1]−1) + output_padding[1] + 1
        self.decoder = nn.Sequential( 
            nn.Linear(latent_dim, 256*8*8),
            nn.Unflatten(1, (256, 8, 8)),
            #8 => 16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            #16 => 32
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            #32 => 64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            #64 => 128
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        #initialise optimiser
        self.optimiser = optim.Adam(self.parameters(), lr = lr)

        self.to(self.device)

    def forward(self, depth_img):
        #move to correct device
        depth_img = depth_img.to(self.device)

        #compute latent vector
        latent_vector = self.encoder(depth_img)

        #compute reconstructed image
        reconstructed = self.decoder(latent_vector)

        return latent_vector, reconstructed
    
    def get_latent_vectors(self, inputs):
        self.encoder.eval()

        #move to correct device
        inputs = inputs.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)  # Adjust if data_loader returns more than just the images
            latent = self.encoder(inputs)

        return latent

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))

    # def train_autoencoder(autoencoder, data_loader, epochs=10, lr=1e-3):
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     autoencoder.to(device)
    #     optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    #     criterion = nn.MSELoss()

    #     autoencoder.train()
    #     for epoch in range(epochs):
    #         for batch in data_loader:
    #             inputs = batch[0].to(device)  # Adjust if data_loader returns more than just the images
    #             optimizer.zero_grad()
    #             latent, outputs = autoencoder(inputs)
    #             loss = criterion(outputs, inputs)
    #             loss.backward()
    #             optimizer.step()
    #         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



class Critic(nn.Module):

    def __init__(self,                                                         
                 max_action = [0.05, 0.05, 0.05, np.deg2rad(10.)], #action range 
                 input_dims     = [128],                 #input dimension
                 FCL_dims       = [256, 256],             #network dimension
                 N_output       = 1,                      #output dimension
                 N_action       = 4,                      #action dimension
                 N_action_type  = 3,                      #action type dimension
                 lr             = 1e-4,                   #learning rate
                 name           = 'critic',               #define the network name
                 checkpt_dir    = 'logs/models'):
        
        super(Critic, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        #initialise network layers
        self.fc = nn.Sequential(
            nn.Linear(input_dims[0] + N_action_type + N_action, FCL_dims[0]),
            nn.ReLU(),
            nn.Linear(FCL_dims[0], FCL_dims[1]),
            nn.ReLU()
        )

        self.Q_value      = nn.Linear(FCL_dims[1], N_output)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir
        #check if dir exists 
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.optimiser = optim.Adam(self.parameters(), lr = lr)

        self.to(self.device)

    def forward(self, state, action, action_type):

        #move to correct device
        state       = state.to(self.device)
        action      = action.to(self.device)
        action_type = action_type.to(self.device)

        #concat all inputs 
        x = torch.cat([state, action, action_type], dim = 1).float()

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
                 max_action = [0.05, 0.05, 0.05, np.deg2rad(10.)], #action range 
                 input_dims   = [128],                 #input dimension
                 FCL_dims     = [256, 256],             #network dimension
                 N_action       = 4,                    #action dimension
                 N_action_type  = 3,                    #action type dimension
                 lr             = 1e-4,                 #learning rate
                 name           = 'actor',              #define the network name
                 checkpt_dir    = 'logs/models'):     #define checkpoint directory
        
        super(Actor, self).__init__()

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        #initialise action dimention
        self.N_action = N_action
        self.N_action_type = N_action_type

        #initialise network layers
        self.fc = nn.Sequential(
            nn.Linear(input_dims[0], FCL_dims[0]),
            nn.ReLU(),
            nn.Linear(FCL_dims[0], FCL_dims[1]),
            nn.ReLU()
        )

        self.mean        = nn.Linear(FCL_dims[1], N_action)
        self.std         = nn.Linear(FCL_dims[1], N_action)
        self.action_type = nn.Linear(FCL_dims[1], N_action_type)

        #initialise max action range
        self.max_action = torch.tensor(max_action).to(self.device)

        #initialise checkpoint directory
        self.checkpt_dir  = checkpt_dir
        #check if dir exists 
        if not os.path.exists(self.checkpt_dir):
            os.makedirs(self.checkpt_dir)
        self.checkpt_file = os.path.abspath(os.path.join(self.checkpt_dir, name+ '_sac'))

        #initialise optimiser
        self.optimiser = optim.Adam(self.parameters(), lr = lr)

        #used for preventing 0 value
        self.sm_c = 1e-6                 

        self.to(self.device)

    def forward(self, state):

        #move to correct device
        state = state.to(self.device)

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
        action_type_probs = torch.softmax(self.action_type(x), dim=-1)

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

        actions     = torch.tanh(z)*self.max_action
        action_type = Categorical(action_type_probs).sample()

        return actions, action_type, z, normal, action_type_probs

    def compute_log_prob(self, normal, a, z):

        log_probs  = normal.log_prob(z) - torch.log(1-a.pow(2) + self.sm_c)
        log_probs  = log_probs.sum(-1, keepdim = True)

        return log_probs.float()

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpt_file))