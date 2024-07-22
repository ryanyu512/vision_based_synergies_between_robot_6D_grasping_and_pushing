import copy
import torch
import numpy as np

from env import Env
from PIL import Image
from buffer import BufferReplay
from torchvision import transforms
from network import Encoder, Actor, Critic



class Agent():

    def __init__(self, 
                 env,
                 N_batch = 32, 
                 alpha = 0.2,
                 tau   = 0.05,
                 gamma = 0.99):

        #initialise env
        self.env     = env
        print("[SUCCESS] initialise environment")
        #initialise encoder
        self.encoder = Encoder()
        #initialise actor
        self.actor   = Actor()
        #initialise critic network 1
        self.critic1 = Critic()
        #initialise ciritc network 2
        self.critic2 = Critic()
        #initialise critic network target 1
        self.critic1_target = Critic()
        #initialise critic network target 2
        self.critic2_target = Critic()
        print("[SUCCESS] initialise networks")
        #initialise buffer replay
        self.buffer_replay = BufferReplay()
        print("[SUCCESS] initialise memory buffer")
        #initialise batch size
        self.N_batch = N_batch
        #initalise small constant to prevent zero value
        self.sm_c    = 1e-6
        #initialise temperature factor
        self.alpha   = alpha
        #initialise discount factor
        self.gamma   = gamma
        #initialise soft update factor
        self.tau     = tau
        #initialise input image transform
        self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                  std =[0.229, 0.224, 0.225])])  

    def preprocess_input(self, color_img, depth_img):
        
        #copy image
        in_depth_img = copy.copy(depth_img)
        in_color_img = copy.copy(color_img)

        #check nan
        in_depth_img[np.isnan(in_depth_img)] = 0
        #check negative value
        in_depth_img[in_depth_img < 0] = 0

        #stack up depth image
        in_depth_img.shape = (in_depth_img.shape[0], in_depth_img.shape[1], 1)
        in_depth_img = np.concatenate((in_depth_img, in_depth_img, in_depth_img), axis=2)

        #scale depth image into range 0 - 1
        in_depth_img = (in_depth_img - self.env.near_clip_plane)/(self.env.far_clip_plane - self.env.near_clip_plane)

        #convert to pil image
        in_depth_img = Image.fromarray(in_depth_img.astype(np.uint8))
        in_color_img = Image.fromarray(in_color_img.astype(np.uint8))

        #transform input
        in_depth_img = self.transform(in_depth_img)
        in_color_img = self.transform(in_color_img)

        return in_color_img, in_depth_img
    
    def soft_update(self, tau = None):
        
        if tau is None:
            tau = self.tau

        #get parameters of target network
        target_c1_params = self.critic1_target.named_parameters()
        target_c2_params = self.critic2_target.named_parameters()

        #get parameters of current network
        c1_params = self.critic1.named_parameters()
        c2_params = self.critic2.named_parameters()

        #transform to dict sturcture
        target_c1_params_dict = dict(target_c1_params)
        target_c2_params_dict = dict(target_c2_params)
        c1_params_dict = dict(c1_params)
        c2_params_dict = dict(c2_params)

        for name in target_c1_params_dict:
            target_c1_params_dict[name] = tau*c1_params_dict[name].clone() + \
                                          (1-tau)*target_c1_params_dict[name].clone()

        for name in target_c2_params_dict:
            target_c2_params_dict[name] = tau*c2_params_dict[name].clone() + \
                                          (1-tau)*target_c2_params_dict[name].clone()

        self.critic1_target.load_state_dict(target_c1_params_dict)
        self.critic2_target.load_state_dict(target_c2_params_dict)

    def online(self):

        batch, s, a, a_type, r, ns, done = self.buffer_replay.sample_buffer(self.N_batch)
        
        s = torch.FloatTensor(s).cuda()
        a = torch.FloatTensor(a).cuda()
        r = torch.FloatTensor(r).cuda()
        a_type = torch.FloatTensor(ns).cuda()
        done = torch.FloatTensor(done).cuda()

        #update critic
        with torch.no_grad():
            #compute next action, next action type, next action normals, next action type probability
            na, na_type, n_normal, na_type_probs = self.actor.get_actions(ns)
            #compute one hot vector
            na_type_onehot = torch.nn.functional.one_hot(na_type, num_classes = na_type.shape[1]).float()
            #compute log probability
            nlog_probs = self.actor.compute_log_prob(normal = n_normal, a = na)
            #compute next q value
            nq1 = self.critic1_target(ns, na, na_type_onehot)
            nq2 = self.critic2_target(ns, na, na_type_onehot)
            nq  = torch.min(nq1, nq2) - self.alpha*nlog_probs
            q_target = r + (1-done)*self.gamma*nq

        #compute one hot vector
        a_type_onehot = torch.nn.functional.one_hot(a_type, num_classes = a_type.shape[1]).float()
        q1 = self.critic1(s, a, a_type_onehot)
        q2 = self.critic1(s, a, a_type_onehot)
        c1_loss = 0.5*torch.nn.MSELoss()(q1, q_target)
        c2_loss = 0.5*torch.nn.MSELoss()(q2, q_target)
        c_loss  = c1_loss + c2_loss

        self.critic1.optimiser.zero_grad()
        self.critic2.optimiser.zero_grad()
        c_loss.backward()
        self.critic1.optimiser.step()
        self.critic2.optimiser.step()

        #update actor
        a, a_type, normal, atype_probs = self.actor.get_actions(s)
        log_probs = self.actor.compute_log_prob(normal = normal, a = a)
        min_q = torch.min(
            self.critic1(s, a, torch.nn.functional.one_hot(a_type, atype_probs.shape[1]).float()),
            self.critic2(s, a, torch.nn.functional.one_hot(a_type, atype_probs.shape[1]).float())
        )
        a_loss = self.alpha*log_probs - min_q

        self.actor.zero_grad()
        a_loss.backward()
        self.actor.optimiser.step()

        # Soft update target critic networks
        self.soft_update()

    def interact(self, 
                 is_train = True, 
                 max_episode = 1000):
        
        #start trainiing/evaluation loop
        for episode in range(max_episode) if is_train else 1:

            while True:

                #get state

                #action selection

                #store experience 

                #update parameter

                #check if all the 

            


