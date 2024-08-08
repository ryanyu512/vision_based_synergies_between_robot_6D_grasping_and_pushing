import copy
import torch
import utils
import numpy as np
import matplotlib.pyplot as plt

from env import Env
from PIL import Image
from buffer import BufferReplay
from torchvision import transforms
from network import Encoder, Actor, Critic

MOVE  = 0
GRASP = 1
PUSH  = 2

class Agent():

    def __init__(self, 
                 env,
                 Na      = 4,
                 Na_type = 3,
                 N_batch = 32, 
                 alpha   = 0.2,
                 tau     = 0.05,
                 gamma   = 0.99,
                 save_step_interval = 10,
                 is_debug = False):

        #initialise inference device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        #initialise env
        self.env     = env
        print("[SUCCESS] initialise environment")
        #initialise action and action_type dimension
        self.Na      = Na
        self.Na_type = Na_type
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
        #soft update to make critic target align with critic
        self.soft_update()

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
        # self.transform = transforms.Compose([transforms.ToTensor()])  
        #initialise history 
        self.r_hist    = []
        self.step_hist = []

        #initialise if debug
        self.is_debug = is_debug

        #initialise save interval
        self.save_step_interval = save_step_interval

        #for guidance demonstration
        self.enter_pushing = False
        self.push_point    = None
        self.be4_push      = None

    def preprocess_input(self, color_img, depth_img):
        
        #copy image
        in_depth_img = copy.copy(depth_img)
        in_color_img = copy.copy(color_img)

        #check nan
        in_depth_img[np.isnan(in_depth_img)] = 0
        #check negative value
        in_depth_img[in_depth_img < 0] = 0

        #scale depth image into range 0 - 1
        in_depth_img = (in_depth_img.astype(float) - self.env.near_clip_plane)/(self.env.far_clip_plane - self.env.near_clip_plane)
        in_depth_img = np.expand_dims(in_depth_img, axis = 2)
        #scale color image into range 0 - 1
        in_color_img = in_color_img.astype(float)/255.

        #convert to pil image
        # in_depth_img = Image.fromarray(in_depth_img.astype(np.uint8))
        # in_color_img = Image.fromarray(in_color_img.astype(np.uint8))

        #transform input
        # in_depth_img = self.transform(in_depth_img)
        # in_color_img = self.transform(in_color_img)
        # in_depth_img = torch.FloatTensor(in_depth_img)
        # in_color_img = torch.FloatTensor(in_color_img)

        in_color_img = torch.from_numpy(in_color_img.astype(np.float32)).permute(2,0,1)
        in_depth_img = torch.from_numpy(in_depth_img.astype(np.float32)).permute(2,0,1)

        return in_color_img, in_depth_img
    
    def soft_update(self, tau = None):
        
        if tau is None:
            tau = 1.
        else:
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

        if self.buffer_replay.memory_cntr < self.N_batch:
            return

        batch, s, a, a_type, r, ns, done = self.buffer_replay.sample_buffer(self.N_batch)
        
        r.shape    = (r.shape[0], 1)
        done.shape = (done.shape[0], 1)

        s      = torch.FloatTensor(s).to(self.device)
        a      = torch.FloatTensor(a).to(self.device)
        a_type_onehot = torch.FloatTensor(a_type).to(self.device)
        ns     = torch.FloatTensor(ns).to(self.device)
        r      = torch.FloatTensor(r).to(self.device)
        done   = torch.FloatTensor(done).to(self.device)

        #update critic
        with torch.no_grad():
            #compute next action, next action type, next action normals, next action type probability
            na, na_type, nz, n_normal, na_type_probs = self.actor.get_actions(ns)
            
            #compute one hot vector
            na_type_onehot = torch.nn.functional.one_hot(na_type.long(), 
                                                         num_classes = self.Na_type).float()
            #compute log probability
            nlog_probs = self.actor.compute_log_prob(normal = n_normal, a = na, z = nz)

            #compute next q value
            nq1 = self.critic1_target(ns, na, na_type_onehot)
            nq2 = self.critic2_target(ns, na, na_type_onehot)
            nq  = (torch.min(nq1, nq2) - self.alpha*nlog_probs)
            q_target = (r + (1-done)*self.gamma*nq)

        q1 = self.critic1(s, a, a_type_onehot)
        q2 = self.critic2(s, a, a_type_onehot)

        # print(f'[online] q_target.shape: {q_target.shape}')
        c1_loss = (0.5*torch.nn.MSELoss()(q1, q_target))
        c2_loss = (0.5*torch.nn.MSELoss()(q2, q_target))
        c_loss  = (c1_loss + c2_loss)

        self.critic1.optimiser.zero_grad()
        self.critic2.optimiser.zero_grad()
        c_loss.backward()
        self.critic1.optimiser.step()
        self.critic2.optimiser.step()

        #update actor
        a, a_type, z, normal, atype_probs = self.actor.get_actions(s)
        log_probs = self.actor.compute_log_prob(normal = normal, a = a, z = z)
        min_q = torch.min(
            self.critic1(s, a, torch.nn.functional.one_hot(a_type.long(), 
                                                           self.Na_type).float()),
            self.critic2(s, a, torch.nn.functional.one_hot(a_type.long(), 
                                                           self.Na_type).float())
        )
        a_loss = (self.alpha*log_probs - min_q).mean()

        self.actor.zero_grad()
        a_loss.backward()
        self.actor.optimiser.step()

        # Soft update target critic networks
        self.soft_update(self.tau)

        #TODO update the buffer priority
        with torch.no_grad():
            #compute next action, next action type, next action normals, next action type probability
            a, a_type, z, normal, a_type_probs = self.actor.get_actions(ns)
            
            #compute one hot vector
            a_type_onehot = torch.nn.functional.one_hot(a_type.long(), 
                                                         num_classes = self.Na_type).float()

            #compute next q value
            q1 = self.critic1_target(ns, na, na_type_onehot)
            q2 = self.critic2_target(ns, na, na_type_onehot)
            q  = (q1 + q2)*0.5

            q = q.to(torch.device('cpu')).detach().numpy()
            q.shape = (q.shape[0],);
            self.buffer_replay.update_buffer(batch, q)

    def grasp_guidance_generation(self, max_move = 0.05, max_ori = np.deg2rad(30), min_distance_threshold = 0.025):

        #initialise move delta
        delta_move = []

        #step0: start from home position
        self.env.return_home()

        #step1: check if all items are picked
        if self.env.N_pickable_obj == 0:
            return delta_move

        #step 2: get cloest item relative to gripper tip
        sorted_item_ind, sorted_items_pos, gripper_tip_pos, sorted_delta_vecs = self.env.sort_item_from_nearest_to_farest()

        #step 3: [MOVE] move in a straight line to the top of the object and adjust the yaw orientation
        for i in range(len(sorted_item_ind)):
            
            if self.env.item_data_dict['picked'][sorted_item_ind[i]]:
                continue
            
            if self.env.N_pickable_obj >= 2:
                neighbour_pos, neighbour_ind, min_d, push_point, be4_push_point = self.env.get_closest_item_neighbour(sorted_item_ind[i])
            else:
                min_d = np.inf

            #check if item is graspable
            if min_d > min_distance_threshold:

                item_ind        = sorted_item_ind[i]
                target_item_pos = sorted_items_pos[i]
                delta           = sorted_delta_vecs[i]

                #offset the delta to ensure the gripper tip is not in collision with the ground
                delta += np.array([0,0,0.03])

                #get delta yaw
                item_yaw, gripper_yaw, delta_ori = self.env.compute_guidance_grasp_ang(item_ind)

                delta_norm  = np.linalg.norm(delta)
                unit_vector = delta/delta_norm
                N_step      = np.ceil(delta_norm/max_move).astype(np.int32) + 1

                step_mag    = np.linspace(0, delta_norm, N_step, endpoint = True)
                step_mag    = step_mag[1:] - step_mag[0:-1]

                N_step_ori   = np.ceil(abs(delta_ori)/max_ori).astype(np.int32) + 1
                step_mag_ori = np.linspace(0, delta_ori, N_step_ori, endpoint = True)
                step_mag_ori = step_mag_ori[1:] - step_mag_ori[0:-1]

                for i in range(max(len(step_mag), len(step_mag_ori))):
                    delta_move.append([unit_vector[0]*step_mag[i] if i < len(step_mag) else 0, 
                                       unit_vector[1]*step_mag[i] if i < len(step_mag) else 0,
                                       unit_vector[2]*step_mag[i] if i < len(step_mag) else 0,
                                       step_mag_ori[i] if i < len(step_mag_ori) else 0,
                                       MOVE])

                #step 4: [GRASP] open gripper and move vertically down by a constant height
                delta = np.array(target_item_pos) - np.array(gripper_tip_pos)
                delta_move.append([0, 0, -0.03, 0, GRASP])

                return delta_move
        
        return delta_move


    def push_guidance_generation(self, max_move = 0.05, max_ori = np.deg2rad(30)):

        #step 0: start from home position
        self.env.return_home()

        #initialise delta move
        delta_move = []

        #step 1: check if all items are picked
        #check if # of pickable objects 
        if self.env.N_pickable_obj <= 1:
            return delta_move

        #step 2: get cloest item relative to gripper tip
        # item_ind, target_item_pos, gripper_tip_pos, delta = self.env.get_closest_item()
        sorted_item_ind, sorted_items_pos, gripper_tip_pos, sorted_delta_vecs = self.env.sort_item_from_nearest_to_farest()

        #step 3: [PUSH] if min. neighbor distance < threshold => push, otherwise skip
        #a:  Get the 
        #a:  Move in a straight line to the position closed to the chosen object
        #b:  Push towards the center of chosen object by a constant distance
        #c:  Move back to the home position

        #get cloest item relative to target item

        for i in range(len(sorted_item_ind)):
            
            if self.env.item_data_dict['picked'][sorted_item_ind[i]]:
                continue

            neighbour_pos, neighbour_ind, min_d, push_point, be4_push_point = self.env.get_closest_item_neighbour(sorted_item_ind[i])

            if min_d <= 0.025:
                item_ind = sorted_item_ind[i]

                print(f'[guidance_generation] item_ind: {item_ind}') 
                print(f'[guidance_generation] neighbor_ind: {neighbour_ind}') 
                print(f'[guidance_generation] neighbour_pos: {neighbour_pos}') 
                print(f'[guidance_generation] min_d: {min_d}') 
                print(f'[guidance_generation] push_point: {push_point}') 

                #move from the home position to just be4 pushing
                push_delta       = be4_push_point + np.array([0, 0, 0.05]) - np.array(gripper_tip_pos)
                push_delta_norm  = np.linalg.norm(push_delta)

                #move from just be4 pushing to pushing
                push_delta2       = push_point - be4_push_point
                push_delta_norm2  = np.linalg.norm(push_delta2)
                push_unit_vector2 = push_delta2/push_delta_norm2

                ang1 = np.arctan2(push_unit_vector2[1], push_unit_vector2[0]) - np.deg2rad(90.)
                if ang1 > np.pi:
                    ang1 -= 2*np.pi
                elif ang1 < -np.pi:
                    ang1 += 2*np.pi

                # ang1 = np.arctan2(push_unit_vector2[1], push_unit_vector2[0])

                print(f'[guidance_generation] axis: {push_unit_vector2}')

                #compute delta move from home position to just be4 pushing
                _, gripper_tip_ori = self.env.get_obj_pose(self.env.gripper_tip_handle, self.env.sim.handle_world)
                delta_ori = ang1 - gripper_tip_ori[2]

                N_step_ori   = np.ceil(abs(delta_ori)/max_ori).astype(np.int32) + 1
                step_mag_ori = np.linspace(0, delta_ori, N_step_ori, endpoint = True)
                step_mag_ori = step_mag_ori[1:] - step_mag_ori[0:-1]

                push_unit_vector = push_delta/push_delta_norm
                push_N_step      = np.ceil(push_delta_norm/max_move).astype(np.int32) + 1
                
                push_step_mag    = np.linspace(0, push_delta_norm, push_N_step, endpoint = True)
                push_step_mag    = push_step_mag[1:] - push_step_mag[0:-1]

                for i in range(np.max([len(push_step_mag), len(step_mag_ori)])):
                    delta_move.append([push_unit_vector[0]*push_step_mag[i] if i < len(push_step_mag) else 0, 
                                    push_unit_vector[1]*push_step_mag[i] if i < len(push_step_mag) else 0,
                                    push_unit_vector[2]*push_step_mag[i] if i < len(push_step_mag) else 0,
                                    step_mag_ori[i] if i < len(step_mag_ori) else 0,
                                    PUSH])
                    
                delta_move.append([0,0,-0.050,0, PUSH])

                #compute delta move from just be4 pushing to pushing
                push_delta_norm2  = np.linalg.norm(push_delta2)
                push_N_step2      = np.ceil(push_delta_norm2/max_move).astype(np.int32) + 1
                
                push_step_mag2    = np.linspace(0, push_delta_norm2, push_N_step2, endpoint = True)
                push_step_mag2    = push_step_mag2[1:] - push_step_mag2[0:-1]

                for i in range(len(push_step_mag2)):
                    delta_move.append([push_unit_vector2[0]*push_step_mag2[i], 
                                       push_unit_vector2[1]*push_step_mag2[i],
                                       push_unit_vector2[2]*push_step_mag2[i],
                                       0,
                                       PUSH])

                delta_move.append([push_unit_vector2[0]*0.05, 
                                   push_unit_vector2[1]*0.05,
                                   push_unit_vector2[2]*0.05,
                                   0,
                                   PUSH])

                return delta_move
        
        return delta_move


    def interact(self, 
                 is_train = True, 
                 max_episode = 1,
                 is_debug = True):
        
        #start trainiing/evaluation loop
        for episode in range(max_episode) if is_train else 1:

            #initialise episode data
            step = 0
            ep_r = 0.

            while True:

                #get raw data
                color_img, depth_img = self.env.get_rgbd_data()

                #preprocess raw data
                in_color_img, in_depth_img = self.preprocess_input(color_img = color_img, 
                                                                   depth_img = depth_img)

                #add the extra dimension in the 1st dimension
                in_color_img = in_color_img.unsqueeze(0)
                in_depth_img = in_depth_img.unsqueeze(0)

                #get state
                s = self.encoder.get_latent_vectors(inputs = in_depth_img)

                #action selection
                a, a_type, z, normal, a_type_probs = self.actor.get_actions(s)

                #compute one hot vector
                a_type_onehot = torch.nn.functional.one_hot(a_type.long(), 
                                                            num_classes = self.Na_type).float()

                #step
                #TODO:test the step function
                next_color_img, next_depth_img, r, is_success_grasp = self.env.step(a_type.to(torch.device('cpu')).detach().numpy()[0], 
                                                                                    a.to(torch.device('cpu')).detach().numpy()[0][0:3], 
                                                                                    a.to(torch.device('cpu')).detach().numpy()[0][3])

                print(f"[STEP]: {step} [ACTION TYPE]: {a_type} [REWARD]: {r}") if is_debug else None
                print(f"[MOVE]: {a.to(torch.device('cpu')).detach().numpy()[0]}") if is_debug else None

                #preprocess raw data
                next_in_color_img, next_in_depth_img = self.preprocess_input(color_img = next_color_img, 
                                                                             depth_img = next_depth_img)

                next_in_color_img = next_in_color_img.unsqueeze(0)
                next_in_depth_img = next_in_depth_img.unsqueeze(0)

                #convert next color img and next depth img into next state
                ns = self.encoder.get_latent_vectors(inputs = next_depth_img)

                #check if terminate this episode
                #TODO: test the function
                done = False if self.env.N_pickable_obj > 0 else True

                #compute predict q value and labeled q value
                with torch.no_grad(): 
                    
                    #critic the current state + action + action type
                    q1 = self.critic1(state = s, action = a, action_type = a_type_onehot)
                    q2 = self.critic2(state = s, action = a, action_type = a_type_onehot)
                    q  = torch.min(q1, q2)

                    #compute next action based on next action type
                    na, na_type, nz, n_normal, na_type_probs = self.actor.get_actions(ns)

                    #compute one hot vector
                    na_type_onehot = torch.nn.functional.one_hot(na_type.long(), 
                                                                 num_classes = self.Na_type).float()

                    #critic the next state + action + action type
                    nq1 = self.critic1_target(state = ns, action = na, action_type = na_type_onehot)
                    nq2 = self.critic2_target(state = ns, action = na, action_type = na_type_onehot)
                    nq  = torch.min(nq1, nq2)

                #store experience 
                self.buffer_replay.store_transition(s.to(torch.device('cpu')).detach().numpy(), 
                                                    a.to(torch.device('cpu')).detach().numpy(), 
                                                    a_type_onehot.to(torch.device('cpu')).detach().numpy(), 
                                                    r, 
                                                    ns.to(torch.device('cpu')).detach().numpy(), 
                                                    done, 
                                                    q.to(torch.device('cpu')).detach().numpy(), 
                                                    nq.to(torch.device('cpu')).detach().numpy())    

                #update parameter
                self.online()

                #update history
                ep_r += r
                step += 1
                
                #check if done
                if done:
                    self.env.reset(reset_obj = True)
                    print("[SUCCESS] finish one episode")
                    self.r_hist.append(ep_r)
                    self.step_hist.append(step)         
                    break 
                else:

                    #return home position if grasp successfully
                    if is_success_grasp:
                        print("[SUCCESS] grasp an object")
                        self.env.return_home()

                    #check if out of working space
                    elif self.env.is_out_of_working_space:
                        print("[WARN] out of working space")
                        self.env.reset(reset_obj = False)

                    #check if action executable
                    elif not self.env.can_execute_action:
                        print("[WARN] action is not executable")
                        self.env.reset(reset_obj = False)

                    #check if collision to ground
                    elif self.env.is_collision_to_ground:
                        print("[WARN] collision to ground")
                        self.env.reset(reset_obj = False)
                
    def interact_by_guidance(self,
                             is_train = True, 
                             max_episode = 1,
                             is_debug = True, 
                             grasp_guidance = True):
        
        #start trainiing/evaluation loop
        for episode in range(max_episode) if is_train else 1:

            #initialise episode data
            step = 0
            ep_r = 0.
            done = False
            self.enter_pushing = False
            while not done:
                
                print(f"==== episode: {episode} step: {step} ====")

                #action selection
                delta_moves = self.grasp_guidance_generation()
                
                if len(delta_moves) == 0:
                    delta_moves = self.push_guidance_generation()

                if len(delta_moves) == 0:
                    done     = True

                for i in range(len(delta_moves)):

                    #get raw data
                    color_img, depth_img = self.env.get_rgbd_data()

                    #preprocess raw data
                    in_color_img, in_depth_img = self.preprocess_input(color_img = color_img, 
                                                                       depth_img = depth_img)

                    #add the extra dimension in the 1st dimension
                    in_color_img = in_color_img.unsqueeze(0)
                    in_depth_img = in_depth_img.unsqueeze(0)

                    #get state
                    s = self.encoder.get_latent_vectors(inputs = in_depth_img)

                    #action selection
                    move   = np.array(delta_moves[i])
                    n_move = np.array(delta_moves[i+1] if i+1 < len(delta_moves) else [0,0,0,0,move[-1]])

                    #select action
                    a, a_type    =   np.array(move[0:self.Na]),   np.array(move[self.Na])
                    na, na_type  = np.array(n_move[0:self.Na]), np.array(n_move[self.Na])

                    a_type  = torch.FloatTensor(a_type).unsqueeze(0).to(self.device)
                    a       = torch.FloatTensor(a).unsqueeze(0).to(self.device)
                    na_type = torch.FloatTensor(na_type).unsqueeze(0).to(self.device)
                    na      = torch.FloatTensor(na).unsqueeze(0).to(self.device)

                    #compute one hot vector
                    a_type_onehot = torch.nn.functional.one_hot(a_type.long(), 
                                                                num_classes = self.Na_type).float()
                    
                    #step
                    #TODO:test the step function
                    next_color_img, next_depth_img, r, is_success_grasp = self.env.step(a_type.to(torch.device('cpu')).detach().numpy()[0], 
                                                                                        a.to(torch.device('cpu')).detach().numpy()[0][0:3], 
                                                                                        a.to(torch.device('cpu')).detach().numpy()[0][3])

                    print(f"[STEP]: {step} [ACTION TYPE]: {a_type} [REWARD]: {r}") if is_debug else None
                    print(f"[MOVE]: {a.to(torch.device('cpu')).detach().numpy()[0]}") if is_debug else None

                    #preprocess raw data
                    next_in_color_img, next_in_depth_img = self.preprocess_input(color_img = next_color_img, 
                                                                                depth_img = next_depth_img)

                    next_in_color_img = next_in_color_img.unsqueeze(0)
                    next_in_depth_img = next_in_depth_img.unsqueeze(0)

                    #convert next color img and next depth img into next state
                    ns = self.encoder.get_latent_vectors(inputs = next_in_depth_img)

                    #check if terminate this episode
                    #TODO: test the function
                    done = False if self.env.N_pickable_obj > 0 else True

                    #compute predict q value and labeled q value
                    with torch.no_grad(): 
                        
                        #critic the current state + action + action type
                        q1 = self.critic1(state = s, action = a, action_type = a_type_onehot)
                        q2 = self.critic2(state = s, action = a, action_type = a_type_onehot)
                        q  = torch.min(q1, q2)

                        #compute next action based on next action type
                        na, na_type, nz, n_normal, na_type_probs = self.actor.get_actions(ns)

                        #compute one hot vector
                        na_type_onehot = torch.nn.functional.one_hot(na_type.long(), 
                                                                    num_classes = self.Na_type).float()

                        #critic the next state + action + action type
                        nq1 = self.critic1_target(state = ns, action = na, action_type = na_type_onehot)
                        nq2 = self.critic2_target(state = ns, action = na, action_type = na_type_onehot)
                        nq  = torch.min(nq1, nq2)

                    #store experience 
                    self.buffer_replay.store_transition(s.to(torch.device('cpu')).detach().numpy(), 
                                                        a.to(torch.device('cpu')).detach().numpy(), 
                                                        a_type_onehot.to(torch.device('cpu')).detach().numpy(), 
                                                        r, 
                                                        ns.to(torch.device('cpu')).detach().numpy(), 
                                                        done, 
                                                        q.to(torch.device('cpu')).detach().numpy(), 
                                                        nq.to(torch.device('cpu')).detach().numpy())    

                    #update parameter [TODO]
                    

                    #update history
                    ep_r += r
                    step += 1
                    
                    #save model   
                    if step % self.save_step_interval == 0:
                        self.save_models()

                    #check if done
                    if done:
                        self.env.reset(reset_obj = True)
                        print("[SUCCESS] finish one episode")
                        self.r_hist.append(ep_r)
                        self.step_hist.append(step)         
                        break 
                    else:

                        #return home position if grasp successfully
                        if is_success_grasp:
                            print("[SUCCESS] grasp an object")
                            self.env.return_home()
                            print("[SUCCESS] return home position")

                        #check if out of working space
                        elif self.env.is_out_of_working_space:
                            print("[WARN] out of working space")
                            self.env.reset(reset_obj = False)

                        #check if action executable
                        elif not self.env.can_execute_action:
                            print("[WARN] action is not executable")
                            self.env.reset(reset_obj = False)

                        #check if collision to ground
                        elif self.env.is_collision_to_ground:
                            print("[WARN] collision to ground")
                            self.env.reset(reset_obj = False)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.critic1_target.save_checkpoint()
        self.critic2_target.save_checkpoint()

        print('[SUCCESS] save_models')
         


