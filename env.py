import os
import cv2
import sys
import time
import copy
import utils
import numpy as np
import matplotlib.pyplot as plt


from coppeliasim_zmqremoteapi_client import RemoteAPIClient

#initialise color space
COLOR_SPACE = np.asarray([[78.0, 121.0, 167.0],   # blue
                          [89.0, 161.0, 79.0],    # green
                          [156, 117, 95],         # brown
                          [242, 142, 43],         # orange
                          [237.0, 201.0, 72.0],   # yellow
                          [186, 176, 172],        # gray
                          [255.0, 87.0, 89.0],    # red
                          [176, 122, 161],        # purple
                          [118, 183, 178],        # cyan
                          [255, 157, 167]])/255.0 # pink

#initialise gripper status
GRIPPER_FULL_CLOSE = 0
GRIPPER_FULL_OPEN  = 1
GRIPPER_NON_CLOSE_NON_OPEN = 2

#initialise action type
MOVE  = 0
GRASP = 1
PUSH  = 2

#initialise home pose 
HOME_POSE = [-0.1112, 
             0.48541, 
             0.26883, 
             0, 0, 0]

class Env():
    def __init__(self, 
                 obj_dir, 
                 N_obj, 
                 workspace_lim, 
                 is_test = False, 
                 use_preset_test = False,
                 detect_obj_z_thres       =  0.015,
                 can_execute_action_thres =  0.005,
                 push_reward_thres        =  0.02,
                 ground_collision_thres   =  0.00,
                 move_collision_thres     =  0.02,
                 lift_z_after_grasp       =  0.05,
                 gripper_open_force       =  20.,
                 gripper_close_force      =  100.,
                 gripper_velocity         =  0.50,
                 gripper_joint_open       =  0.03,
                 gripper_joint_close      = -0.045):

        #define workingspace limitation
        self.workspace_lim = workspace_lim

        #compute working space dimension        
        self.x_length = self.workspace_lim[0][1] - self.workspace_lim[0][0]
        self.y_length = self.workspace_lim[1][1] - self.workspace_lim[1][0]
        self.x_center = self.x_length/2. + self.workspace_lim[0][0]
        self.y_center = self.y_length/2. + self.workspace_lim[1][0]

        #define is_test
        self.is_test = is_test
        self.use_preset_test = use_preset_test

        #connect to simulator
        self.sim_client = RemoteAPIClient()
        self.sim = self.sim_client.require('sim')
        self.sim.startSimulation()

        #define the lift height after grasping successfully
        self.lift_z_after_grasp = lift_z_after_grasp

        #set reward threshold
        self.push_reward_thres        = push_reward_thres
        self.detect_obj_z_thres       = detect_obj_z_thres
        self.move_collision_thres     = move_collision_thres
        self.ground_collision_thres   = ground_collision_thres
        self.can_execute_action_thres = can_execute_action_thres

        #define gripper related velocity and force
        self.gripper_open_force  = gripper_open_force
        self.gripper_close_force = gripper_close_force
        self.gripper_velocity    = gripper_velocity
        self.gripper_joint_open  = gripper_joint_open
        self.gripper_joint_close = gripper_joint_close
        self.gripper_status      = None

        #define obj directory
        self.obj_dir = os.path.abspath(obj_dir)
        #define objects in the scene
        self.N_pickable_obj = self.N_obj   = N_obj

        #reset
        self.reset_success = True
        self.reset()

    def reset(self, reset_obj = True):

        if reset_obj:
            self.item_data_dict = dict()
            self.item_data_dict['color']   = []
            self.item_data_dict['obj_ind'] = []
            self.item_data_dict['picked']  = []
            self.item_data_dict['handle']  = []
            self.item_data_dict['p_pose']  = []
            self.item_data_dict['c_pose']  = []
        else:
            self.item_data_dict['c_pose']  = self.update_item_pose()
            self.item_data_dict['p_pose']  = copy.copy(self.item_data_dict['c_pose'])

        #start setting
        self.start_env()

        #setup virtual RGB-D camera
        self.setup_rgbd_cam()
        try:
            print("[SUCCESS] setup rgbd camera")
        except:
            print("[FAIL] setup rgbd camera")
            self.reset_success = False

        self.N_pickable_obj = self.N_obj

        try:
            #get obj paths
            self.obj_paths  = os.listdir(self.obj_dir)
            print("[SUCCESS] load obj paths")

            #assign obj type
            if reset_obj:
                self.item_data_dict['obj_ind'] = np.random.choice(np.arange(len(self.obj_paths)), self.N_obj, replace = True).tolist()
                
                print("[SUCCESS] randomly choose objects")

                self.item_data_dict['color'] = COLOR_SPACE[np.arange(self.N_obj) % COLOR_SPACE.shape[0]].tolist()

                print("[SUCCESS] randomly choose object colors")
            
            #add objects randomly to the simulation
            self.add_objs(reset_obj)
            print("[SUCCESS] add objects to simulation")            
        except:
            print("[FAIL] add objects to simulation")
            self.reset_success = False

    def start_env(self):

        # get UR5 goal handle
        self.UR5_goal_handle = self.sim.getObject('/UR5_goal')

        # get gripper handle
        self.gripper_tip_handle = self.sim.getObject('/UR5_tip')        

        # stop the simulation to ensure successful reset
        self.sim.stopSimulation()
        time.sleep(1)

        while True:

            # self.sim.stopSimulation()
            self.sim.startSimulation()
            time.sleep(1)

            gripper_tip_pos = self.sim.getObjectPosition(self.gripper_tip_handle, 
                                                         self.sim.handle_world)
            
            UR5_goal_pos    = self.sim.getObjectPosition(self.UR5_goal_handle, 
                                                         self.sim.handle_world)
            
            d_gripper_goal = np.linalg.norm(np.array(gripper_tip_pos) - np.array(UR5_goal_pos))

            if abs(gripper_tip_pos[2] - HOME_POSE[2]) <= 1e-2 and d_gripper_goal <= 1e-2:
                print("[SUCCESS] restart environment")
                break
            else:
                #reset UR5 goal to home pose
                self.set_obj_pose(HOME_POSE[0:3], HOME_POSE[3:6], self.UR5_goal_handle, self.sim.handle_world)

    def setup_rgbd_cam(self):

        # get camera handle
        self.cam_handle = self.sim.getObject('/Vision_sensor')

        # get depth sensor related parameter
        self.near_clip_plane = self.sim.getObjectFloatParam(self.cam_handle, 
                                                            self.sim.visionfloatparam_near_clipping)
        
        self.far_clip_plane = self.sim.getObjectFloatParam(self.cam_handle, 
                                                           self.sim.visionfloatparam_far_clipping)
        
        # get camera pose
        cam_pos, cam_ori = self.get_obj_pose(self.cam_handle, self.sim.handle_world)

        # get rotation matrix relative to world frame
        rotmat    = utils.euler2rotm(cam_ori)

        # construct transformation matrix (from camera frame to world frame)
        self.cam_TM          = np.eye(4)
        self.cam_TM[0:3,3]   = np.array(cam_pos)
        self.cam_TM[0:3,0:3] = copy.copy(rotmat)
        self.K               = np.asarray([[618.62,      0, 320], 
                                           [     0, 618.62, 240], 
                                           [     0,      0,   1]])
        
        print(f"[setup_rgbd_cam] \n {self.cam_TM}")

        # get RGB-D data
        self.depth_scale = 1.
        self.bg_color_img, self.bg_depth_img = self.get_rgbd_data()
        self.bg_depth_img = self.bg_depth_img * self.depth_scale

    def get_rgbd_data(self):
        #TODO: may need to flip left right (not sure now)

        #get color_img
        
        raw_color_img_byte, resol = self.sim.getVisionSensorImg(self.cam_handle)
        raw_color_img = self.sim.unpackUInt8Table(raw_color_img_byte)

        color_img = np.array(raw_color_img)
        color_img.shape = (resol[1], resol[0], 3)
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(float)/255.
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = color_img.astype(np.uint8)


        #get depth_img
        
        raw_depth_img_byte, resol = self.sim.getVisionSensorDepth(self.cam_handle)
        raw_depth_img = self.sim.unpackFloatTable(raw_depth_img_byte)
        
        depth_img = np.array(raw_depth_img)
        depth_img.shape = (resol[1], resol[0])
        depth_img = np.fliplr(depth_img)
        depth_img = depth_img*(self.far_clip_plane - self.near_clip_plane) + self.near_clip_plane

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(depth_img)
        # ax[1].imshow(color_img)
        # plt.show()

        # print(color_img.shape, depth_img.shape)

        return color_img, depth_img

    def is_within_working_space(self, pos, margin = 0.1):

        #check if the gipper tip is within working position + margin area
        min_x, max_x = self.workspace_lim[0]
        min_y, max_y = self.workspace_lim[1]
        is_within_x = min_x - margin <= pos[0] <= max_x + margin
        is_within_y = min_y - margin <= pos[1] <= max_y + margin

        return is_within_x, is_within_y

    def check_is_sim_stable(self):

        #get gripper_tip_pos        
        gripper_tip_pos = self.sim.getObjectPosition(self.gripper_tip_handle, 
                                                        self.sim.handle_world)

        is_within_x, is_within_y = self.is_within_working_space(gripper_tip_pos, 
                                                                margin = 0.1)

        if  not (is_within_x and is_within_y):
            print("[WARNING] simulation unstable")
            self.start_env()
            self.add_objs()

    def reset_obj2workingspace(self):

        continue_to_check = True
        while continue_to_check:

            has_reset = False
            for i, obj_handle in enumerate(self.item_data_dict['handle']):
                
                obj_pos = self.sim.getObjectPosition(obj_handle, self.sim.handle_world)

                #check if the obj is within the working space
                is_within_x, is_within_y = self.is_within_working_space(obj_pos, 0)

                if not(is_within_x and is_within_y) and (not self.item_data_dict['picked'][i]):
                    #randomise obj pose
                    obj_pos, obj_ori = self.randomise_obj_pose(xc=self.x_center,
                                                               yc=self.y_center,
                                                               xL=self.x_length,
                                                               yL=self.y_length)

                    #set obj pose
                    self.set_obj_pose(obj_pos, obj_ori, obj_handle, self.sim.handle_world)

                    time.sleep(2)
                    has_reset = True
                    print(f"[SUCCESS] reset obj {i} to working space")

            if not has_reset:
                continue_to_check = False

    def randomise_obj_pose(self, xc, yc, xL, yL):

        #initialise object x, y
        obj_x = xc + np.random.uniform(-1, 1)*xL/2.
        obj_y = yc + np.random.uniform(-1, 1)*yL/2.
        obj_z = 0.15

        #define object droping position
        obj_pos = [obj_x, obj_y, obj_z]

        #define object droping orientation
        obj_ori = [2*np.pi*np.random.uniform(0, 1),
                   2*np.pi*np.random.uniform(0, 1),
                   2*np.pi*np.random.uniform(0, 1)]
        
        return obj_pos, obj_ori

    def add_objs(self, reset_obj = False):

        #initialise object handles
        for i, obj_ind in enumerate(self.item_data_dict['obj_ind']):
            
            if self.is_test and self.use_preset_test:
                #TODO: add test cases
                pass
            else:
                
                #get object file path
                c_obj_file = os.path.join(self.obj_dir, self.obj_paths[obj_ind])
                
                if reset_obj:
                    #randomise obj pose
                    obj_pos, obj_ori = self.randomise_obj_pose(xc=self.x_center,
                                                               yc=self.y_center,
                                                               xL=self.x_length,
                                                               yL=self.y_length)
                else:
                    obj_pos, obj_ori = self.item_data_dict['c_pose'][i][0:3], self.item_data_dict['c_pose'][i][3:6]
            
            #define the shape name
            c_shape_name = f'shape_{i}'

            #define object color
            obj_color = self.item_data_dict['color'][i]

            print(f"object {i}: {c_shape_name}, pose: {obj_pos + obj_ori}")

            fun_out = self.sim.callScriptFunction('importShape@remoteApiCommandServer',
                                                  self.sim.scripttype_childscript,
                                                  [0,0,255,0],
                                                  obj_pos + obj_ori + obj_color,
                                                  [c_obj_file, c_shape_name],
                                                  bytearray())
            
            c_shape_handle = fun_out[0][0]
            # self.obj_handles.append(c_shape_handle)

            #update obj data
            if not reset_obj:
                self.item_data_dict['handle'][i]  = c_shape_handle
            else:
                self.item_data_dict['handle'].append(c_shape_handle)
                self.item_data_dict['picked'].append(False)

            if not (self.is_test and self.use_preset_test):
                time.sleep(2 if reset_obj else 0.5)                                      

        #check anything is outside of working space
        #if yes, reset pose
        self.reset_obj2workingspace()

        #update objs position
        self.item_data_dict['c_pose'] = self.update_item_pose()
        self.item_data_dict['p_pose'] = copy.copy(self.item_data_dict['c_pose'])


    def track_target(self, cur_pos, cur_ori, delta_pos_step, delta_ori_step):

        self.sim.setObjectPosition(self.UR5_goal_handle,
                                   (cur_pos[0] + delta_pos_step[0], 
                                    cur_pos[1] + delta_pos_step[1], 
                                    cur_pos[2] + delta_pos_step[2]),
                                   self.sim.handle_world)

        self.sim.setObjectOrientation(self.UR5_goal_handle, 
                                      (cur_ori[0] + delta_ori_step[0], 
                                       cur_ori[1] + delta_ori_step[1], 
                                       cur_ori[2] + delta_ori_step[2]), 
                                        self.sim.handle_world)

        fun_out = self.sim.callScriptFunction('ik@UR5',
                                              self.sim.scripttype_childscript,
                                              [0], 
                                              [0.], 
                                              ['0'], 
                                              bytearray())

        #get current goal position        
        cur_pos = self.sim.getObjectPosition(self.UR5_goal_handle, 
                                             self.sim.handle_world)

        #get current goal orientation            
        cur_ori = self.sim.getObjectOrientation(self.UR5_goal_handle, 
                                                self.sim.handle_world)

        return cur_pos, cur_ori
            
    def compute_move_step(self, cur, next, N_steps):

        #get move vector
        move_vector = np.asarray([next[0] - cur[0], 
                                  next[1] - cur[1], 
                                  next[2] - cur[2]])
        
        #get move vector magnitude
        move_norm = np.linalg.norm(move_vector)

        #get unit direction
        unit_dir  = move_vector/move_norm

        #get move step
        #TODO: may not work for norm very closed to zero
        delta_step = unit_dir*(move_norm/N_steps) if move_norm != 0. else np.array([0, 0, 0])

        return delta_step

    def move(self, delta_pos, delta_ori):

        #get gripper tip pose
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        #get current goal pose (ensure gripper tip and goal pose are the same)
        self.set_obj_pose(gripper_tip_pos, gripper_tip_ori, self.UR5_goal_handle, self.sim.handle_world)
        
        #get position
        UR5_cur_goal_pos, UR5_cur_goal_ori = self.get_obj_pose(self.UR5_goal_handle, self.sim.handle_world)

        #compute delta move step
        #TODO: set N_step = 50 now, may change later
        N_steps = 50 
        next_pos = np.array(UR5_cur_goal_pos) + np.array(delta_pos)
        next_ori = np.array(UR5_cur_goal_ori) + np.array(delta_ori)

        delta_pos_step = self.compute_move_step(UR5_cur_goal_pos, next_pos, N_steps)
        delta_ori_step = self.compute_move_step(UR5_cur_goal_ori, next_ori, N_steps)

        for step_iter in range(N_steps):
            UR5_cur_goal_pos, UR5_cur_goal_ori = self.track_target(UR5_cur_goal_pos, 
                                                                   UR5_cur_goal_ori,
                                                                   delta_pos_step, 
                                                                   delta_ori_step)
            
        UR5_cur_goal_pos, UR5_cur_goal_ori = self.track_target(next_pos, 
                                                               next_ori,
                                                               [0,0,0],
                                                               [0,0,0])
        
    def open_close_gripper(self, is_open_gripper):

        #get gripper open close joint handle        
        RG2_gripper_handle = self.sim.getObject('/openCloseJoint')

        #get joint position
        gripper_joint_position = self.sim.getJointPosition(RG2_gripper_handle)

        #set gripper force
        gripper_force = self.gripper_open_force if is_open_gripper else self.gripper_close_force
        
        self.sim.setJointTargetForce(RG2_gripper_handle, gripper_force)

        #set gripper velocity
        gripper_vel   = self.gripper_velocity if is_open_gripper else -self.gripper_velocity

        self.sim.setJointTargetVelocity(RG2_gripper_handle, gripper_vel)

        grasp_cnt = 0
        while True:

            new_gripper_joint_position = self.sim.getJointPosition(RG2_gripper_handle)
            
            if is_open_gripper:
                if (new_gripper_joint_position > self.gripper_joint_open):
                    self.gripper_status = GRIPPER_FULL_OPEN
                    return
            else:
                if new_gripper_joint_position <= self.gripper_joint_close:
                    self.gripper_status = GRIPPER_FULL_CLOSE
                    return 
                elif new_gripper_joint_position >= gripper_joint_position:
                    grasp_cnt += 1
                    #TODO: more testing
                    if grasp_cnt >= 20:
                        print("[GRASP] grasp something")
                        self.gripper_status = GRIPPER_NON_CLOSE_NON_OPEN
                        return
                else:
                    grasp_cnt = 0

            gripper_joint_position = new_gripper_joint_position

    def update_item_pose(self):

        poses = []
        for obj_handle in self.item_data_dict['handle']:
            obj_pos, obj_ori = self.get_obj_pose(obj_handle, self.sim.handle_world)
            poses.append(obj_pos + obj_ori)

        return poses

    def get_obj_pose(self, handle, ref_frame):

        pos = self.sim.getObjectPosition(handle, ref_frame)
        ori = self.sim.getObjectOrientation(handle, ref_frame)

        return pos, ori
    
    def set_obj_pose(self, pos, ori, handle, ref_frame):

        self.sim.setObjectPosition(handle, pos,ref_frame)
        self.sim.setObjectOrientation(handle, ori, ref_frame)

    def move_reward(self):

        #TODO: should track object within vision?

        #get gripper tip position
        gripper_tip_pos, _ = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        #get current object position
        self.item_data_dict['c_pose'] = self.update_item_pose()

        d_min = np.inf
        # for i, obj_handle in enumerate(self.obj_handles):
        for i, obj_handle in enumerate(self.item_data_dict['handle']): 
            #compute distance between gripper tip and obj
            d = np.linalg.norm(np.array(self.item_data_dict['c_pose'][i][0:3]) - np.array(gripper_tip_pos))

            if d_min > d:
                d_min = d

        reward = -d_min

        print(f"[move_reward] r: {reward}")

        return reward

    def move_collision_reward(self):
    
        d_obj_move = 0
        reward     = 0

        #get current object position
        self.item_data_dict['c_pose'] = self.update_item_pose()

        for i, obj_handle in enumerate(self.item_data_dict['handle']):

            #compute distance between current object position and previous object position
            d = np.linalg.norm(np.array(self.item_data_dict['c_pose'][i][0:3]) - np.array(self.item_data_dict['p_pose'][i][0:3]))

            if d > self.move_collision_thres:

                if d_obj_move < d:
                    d_obj_move = d

                print("[WARN] collision with object during movement")

        reward = -d_obj_move

        print(f"[move_collision_reward] r: {reward}")

        return reward
    
    def grasp_reward(self):

        #check if the object is grasped firmly
        is_success_grasp = False

        if self.gripper_status == GRIPPER_FULL_CLOSE:
            #fail to grasp anything
            reward = -1.
        elif self.gripper_status == GRIPPER_NON_CLOSE_NON_OPEN:

            #update current goal position
            UR5_cur_goal_pos = self.sim.getObjectPosition(self.UR5_goal_handle, 
                                                          self.sim.handle_world)
            
            #lift up the object for testing if the grasping is successful

            self.move(delta_pos = [0, 0, self.lift_z_after_grasp],
                      delta_ori = [0, 0, 0])

            #update current object position

            self.item_data_dict['c_pose'] = self.update_item_pose()



            for i, obj_handle in enumerate(self.item_data_dict['handle']):
                
                #mark it as grasped 
                change_z = abs(self.item_data_dict['c_pose'][i][2] - self.item_data_dict['p_pose'][i][2])
                print(f"[GRASP] change_z: {change_z}")
                if change_z >= self.lift_z_after_grasp*0.5:
                    is_success_grasp = True
                    self.N_pickable_obj -= 1

                    obj_pos, obj_ori = self.randomise_obj_pose(xc = self.x_center - 1., 
                                                               yc = self.y_center,
                                                               xL = 0.,
                                                               yL = 0.)
                    
                    self.set_obj_pose(obj_pos, 
                                      obj_ori, 
                                      obj_handle, 
                                      self.sim.handle_world)
                    # self.sim.setObjectPosition(obj_handle, obj_pos, self.sim.handle_world)
                    self.item_data_dict['picked'][i] = True
                    break
            
            if is_success_grasp:
                #full success
                reward =  2.0
                print("[GRASP] successful grasp")
            else:
                #partial success 
                reward = -0.5

        print(f"[grasp_reward] r: {reward}")

        return reward, is_success_grasp

    def push_reward(self):

        #update gripper tip position
        UR5_cur_goal_pos = self.sim.getObjectPosition(self.UR5_goal_handle, 
                                                      self.sim.handle_world)

        #update current object position after action
        self.item_data_dict['c_pose'] = self.update_item_pose()

        #check if anything is pushed
        max_d = 0.0
        for i, obj_handle in enumerate(self.item_data_dict['handle']):
            
            #compute distance
            d = np.linalg.norm(np.array(self.item_data_dict['c_pose'][i][0:3]) - np.array(self.item_data_dict['p_pose'][i][0:3]))

            if max_d < d:
                max_d = d

        if max_d >= self.push_reward_thres:
            reward =  1.0
        else:
            reward = -1.0
        
        print(f"[push_reward] r: {reward}")

        return reward

    def executable_reward(self):

        #get UR5 goal 
        UR5_cur_goal_pos, _ = self.get_obj_pose(self.UR5_goal_handle, self.sim.handle_world)    

        #get gripper tip
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)    

        d = np.linalg.norm(np.array(UR5_cur_goal_pos) - np.array(gripper_tip_pos))

        #check if the action is executable
        self.can_execute_action = True if d <= self.can_execute_action_thres else False

        #ensure goal pose and gripper pose are the same
        self.set_obj_pose(gripper_tip_pos, gripper_tip_ori, self.UR5_goal_handle, self.sim.handle_world)

        #set reward
        reward = 0 if self.can_execute_action else -1.0

        print(f"[executable_reward] r: {reward}")

        return reward

    def collision_to_ground_reward(self):
        #get gripper_tip_pos        
        gripper_tip_pos, _ = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        #check if the tipper is in collision with ground
        self.is_collision_to_ground = (gripper_tip_pos[2] <= self.ground_collision_thres)

        #set reward
        reward = -1. if self.is_collision_to_ground else 0
        
        print(f"[collision_to_ground_reward] r: {reward}")

        return reward
    
    def workingspace_reward(self, margin = 0.1):

        #get gripper_tip_pos        
        gripper_tip_pos, _ = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)
        
        if (gripper_tip_pos[0] < self.workspace_lim[0][0] - margin) or \
           (gripper_tip_pos[0] > self.workspace_lim[0][1] + margin):
            reward = -1.
        elif (gripper_tip_pos[1] < self.workspace_lim[1][0] - margin) or \
             (gripper_tip_pos[1] > self.workspace_lim[1][1] + margin):
            reward = -1.
        elif (gripper_tip_pos[2] < self.workspace_lim[2][0]) or \
             (gripper_tip_pos[2] > self.workspace_lim[2][1]):
            reward = -1.
        else:
            reward = 0

        self.is_out_of_working_space = True if reward < 0 else False
        
        print(f"[workingspace_reward] gripper_tip_pos: {gripper_tip_pos}") if self.is_out_of_working_space else None
        print(f"[workingspace_reward] r: {reward}")

        return reward

    def reward(self, action_type):

        #compute how close between gripper tip and the nearest object
        reward = self.move_reward()
        is_success_grasp = False
        #compute reward
        if action_type == MOVE:
            reward += self.move_collision_reward()

        elif action_type == GRASP:
            grasp_reward, is_success_grasp = self.grasp_reward()
            reward += grasp_reward

        elif action_type == PUSH:
            reward += self.push_reward()

        #compute action executable reward
        reward += self.executable_reward()

        #check if the tipper is in collision with ground
        reward += self.collision_to_ground_reward()

        #check if the tipper is out of working space
        reward += self.workingspace_reward()

        #update previous object position
        self.item_data_dict['p_pose'] = self.update_item_pose()

        return reward, is_success_grasp

    def step(self, action_type, delta_pos, delta_ori):
        
        is_success_grasp = False

        delta_ori = [0, 0, delta_ori]
        if action_type == MOVE:
            self.open_close_gripper(is_open_gripper = True)
            self.move(delta_pos = delta_pos,
                      delta_ori = delta_ori)
        elif action_type == GRASP:
            self.open_close_gripper(is_open_gripper = True)
            self.move(delta_pos = delta_pos,
                      delta_ori = delta_ori)
            self.open_close_gripper(is_open_gripper = False)
        elif action_type == PUSH:

            print("[STEP] PUSHHHHHHHHHHHHHHHHH!!!!")
            self.open_close_gripper(is_open_gripper = False)
            self.move(delta_pos = delta_pos,
                      delta_ori = delta_ori)
            # #TODO: it seems necessary
            # #following MOVE acttion will open gripper and affect the scene object 
            # self.move(delta_pos = [0,0,0.05],
            #           delta_ori = [0,0,0])

        #get color image and depth image after action    
        ncolor_img, ndepth_img = self.get_rgbd_data() 

        #compute reward
        reward, is_success_grasp = self.reward(action_type)
        
        #check if the vision has any object

        im_h, im_w = ndepth_img.shape[0], ndepth_img.shape[1] 
        pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), 
                                  np.linspace(0,im_h-1,im_h))

        x = np.multiply(pix_x-self.K[0][2], ndepth_img/self.K[0][0])
        y = np.multiply(pix_y-self.K[1][2], ndepth_img/self.K[1][1])
        z = copy.copy(ndepth_img)

        x.shape = (im_h*im_w,1)
        y.shape = (im_h*im_w,1)
        z.shape = (im_h*im_w,1)
        ones    = np.ones_like(z)
        xyz_pts = np.concatenate((x, y, z, ones), axis=1)

        # get camera pose
        cam_pos, cam_ori = self.get_obj_pose(self.cam_handle, self.sim.handle_world)

        # get rotation matrix relative to world frame
        rotmat    = utils.euler2rotm(cam_ori)

        # construct transformation matrix (from camera frame to world frame)
        self.cam_TM          = np.eye(4)
        self.cam_TM[0:3,3]   = np.array(cam_pos)
        self.cam_TM[0:3,0:3] = copy.copy(rotmat)

        #trasnform to world frame (coninside with robot frame)
        T_xyz_pts = np.matmul(self.cam_TM, xyz_pts.T)
        T_z       = T_xyz_pts[2,:]
        T_z.shape = (im_h, im_w)

        #get object points
        N_obj_pts = np.sum(T_xyz_pts[2,:] > self.detect_obj_z_thres) 

        #anything observed?
        self.has_obj_within_vision = N_obj_pts > 0

        #compute vision reward
        reward = reward if self.has_obj_within_vision else reward - 1.

        # print(f"[STEP] any_obj: {self.has_obj_within_vision}, N_obj_pts: {N_obj_pts}")
        # print(f"[STEP] min_z: {np.min(T_xyz_pts[2,:])}, max_z: {np.max(T_xyz_pts[2,:])}")
        # fig, ax = plt.subplots(1, 2)

        # ax[0].imshow(T_z)
        # ax[1].imshow(ncolor_img)
        # plt.show()

        return ncolor_img, ndepth_img, reward, True if is_success_grasp else False
    
    def return_home(self):

        #get gripper tip pose
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)

        print('[return_home]', gripper_tip_ori, HOME_POSE[3:6])

        #compute delta position and orientation
        delta_pos = np.array(HOME_POSE[0:3]) - np.array(gripper_tip_pos)
        delta_ori = np.array(HOME_POSE[3:6]) - np.array(gripper_tip_ori)

        self.move(delta_pos = delta_pos, delta_ori = delta_ori)

        print("[SUCCESS] return home pose")

    def compute_item_bboxes(self):

        bbox_items          = []
        size_items          = []
        center_items        = []
        face_pts_items      = []
        face_normals_items  = []
        face_centers_items  = []
        face_plane_items    = []
        Ro2w_items          = []

        for i, handle in enumerate(self.item_data_dict['handle']):

            item_ind = self.item_data_dict['obj_ind'][i]

            #get min/max xyz based on body frame
            r, max_x = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_max_x)
            r, min_x = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_min_x)
            r, max_y = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_max_y)
            r, min_y = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_min_y)
            r, max_z = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_max_z)
            r, min_z = self.sim.getObjectFloatParameter(handle, self.sim.objfloatparam_objbbox_min_z)

            #get size
            size_items.append([max_x - min_x, max_y - min_y, max_z - min_z])

            if item_ind != 2:
                R_o2b = np.array([[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 0]])
            else:
                R_o2b = np.eye(3)

            pose = self.sim.getObjectPose(handle, self.sim.handle_world)
            cxyz = pose[0:3]

            center_items.append(cxyz)

            xyzw = pose[3:7]
            wxyz = [xyzw[-1], xyzw[0], xyzw[1], xyzw[2]]

            R_b2w  = utils.quaternion_to_rotation_matrix(wxyz)

            #x => z, y => y z => x
            p_obj = np.array([[max_x, max_y, min_z], #1a,3a,5a,0
                              [max_x, max_y, max_z], #2a,3b,5b,1
                              [max_x, min_y, min_z], #1b,3d,6a,2
                              [max_x, min_y, max_z], #2b,3c,6b,3
                              [min_x, max_y, min_z], #1d,4a,5d,4
                              [min_x, max_y, max_z], #2d,4b,5c,5
                              [min_x, min_y, min_z], #1c,4d,6d,6
                              [min_x, min_y, max_z]])#2c,4c,6c,7

            R = np.matmul(R_b2w, R_o2b)
            Ro2w_items.append(R)

            p = np.zeros((8,3))
            for j in range(p_obj.shape[0]):
                p[j,:] = np.matmul(R, p_obj[j,:]) + np.array(cxyz)

            bbox_items.append(p)

            face_pts_items.append([[p[0,:], p[2,:], p[6,:], p[4,:]],
                                   [p[1,:], p[3,:], p[7,:], p[5,:]],
                                   [p[0,:], p[1,:], p[3,:], p[2,:]],
                                   [p[4,:], p[5,:], p[7,:], p[6,:]],
                                   [p[0,:], p[1,:], p[5,:], p[4,:]],
                                   [p[2,:], p[3,:], p[7,:], p[6,:]]])
            
            normals = []
            centers = []
            planes  = []
            for face in face_pts_items[i]:

                #compute face normals
                v1 = np.array(face[0]) - np.array(face[1]) 
                v2 = np.array(face[2]) - np.array(face[1])
                n  = np.cross(v1, v2)
                n /= np.linalg.norm(n)

                normals.append(n)

                #compute face center
                center = np.zeros((3,))
                for pt in face:
                    center += pt
                center /= 4.
                centers.append(center)

                a,b,c,d = n[0], n[1], n[2], -np.dot(n, center)
                planes.append([a,b,c,d])


            face_normals_items.append(normals)
            face_centers_items.append(centers)
            face_plane_items.append(planes)

        return bbox_items, size_items, center_items, face_pts_items, face_normals_items, face_centers_items, face_plane_items, Ro2w_items

    def get_closest_item(self, item_ind = None):

        #get item poses
        item_poses = self.update_item_pose()

        #get gripper pose
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, 
                                                             self.sim.handle_world)

        #get closest item index
        if item_ind is None:
            min_norm     = np.inf
            min_delta    = None

            for i in range(len(item_poses)):
                delta_vector      = np.array(item_poses[i][0:3]) - np.array(gripper_tip_pos)
                delta_vector_norm = np.linalg.norm(delta_vector)

                if min_norm > delta_vector_norm:
                    min_norm   = delta_vector_norm
                    item_ind = i
                    min_delta = delta_vector
        else:
             min_delta = np.array(item_poses[item_ind][0:3]) - np.array(gripper_tip_pos)

        item_pos = item_poses[item_ind][0:3]

        return item_ind, item_pos, gripper_tip_pos, min_delta
    
    def sort_item_from_nearest_to_farest(self):

        #get item poses
        items_pose = self.update_item_pose()

        #get gripper pose
        gripper_tip_pos, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, 
                                                             self.sim.handle_world)

        #compute distance

        distances  = []
        delta_vecs = []

        for i in range(len(items_pose)):
            delta_vector      = np.array(items_pose[i][0:3]) - np.array(gripper_tip_pos)

            delta_vecs.append(delta_vector)
            distances.append(np.linalg.norm(delta_vector))

        items_pose      = np.array(items_pose)
        distances       = np.array(distances)
        delta_vecs      = np.array(delta_vecs)

        sorted_item_ind   = np.argsort(distances)
        sorted_items_pose = items_pose[sorted_item_ind,:]
        sorted_distances  = distances[sorted_item_ind]
        sorted_delta_vecs = delta_vecs[sorted_item_ind,:]

        return sorted_item_ind, sorted_items_pose[:,0:3], gripper_tip_pos, sorted_delta_vecs

    def get_closest_item_neighbour(self, item_ind):

        #get all items bounding boxes
        bbox_items, size_items, center_items, face_pts_items, face_normals_items, face_centers_items, face_plane_items, Ro2w_items = self.compute_item_bboxes()

        #get all item poses
        item_poses   = self.update_item_pose()

        #get closest neighbour item
        target_pose   = item_poses[item_ind]
        min_d         = np.inf
        neighbour_ind = None

        for i, item_pose in enumerate(item_poses):

            if i == item_ind or self.item_data_dict['picked'][i]:
                continue

            d = np.linalg.norm(np.array(item_pose[0:3]) - np.array(target_pose[0:3]))
            if min_d > d:
                min_d = d
                neighbour_ind = i

        #find which bounding points are the closest
        target_bbox_pts    = np.vstack((bbox_items[item_ind], face_centers_items[item_ind]))
        target_bbox_size   = size_items[item_ind]
        target_R           = Ro2w_items[item_ind]
        target_center      = center_items[item_ind]

        #get neighbour normals and centers
        neighbour_bbox_pts     = np.vstack((bbox_items[neighbour_ind], face_centers_items[neighbour_ind]))
        neighbour_bbox_size    = size_items[neighbour_ind]
        neighbour_normals      = face_normals_items[neighbour_ind]
        neighbour_face_centers = face_centers_items[neighbour_ind]
        neighbour_R            = Ro2w_items[neighbour_ind]
        neighbour_center       = center_items[neighbour_ind]

        min_d    = np.inf
        min_pt_plane_dist = np.inf
        min_pair = None
        neighbour_pt   = None
        target_pt      = None

        #TODO [FINISH]: make a five point detection

        for i, t_bbox_pt in enumerate(target_bbox_pts):

            for j, n_bbox_pt in enumerate(neighbour_bbox_pts):
                    
                    # if np.argmax(n) == 2:
                    #     continue

                    #compute distance between points
                    d = np.linalg.norm(t_bbox_pt - n_bbox_pt)

                    if min_d > d:
                        min_d = d

                        target_pt    = copy.copy(t_bbox_pt)
                        neighbour_pt = copy.copy(n_bbox_pt)
        
                        # pt1 = np.dot( n, pt_plane_dist) + t_bbox_pt
                        # pt2 = np.dot(-n, pt_plane_dist) + t_bbox_pt

                        # v1  = np.dot(n, pt1 - neighbour_face_centers[j])
                        # v2  = np.dot(n, pt2 - neighbour_face_centers[j])

                        # if abs(v1) < abs(v2):
                        #     neighbour_pt = pt1
                        # else:
                        #     neighbour_pt = pt2

        #compute the point on the neighbour plane
        #get the difference vector between bbox pt on the face - bbox pt on the target item

        #get the push point
        push_point = (np.array(neighbour_pt[0:2]) + np.array(target_pt[0:2]))/2.
        push_point_z = (item_poses[neighbour_ind][2] + item_poses[item_ind][2])/2.
        push_point = np.array([push_point[0], 
                               push_point[1], 
                               push_point_z + 0.025])

        #get the point just right before pushing
        ang_list = np.linspace(0, 2*np.pi, 36, endpoint = True)
        be4_push_point = None

        p_list     = []
        p_path_cnt = []

        for i, ang, in enumerate(ang_list): 

            #comput the rotation matrix for various angles
            R = np.array([[np.cos(ang), -np.sin(ang), 0],
                          [np.sin(ang),  np.cos(ang), 0],
                          [          0,            0, 1]])
            
            #compute the just be4 pushing candidate
            p  = np.matmul(R, np.array([0.03, 0, 0])) + push_point

            #deep copy to ensure correct storage
            p_list.append(copy.copy(p))

            #create 2d box at gripper tip for collision checking
            gripper_tip_box = np.array([[-0.0175,       0, -0.025], 
                                        [ 0.0175,       0, -0.025],
                                        [      0,  0.0125, -0.025],
                                        [      0, -0.0125, -0.025],
                                        [      0,       0, -0.025]]).T
            
            #compute the gripper 2d box for collision checking
            gripper_tip_box = np.matmul(R, gripper_tip_box).T + p

            #ensure the starting point of pushing is not in collision with other items
            is_collision_at_start_pt = False

            for j in range(len(bbox_items)):
                is_collision_at_start_pt = self.is_within_bbox(gripper_tip_box, center_items[j], size_items[j], Ro2w_items[j])
                if is_collision_at_start_pt:
                    break

            cnt = 0
            if not is_collision_at_start_pt:
                
                #generate trajectory points for collision checking
                push_delta       = push_point - p
                push_delta_norm  = np.linalg.norm(push_delta)
                push_unit_vector = push_delta/push_delta_norm
                push_N_step      = np.ceil(push_delta_norm/0.005).astype(np.int32) + 1
                
                push_step_mag    = np.linspace(0, push_delta_norm, push_N_step, endpoint = True)
                push_step_mag    = push_step_mag[1:] - push_step_mag[0:-1]

                for step_mag in push_step_mag:
                    
                    #update the gripper box coordinates for collision checking
                    for j in range(gripper_tip_box.shape[0]):
                        gripper_tip_box[j] += push_unit_vector*step_mag

                    if (self.is_within_bbox(gripper_tip_box, target_center,    target_bbox_size,    target_R) or 
                        self.is_within_bbox(gripper_tip_box, neighbour_center, neighbour_bbox_size, neighbour_R)):
                            break
                    
                    cnt += 1
                
                #it means that this pushing cannot touch any object => useless push
                #reset the cnt to 0
                if cnt == len(push_step_mag) - 1:
                    cnt = 0
            
            #store how many trajectory points are not in collision with items 
            p_path_cnt.append(cnt)

        #get the path with max points
        path_ind = np.random.choice((p_path_cnt == np.max(p_path_cnt)).nonzero()[0])
        be4_push_point = p_list[path_ind]
        print(f"be4_push_point: {be4_push_point}")
        print(f"max path cnt: {np.max(p_path_cnt)}")

        return item_poses[neighbour_ind][0:3], neighbour_ind, min_d, push_point, be4_push_point

    def is_within_bbox(self, points, center, size, Ro2w):
        #TODO: [item side] use oriented bounding to check collision

        Rw2o = np.linalg.inv(Ro2w)
        ps_  = copy.copy(points)
        for p in ps_:
            p -= center
            p  = np.matmul(Rw2o, p.T)
            if (0 <= abs(p[0]) <= size[0]/2. and 
                0 <= abs(p[1]) <= size[1]/2. and
                0 <= abs(p[2]) <= size[2]/2.):
                return True
        
        return False


    def compute_guidance_grasp_ang(self, item_ind):

        #get all items bounding boxes
        bbox_items, size_items, center_items, face_pts_items, face_normals_items, face_centers_items, face_plane_items, Ro2w_items = self.compute_item_bboxes()

        #extract bounding box corresponding to item index
        p            = bbox_items[item_ind]

        #initialise vectors, norms and unit vectors
        vecs      = np.zeros((3,3))
        vecs_norm = np.zeros(3)
        unit_vecs = np.zeros((3,3))

        vecs[0,:] = p[0,:] - p[1,:]
        vecs[1,:] = p[0,:] - p[2,:]
        vecs[2,:] = p[0,:] - p[4,:]

        for i in range(vecs.shape[0]):
            vecs_norm[i] = np.linalg.norm(vecs[i,:])
            unit_vecs[i,:] = vecs[i,:]/vecs_norm[i]

            print(f'unit_vecs{i}: {unit_vecs[i,:]}, norm: {vecs_norm[i]}')

        #compute yaw angle orientation
        axis  = None
        max_norm = -np.inf 
        for i in range(unit_vecs.shape[0]):

            z_component = abs(unit_vecs[i,2])
            if z_component < 0.85 and max_norm < vecs_norm[i]:
                max_norm = vecs_norm[i]
                axis = unit_vecs[i,:]

        ang1 = np.arctan2(axis[1], axis[0]) - np.deg2rad(90.)
        if ang1 > np.pi:
            ang1 -= 2*np.pi
        elif ang1 < -np.pi:
            ang1 += 2*np.pi

        ang2 = ang1 + np.pi if np.sign(ang1) <= 0 else ang1 - np.pi

        print(f'[guidance_generation] axis: {axis}')

        _, gripper_tip_ori = self.get_obj_pose(self.gripper_tip_handle, self.sim.handle_world)
        delta_ori1 = ang1 - gripper_tip_ori[2]
        delta_ori2 = ang2 - gripper_tip_ori[2]

        if abs(delta_ori1) <= abs(delta_ori2):
            yaw_angle = ang1
            delta_yaw = delta_ori1
        else:
            yaw_angle = ang2
            delta_yaw = delta_ori2

        return yaw_angle, gripper_tip_ori[2], delta_yaw