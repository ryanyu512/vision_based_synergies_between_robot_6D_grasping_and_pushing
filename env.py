import os
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
HOME_POSE = [-0.1112, 0.19963, 0.5263, -np.deg2rad(165), 0, 0]

class Env():
    def __init__(self, 
                 obj_dir, 
                 N_obj, 
                 workspace_lim, 
                 is_test = False, 
                 use_preset_test = False,
                 gripper_open_force  =  20.,
                 gripper_close_force =  20.,
                 gripper_velocity    =  0.20,
                 gripper_joint_open  =  0.03,
                 gripper_joint_close = -0.045):

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

        #start setting
        self.start_env()

        #define gripper related velocity and force
        self.gripper_open_force  = gripper_open_force
        self.gripper_close_force = gripper_close_force
        self.gripper_velocity    = gripper_velocity
        self.gripper_joint_open  = gripper_joint_open
        self.gripper_joint_close = gripper_joint_close
        self.gripper_status      = None

        #setup virtual RGB-D camera
        self.setup_rgbd_cam()
        try:
            print("[SUCCESS] setup rgbd camera")
        except:
            print("[FAIL] setup rgbd camera")
            exit()

        #add objects to simulation
        self.obj_dir = os.path.abspath(obj_dir)
        self.N_obj   = N_obj
        # try:
        #get obj paths
        self.obj_paths  = os.listdir(self.obj_dir)
        print("[SUCCESS] load obj paths")

        #assign obj type
        self.obj_inds   = np.random.choice(np.arange(len(self.obj_paths)), 
                                            self.N_obj, replace = True)
        print("[SUCCESS] randomly choose objects")

        #assign color
        self.obj_colors = COLOR_SPACE[np.arange(self.N_obj) % COLOR_SPACE.shape[0]]
        print("[SUCCESS] randomly choose object colors")
        
        #add objects randomly to the simulation
        self.add_objs()
        print("[SUCCESS] add objects to simulation")            
        # except:
        #     print("[FAIL] add objects to simulation")
        #     exit()

    def start_env(self):

        # get UR5 goal handle
        self.UR5_goal_handle = self.sim.getObject('/UR5_goal')
        
        # #reset UR5 goal to home pose
        self.sim.setObjectPosition(self.UR5_goal_handle, 
                                   HOME_POSE[0:3],
                                   self.sim.handle_world)
        
        self.sim.setObjectOrientation(self.UR5_goal_handle, 
                                      HOME_POSE[3:6],
                                      self.sim.handle_world)

        # # get gripper handle
        self.gripper_tip_handle = self.sim.getObject('/UR5_tip')        

        # stop the simulation to ensure successful reset
        self.sim.stopSimulation()
        time.sleep(1)

        while True:

            self.sim.stopSimulation()
            self.sim.startSimulation()
            time.sleep(1)

            gripper_tip_pos = self.sim.getObjectPosition(self.gripper_tip_handle, 
                                                         self.sim.handle_world)

            if abs(gripper_tip_pos[2] - 0.5263) <= 1e-2:
                print("[SUCCESS] restart environment")
                break

    def is_within_working_space(self, pos, margin = 0.):

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

    def setup_rgbd_cam(self):

        # get camera handle
        self.cam_handle = self.sim.getObject('/Vision_sensor')

        # get depth sensor related parameter
        self.near_clip_plane = self.sim.getObjectFloatParam(self.cam_handle, 
                                                            self.sim.visionfloatparam_near_clipping)
        
        self.far_clip_plane = self.sim.getObjectFloatParam(self.cam_handle, 
                                                           self.sim.visionfloatparam_far_clipping)
        
        # get camera position
        cam_pos = self.sim.getObjectPosition(self.cam_handle, 
                                             self.sim.handle_world)

        # get camera orientation in euler        
        cam_euler = self.sim.getObjectOrientation(self.cam_handle, 
                                                  self.sim.handle_world)

        #get rotation matrix relative to world frame
        cam_euler = [cam_euler[0], cam_euler[1], cam_euler[2]]
        rotmat    = utils.euler2rotm(cam_euler)

        #construct transformation matrix (from camera frame to world frame)
        cam_TM          = np.eye(4)
        cam_TM[0:3,3]   = np.array(cam_pos)
        cam_TM[0:3,0:3] = copy.copy(rotmat)

        #get RGB-D data
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
        color_img = color_img.astype(float)/255.
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = color_img.astype(np.uint8)

        #get depth_img
        
        raw_depth_img_byte, resol = self.sim.getVisionSensorDepth(self.cam_handle)
        raw_depth_img = self.sim.unpackFloatTable(raw_depth_img_byte)
        
        depth_img = np.array(raw_depth_img)
        depth_img.shape = (resol[1], resol[0])
        depth_img = depth_img*(self.far_clip_plane - self.near_clip_plane) + self.near_clip_plane

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(depth_img)
        # ax[1].imshow(color_img)
        # plt.show()

        # print(color_img.shape, depth_img.shape)

        return color_img, depth_img

    def reset_obj2workingspace(self):

        continue_to_check = True
        while continue_to_check:

            has_reset = False
            for i, obj_handle in enumerate(self.obj_handles):
                
                obj_pos = self.sim.getObjectPosition(obj_handle, 
                                                     self.sim.handle_world)

                #check if the obj is within the working space
                is_within_x, is_within_y = self.is_within_working_space(obj_pos)

                if not(is_within_x and is_within_y):
                    #randomise obj pose
                    obj_pos, obj_ori = self.randomise_obj_pose()

                    #set obj pose

                    self.sim.setObjectPosition(obj_handle, 
                                               obj_pos,
                                               self.sim.handle_world)

                    self.sim.setObjectOrientation(obj_handle, 
                                                  obj_ori,
                                                  self.sim.handle_world)

                    time.sleep(2)
                    has_reset = True
                    print(f"[SUCCESS] reset obj {i} to working space")

            if not has_reset:
                continue_to_check = False

    def randomise_obj_pose(self):

        #initialise object x, y
        obj_x = self.x_center + np.random.uniform(-1, 1)*self.x_length/2.
        obj_y = self.y_center + np.random.uniform(-1, 1)*self.y_length/2.
        obj_z = 0.15

        #define object droping position
        obj_pos = [obj_x, obj_y, obj_z]

        #define object droping orientation
        obj_ori = [2*np.pi*np.random.uniform(0, 1),
                   2*np.pi*np.random.uniform(0, 1),
                   2*np.pi*np.random.uniform(0, 1)]
        
        return obj_pos, obj_ori

    def add_objs(self):

        #initialise object handles
        self.obj_handles = []
        sim_obj_handles  = []

        for i, obj_ind in enumerate(self.obj_inds):
            
            if self.is_test and self.use_preset_test:
                #TODO: add test cases
                pass
            else:
                
                #get object file path
                c_obj_file = os.path.join(self.obj_dir, self.obj_paths[obj_ind])
                
                #randomise obj pose
                obj_pos, obj_ori = self.randomise_obj_pose()
            
            #define the shape name
            c_shape_name = f'shape_{i}'

            #define object color
            obj_color = self.obj_colors[i].tolist()

            print(f"object {i}: {c_shape_name}, pose: {obj_pos + obj_ori}")

            fun_out = self.sim.callScriptFunction('importShape@remoteApiCommandServer',
                                                  self.sim.scripttype_childscript,
                                                  [0,0,255,0],
                                                  obj_pos + obj_ori + obj_color,
                                                  [c_obj_file, c_shape_name],
                                                  bytearray())
            
            c_shape_handle = fun_out[0][0]
            self.obj_handles.append(c_shape_handle)
            if not (self.is_test and self.use_preset_test):
                time.sleep(2)                                      

        #check anything is outside of working space
        #if yes, reset pose
        self.reset_obj2workingspace()

        self.p_objs_pos = []
        self.c_objs_pos = []

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

    def move(self, next_pos, next_ori):

        # #get position

        UR5_cur_goal_pos = self.sim.getObjectPosition(self.UR5_goal_handle, 
                                                      self.sim.handle_world)
            
        UR5_cur_goal_ori = self.sim.getObjectOrientation(self.UR5_goal_handle, 
                                                         self.sim.handle_world)

        #compute delta move step
        #TODO: set N_step = 50 now, may change later
        N_steps = 50 
        delta_pos_step = self.compute_move_step(UR5_cur_goal_pos, next_pos, N_steps)
        delta_ori_step = self.compute_move_step(UR5_cur_goal_ori, next_ori, N_steps)
        
        # print(UR5_cur_goal_pos, next_pos, delta_pos_step)
        # print(UR5_cur_goal_ori, next_ori, delta_ori_step)

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

        print(f'force: {gripper_force}, velocity: {gripper_vel}, pos: {gripper_joint_position}')

        self.sim.setJointTargetVelocity(RG2_gripper_handle, gripper_vel)

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
                    self.gripper_status = GRIPPER_NON_CLOSE_NON_OPEN
                    return

            gripper_joint_position = new_gripper_joint_position

    def step(self, action_type, next_pos, next_ori):
        
        if action_type == MOVE:
            self.open_close_gripper(is_open_gripper = True)
            self.move(next_pos = next_pos,
                      next_ori = next_ori)
        elif action_type == GRASP:
            self.open_close_gripper(is_open_gripper = True)
            self.move(next_pos = next_pos,
                      next_ori = next_ori)
            self.open_close_gripper(is_open_gripper = False)
        elif action_type == PUSH:
            self.open_close_gripper(is_open_gripper = False)
            self.move(next_pos = next_pos,
                      next_ori = next_ori)
            
    def return_home_pose(self):
        
        #close gripper to prevent collision
        self.open_close_gripper(is_open_gripper = False)

        #return home pose 
        self.move(next_pos = HOME_POSE[0:3], next_ori = HOME_POSE[3:6])

        #open gripper to prepare for the next action
        self.open_close_gripper(is_open_gripper = True)

