import gym
import cv2
import time
import logging
from gym import error, spaces
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
from pyrep.pyrep import PyRep
from ECM_dVRL_v01.ArmPSM_model import ArmPSM
from ECM_dVRL_v01.ArmECM_model import ArmECM

class ECMEnv(gym.GoalEnv):
    def __init__(self, psm_num, n_actions, n_states, n_goals, n_substeps, camera_enabled, scene_path):
        
        self.viewer = None
        
        self.pr = PyRep()
        self.pr.launch(scene_path, headless = False)
        
        self.psm_num = psm_num
        self.psm = ArmPSM(self.pr, self.psm_num)
        self.ecm = ArmECM(self.pr)
        
        self.n_substeps = n_substeps
        self.pr.set_simulation_timestep(1./self.n_substeps)
        
        self.sim_timestep = 0.1
        self.success_radius = 1e-3
        self.camera_enabled = camera_enabled
        if self.camera_enabled:
            self.metadata = {'render.modes': ['matplotlib', 'rgb', 'human']}
            self.camera = camera(self.pr, rgb = True)
        else:
            self.metadata = {'render.modes': ['human']}

        self.FOV = np.radians(self.ecm.left_cam.get_perspective_angle())
        base_img = self.ecm.getStereoImagePairs()[0]
        self.w, self.h = base_img.shape[0], base_img.shape[1]
        self.f = self.h/(2.0 * np.tan(self.FOV/2.0))

        self.seed()
        self._env_setup()
        self.done = False
        self.marker_pos = self.psm.get_marker_position(self.ecm.base_handle)
        #self.desired_goal = np.array([self.marker_pos[0], self.marker_pos[1], 1.0000 + self.marker_pos[2]])
        #self.desired_goal = np.array([0., 0., 0., 1.])
        self.desired_goal = np.array([0., 0., 0.7, 1.0])

        #self.bounds = [[-1.000, 0.], [-0.030, 0.045], [0, 0.075], [-0.030, 0.045]]
        self.bounds = np.array([np.radians([-75, 45]), np.radians([-45, 65]), np.array([0, 0.235]), np.radians([-90, 90])])
        self.init_angles = np.array([-0.9000, 0., 0., 0.])
                        
        self.action_space = spaces.Box(-0.02, 0.02, shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
			desired_goal=spaces.Box(-np.inf, np.inf, shape=(n_goals,), dtype='float32'),
			achieved_goal=spaces.Box(-np.inf, np.inf, shape=(n_goals,), dtype='float32'),
			observation=spaces.Box(-np.inf, np.inf, shape=(n_states,), dtype='float32'),
			))
        self.pr.start()
    
    def __del__(self):
        self.close()
        
    def dt(self):
        return self.sim_timestep * self.n_substeps
    
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        valid = True
        logging.info('Logging action update in env.step')
        if not self.done:
            #action = np.clip(action, self.action_space.low, self.action_space.high)
            valid = self._set_action(action)

            logging.info('Logging runtime for _simulator_step()')
            start = time.time()
            if valid:
                self._simulator_step()
                #self._step_callback()
            logging.info(str(time.time() - start))
        obs = self._get_obs() 
        done = False
        #logging.info('Logging success check in env.step')
        start = time.time()
        success = self._is_success()
        #logging.info(str(time.time() - start))
        if success or not valid:
            done = True
        self.done = done
        
        #logging.info('Logging reward acquisition in env.step')
        start = time.time()
        reward = self._interaction_reward(obs['achieved_goal'], obs['desired_goal'])
        #logging.info(str(time.time() - start))
        info = {'success' : success,
                'reward': reward}
        
        return obs, reward, done, info
    
    def reset(self):
        self._reset_sim()
        obs = self._get_obs()
        return obs

    def get_centroid(self):
        left_image, right_image = self.ecm.getStereoImagePairs()
        left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2HSV)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2HSV)
        
        h, s, v = cv2.split(left_image)
        ret, thresh_img = cv2.threshold(h, 200, 255, cv2.THRESH_BINARY) #Threshold set to 200 to detect blue marker
        M = cv2.moments(thresh_img)
        if not M["m00"]:
            c_x = M["m10"]
            c_y = M["m01"]
        else:
            c_x = int(M["m10"]/M["m00"])
            c_y = int(M["m01"]/M["m00"])
        return np.array([c_x, c_y])

    def get_transform_matrices_product(self):

        T1b, T21, T32, T43, Tc4 = self.ecm.getTransformMatrices()
        T1b = np.vstack((np.reshape(T1b, (3, 4)), np.array([0., 0., 0., 1.])))
        T21 = np.vstack((np.reshape(T21, (3, 4)), np.array([0., 0., 0., 1.])))
        T32 = np.vstack((np.reshape(T32, (3, 4)), np.array([0., 0., 0., 1.])))
        T43 = np.vstack((np.reshape(T43, (3, 4)), np.array([0., 0., 0., 1.])))
        Tc4 = np.vstack((np.reshape(Tc4, (3, 4)), np.array([0., 0., 0., 1.])))
        #print (T1b, T21, T32, T43, Tc4)
        T_prod = np.matmul(Tc4, np.linalg.inv(np.matmul(np.matmul(T1b, T21), np.matmul(T32, T43))))
        return T_prod

    def get_achieved_goal(self):
        """logging.info('Logging runtime for get_centroid')
        start = time.time()
        u, v = self.get_centroid()
        logging.info(time.time() - start)
        logging.info('Logging runtime for rest of get_achieved_goal')
        start = time.time()
        z = self.ecm.getDepthImagePairs()[0][int(u), int(v)]
        
        x = (z/self.f) * (u - (self.w/2.0))
        y = (z/self.f) * (v - (self.h/2.0))
        logging.info(time.time() - start)
        #x = (z/f) * u
        #y = (z/f) * v
        return np.array([x, y, z])"""

        """Using the Jacobian end-effector control approach to determine an achieved goal position"""
        T_prod = self.get_transform_matrices_product()
        #base_coordinates = np.append(self.psm.get_marker_position(self.ecm.base_handle), 1.)
        base_coordinates = np.append(self.ecm.getTestGoalPosition(), 1.)
        goal = np.dot(T_prod, base_coordinates)
        
        return goal

    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer.number)
            self.viewer = None
        self.pr.stop()
        self.pr.shutdown()
    
    def _simulator_step(self):
        """for i in range(0, self.n_substeps):
            self.pr.step()
        """
        self.pr.step()

    def _reset_sim(self):
        """Resets the simulation and random initialization
        """
        self.pr.stop()
        #print (self.init_angles, end = '\r')
        self.ecm.setJointAngles(self.init_angles)
        self.pr.start()

    def _get_obs(self):
        """Returns the observation.
        """
        obs = {'observation' : self.ecm.getJointAngles(),
              'achieved_goal': self.get_achieved_goal(),
              'desired_goal' : self.desired_goal}
        return obs

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        obs = self._get_obs()['observation']
        valid = self._step_callback(obs, action)
        return valid 

    def _is_success(self):
        """Indicates whether or not the needle is passed optimally.
        """
        obs = self._get_obs()
        achieved = obs['achieved_goal']
        goal = obs['desired_goal']
        """logging.critical(str(achieved))
        return (achieved == goal).all()"""
        return np.linalg.norm(achieved - goal) < self.success_radius

    def _interaction_reward(self, achieved_goal, goal): 
        """Returns the reward based on the interaction result in the simulator
        """
        return -np.linalg.norm(achieved_goal - goal)**2.0
    
    def _env_setup(self):
        
        self.psm.setJointAngles(np.array([0.,np.radians(-50.0), 0.08, 0., 0., 0.]), np.radians(20.0))
        self.psm.setDynamicsMode(1)
        self.psm.setIkMode(0)
        self.pr.step()
        
    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self, obs, action):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        new_pos = obs + action
        valid = True
        logging.info('Logging runtime for bound check in _step_callback()')
        start = time.time()
        if (not ((new_pos[0] >= self.bounds[0][0]) and (new_pos[0] <= self.bounds[0][1]))) or (not ((new_pos[1] >= self.bounds[1][0]) and (new_pos[1] <= self.bounds[1][1]))) or (not ((new_pos[2] >= self.bounds[2][0]) and (new_pos[2] <= self.bounds[2][1]))) or (not ((new_pos[3] >= self.bounds[3][0]) and (new_pos[3] <= self.bounds[3][1]))):
            valid = False
        logging.info(str(time.time() - start))

        logging.info('Logging runtime for joint angle update in _step_callback()')
        start = time.time()
        if valid:
            self.ecm.setJointAngles(new_pos)
        logging.info(str(time.time() - start))
        return valid
       
