from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import math
import os
import torch
from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import euler2quat
import cv2

class SawyerBlockEnv(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,
            random_init=False,
            tasks = [{'goal': np.array([0.1, 0.8, 0.2]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3}], 
            goal_low=None,
            goal_high=None,
            hand_init_pos = (0, 0.6, 0.0),
            liftThresh = 0.04,
            rewMode = 'orig',
            rotMode='fixed',
            low_dim=False,
            hand_only=False,
            problem="rand",
            **kwargs
    ):
        self.quick_init(locals())
        hand_low=(-0.2, 0.4, 0.0)
        hand_high=(0.2, 0.8, 0.05)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./10,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.epcount = 0
        self.epsucc = []
        self.problem = problem
        self.low_dim = low_dim
        self.hand_only = hand_only
        self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 100
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.randomize = True
        self.hand_init_pos = np.array(hand_init_pos)
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )
        self.goal_space = Box(goal_low, goal_high)
        self.observation_space = Box(0, 1.0, (64,64,6))

   
    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    @property
    def model_name(self):
      path = os.environ['ASSETS_PATH']
      return path + "sawyer_xyz/sawyer_multiobject.xml"

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])

        ob = None
        ob = self._get_obs()
        reward  = self.compute_reward()
        self.curr_path_length +=1
        if self.curr_path_length == self.max_path_length:
            self._reset_hand()
            done = True
        else:
            done = False
          
        return ob, reward, done, {'pos': ob, 'hand': self.get_endeff_pos(), 'success':self.is_goal()}
   
    def _get_obs(self):
        im = self.sim.render(64, 64, camera_name="agentview")
        obs = np.concatenate([im, self.goalim], 2)
        return obs
    
    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        start_id = 9 + self.targetobj*2
        qpos[start_id:(start_id+2)] = pos.copy()
        qvel[start_id:(start_id+2)] = 0
        self.set_state(qpos, qvel)

    def render(self, mode=""):
        return self.sim.render(64, 64, camera_name="agentview")

    def sample_goal(self):
        start_id = 9 + self.targetobj*2
        qpos = self.data.qpos.flat.copy()
        ogpos = qpos[start_id:(start_id+2)]
        self._reset_hand(goal=True)
        goal_pos = np.random.uniform(
                -0.3,
                0.3,
                size=(2,),
            )
        if (self.problem == "1"):
          goal_pos = ogpos.copy()
          goal_pos[1] += 0.15
        elif (self.problem == "2"):
          goal_pos = ogpos.copy()
          goal_pos[0] += 0.15

        
        self._state_goal = goal_pos 
        self._set_obj_xyz(goal_pos) 
        
        if self.problem == "2":
          start_id2 = 9 + self.targetobj2*2
          qpos = self.data.qpos.flat.copy()
          ogpos2 = qpos[start_id2:(start_id2+2)]
          goal_pos2 = ogpos2.copy()
          goal_pos2[1] += 0.15
          self._state_goal2 = goal_pos2
          pl = self.targetobj
          self.targetobj = self.targetobj2
          self._set_obj_xyz(goal_pos2)
          self.targetobj = pl
        
        if not self.low_dim:
            self.goalim = self.sim.render(64, 64, camera_name="agentview")
            
        self._reset_hand()

        self._set_obj_xyz(ogpos)
        if self.problem == "2":
          pl = self.targetobj
          self.targetobj = self.targetobj2
          self._set_obj_xyz(ogpos2)
          self.targetobj = pl


    def reset_model(self):
        self._reset_hand()

        buffer_dis = 0.04
        block_pos = None
        for i in range(3):
            self.targetobj = i
            init_pos = np.random.uniform(
                -0.2,
                0.2,
                size=(2,),
            )
            if (self.problem == "1"):
              init_pos[1] = 0.0
              init_pos[0] = -0.15 + (0.15 * i)
            elif (self.problem == "2"):
              if i == 0:
                init_pos[0] = 0.0
                init_pos[1] = 0 
              if i == 1:
                init_pos[0] = 0.2 
                init_pos[1] = -0.2 
              if i == 2:
                init_pos[0] = 0.0 
                init_pos[1] = 0.15
            self.obj_init_pos = init_pos
            self._set_obj_xyz(self.obj_init_pos)

        for _ in range(100):
          self.do_simulation([0.0, 0.0])
        if not self.randomize:
          self.targetobj = 0
        else:
          self.targetobj = np.random.randint(3)
          if self.problem == "2":
            self.targetobj = 0
            self.targetobj2 = 2
        self.sample_goal()


        place = self.targetobj
        self.curr_path_length = 0
        self.epcount += 1
        o = self._get_obs()
        
        #Can try changing this
        return o

    def _reset_hand(self, goal=False):
        pos = self.hand_init_pos.copy()
        if (self.problem == "1"):
          if not goal:
            pos[1] -= 0.3
          else:
            if self.targetobj == 0:
              pos[0] = -0.2
            elif self.targetobj == 1:
              pos[0] = -0.0
            else:
              pos[0] = 0.2
            pos[1] += 0.15
        elif (self.problem == "2"):
          if not goal:
            pos[0] = -0.15
            pos[1] = 0.50
          else:
            pos[0] = 0.1
            pos[1] = 0.9

        for _ in range(10):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False


    def compute_reward(self):
        start_id = 9 + self.targetobj*2
        qpos = self.data.qpos.flat.copy()
        ogpos = qpos[start_id:(start_id+2)]
        dist = np.linalg.norm(ogpos - self._state_goal)
        dist2 = 0
    
    
        if self.problem == "2":
          start_id2 = 9 + self.targetobj2*2
          ogpos2 = qpos[start_id2:(start_id2+2)]
          dist2 = np.linalg.norm(ogpos2 - self._state_goal2)
          
        return - (dist + dist2)
      
    def is_goal(self):
        start_id = 9 + self.targetobj*2
        qpos = self.data.qpos.flat.copy()
        ogpos = qpos[start_id:(start_id+2)]
        dist = np.linalg.norm(ogpos - self._state_goal)
        dist2 = 0

  
        if self.problem == "2":
          start_id2 = 9 + self.targetobj2*2
          ogpos2 = qpos[start_id2:(start_id2+2)]
          dist2 = np.linalg.norm(ogpos2 - self._state_goal2)
          if (dist < 0.1) and (dist2 < 0.1):
            return 1
          else:
            return 0
          
        if (dist < 0.08):
          return 1
        else:
          return 0
