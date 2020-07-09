from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import torch
import os
from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
import cv2
from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE


class SawyerDoorEnv(SawyerXYZEnv):
    def __init__(
            self,
            random_init=False,
            obs_type='plain',
            goal_low=None,
            goal_high=None,
            rotMode='fixed',
            **kwargs
    ):
        self.quick_init(locals())
        hand_low=(-0.3, 0.40, 0.05)
        hand_high=(0.3, 1, 0.05)
        obj_low=(0., 0.85, 0.1)
        obj_high=(0.1, 0.95, 0.1)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./20,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0.1, 0.95, 0.1], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.problem = "rand"

        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        if goal_low is None:
            goal_low = self.hand_low
        
        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.max_path_length = 100

        self.rotMode = rotMode
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
        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.door_angle_idx = self.model.get_joint_qpos_addr('doorjoint')
      
    @property
    def model_name(self):
      path = os.environ['ASSETS_PATH']
      return path + "sawyer_xyz/sawyer_door_pull.xml"
  
    def get_site_pos(self, siteName):
      _id = self.model.site_names.index(siteName)
      return self.data.site_xpos[_id].copy()


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
        ob = self._get_obs()
        reward  = self.compute_reward()
        self.curr_path_length +=1
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        info =  {'success': self.is_goal()}
        return ob, reward, done, info
   
    def _get_obs(self):
        im = self.sim.render(64, 64, camera_name="agentview")
        obs = np.concatenate([im, self.goalim], 2)
        return obs

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('handle').copy()
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )


    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        start_id =  10 + self.targetobj*2
        qpos[start_id:(start_id+2)] = pos.copy()
        qvel[start_id:(start_id+2)] = 0
        self.set_state(qpos, qvel)

    def _set_door_xyz(self, pos):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos
        qvel[self.door_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self):
        self._reset_hand()
        
        for i in range(3):
            self.targetobj = i
            init_pos = np.random.uniform(
                -0.2,
                0.2,
                size=(2,),
            )
            init_pos[1] = np.random.uniform(-0.2, 0)
            self._set_obj_xyz(init_pos)
        self.objHeight = self.data.get_geom_xpos('handle')[2]
        if self.problem == "3":
          self.goal = 0
        elif self.problem == "4":
          self.goal = -np.pi / 2
        else:
          self.goal = np.random.uniform(-np.pi / 2, 0)
        self._set_door_xyz(self.goal)
        self.goalim = self.sim.render(64, 64, camera_name="agentview")

        self._reset_hand()
        self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos
        if self.problem == "3":
          self._set_door_xyz(np.random.uniform(-np.pi / 2, -np.pi / 3))
        elif self.problem == "4":
          self._set_door_xyz(np.random.uniform(-np.pi / 3, 0))
        else:
          self._set_door_xyz(-np.pi / 3)

        self.curr_path_length = 0
        #Can try changing this
        o = self._get_obs()
        return o

    def _reset_hand(self, pos=None):
        if pos is None:
          pos = self.hand_init_pos
        for _ in range(10):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
            #self.do_simulation(None, self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False


    def is_goal(self):
        curr = self.data.qpos.copy()[self.door_angle_idx]
        if (curr - self.goal)**2 < (np.pi / 6):
          return 1
        return 0

    def compute_reward(self):
        curr = self.data.qpos.copy()[self.door_angle_idx]
        
        return (curr - self.goal)**2
