# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import pickle

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
import pytorch3d.transforms as transform
import torch.nn.functional as F
import json

import random

def read_dict_from_json(file_path):
	# Opening JSON file
	with open(file_path) as json_file:
		data = json.load(json_file)
	return data


def xyzw_to_wxyz(quat):
    # holy****, isaacgym uses xyzw format. pytorch3d uses wxyz format.
    new_quat = quat.clone()
    new_quat[:, :1] = quat[:, -1:]
    new_quat[:, 1:] = quat[:, :-1]
    return new_quat


# Debug script: python ./isaacgymenvs/train.py test=False task=AllegroArmLeftContinuous pipeline=cpu
class AllegroArmMOAR(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.training = True
        self.cfg = cfg
        self.object_size = self.cfg["env"]["objectSize"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.control_penalty_scale = self.cfg["env"]["controlPenaltyScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.m_lower = self.cfg["env"].get("m_low", 0.03)
        self.m_upper = self.cfg["env"].get("m_up", 0.3)

        self.relative_scale = self.cfg["env"].get("relScale", 0.5)

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.random_force_prob_scalar = self.cfg["env"].get("forceProbScalar", 0.25)
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        self.rotation_axis = self.cfg["env"]["axis"]

        self.randomize_friction_lower = 0.2
        self.randomize_friction_upper = 3.0
        self.randomize_mass_lower = self.m_lower
        self.randomize_mass_upper = self.m_upper

        self.sensor_thresh = self.cfg["env"].get("sensorThresh", 0.8)
        self.sensor_noise = self.cfg["env"].get("sensorNoise", 0.2)

        if self.rotation_axis == "x":
            self.rotation_id = 0
        elif self.rotation_axis == "y":
            self.rotation_id = 1
        else:
            self.rotation_id = 2

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.use_prev_target = self.cfg["env"]["usePrevTarget"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.force_debug = self.cfg["env"].get("force_debug", False)

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        self.spin_coef = self.cfg["env"].get("spin_coef", 1.0)
        self.aux_coef = self.cfg["env"].get("aux_coef", -0.1)
        self.contact_coef = self.cfg["env"].get("contact_coef", 1.0)
        self.main_coef = self.cfg["env"].get("main_coef", 1.0)
        self.vel_coef = self.cfg["env"].get("vel_coef", -0.3)
        self.torque_coef = self.cfg["env"].get("torque_coef", -0.01)
        self.work_coef = self.cfg['env'].get("work_coef", -0.0002)
        self.finger_coef = self.cfg['env'].get('finger_coef', 0.1)
        self.latency = self.cfg['env'].get("latency", 0.25)

        self.use_initial_rotation = self.cfg['env'].get('useInitRandomRotation', False)
        self.torque_control = self.cfg["env"].get("torqueControl", True)
        self.skill_step = self.cfg["env"].get("skill_step", 50)
        self.ignore_z = (self.object_type == "pen")

        self.robot_asset_files_dict = {
            "thick": "urdf/xarm6/xarm6_allegro_right_fsr_2023_thin.urdf"
        }

        self.asset_files_dict = {
            "ball": "urdf/objects/ball.urdf",
            "set_obj10_thin_block_corner": "urdf/objects/set_obj10_thin_block_corner.urdf",
            "set_obj11_cylinder": "urdf/objects/set_obj11_cylinder.urdf",
            "set_obj12_cylinder_corner": "urdf/objects/set_obj12_cylinder_corner.urdf",
            "set_obj13_irregular_block": "urdf/objects/set_obj13_irregular_block.urdf",
            "set_obj14_irregular_block_cross": "urdf/objects/set_obj14_irregular_block_cross.urdf",
            "set_obj15_irregular_block_time": "urdf/objects/set_obj15_irregular_block_time.urdf",
            "set_obj16_cylinder_axis": "urdf/objects/set_obj16_cylinder_axis.urdf",
            "set_obj1_regular_block": "urdf/objects/set_obj1_regular_block.urdf",
            "set_obj2_block": "urdf/objects/set_obj2_block.urdf",
            "set_obj3_block": "urdf/objects/set_obj3_block.urdf",
            "set_obj4_block": "urdf/objects/set_obj4_block.urdf",
            "set_obj5_block": "urdf/objects/set_obj5_block.urdf",
            "set_obj6_block_corner": "urdf/objects/set_obj6_block_corner.urdf",
            "set_obj7_block": "urdf/objects/set_obj7_block.urdf",
            "set_obj8_short_block": "urdf/objects/set_obj8_short_block.urdf",
            "set_obj9_thin_block": "urdf/objects/set_obj9_thin_block.urdf",
            "cross4_0": "urdf/objects/cross4_0.urdf", "cross4_1": "urdf/objects/cross4_1.urdf", "cross4_2": "urdf/objects/cross4_2.urdf", "cross4_3": "urdf/objects/cross4_3.urdf", "cross4_4": "urdf/objects/cross4_4.urdf"
        }

        self.object_sets = {
            "ball": ["ball"], 
            "cross": ["cross4_0", "cross4_1", "cross4_2", "cross4_3", "cross4_4"],
            "C": ['set_obj1_regular_block', 'set_obj2_block', 'set_obj3_block',
                  'set_obj4_block', 'set_obj5_block', 'set_obj6_block_corner',
                  'set_obj7_block', 'set_obj8_short_block', 'set_obj9_thin_block',
                  'set_obj10_thin_block_corner', 'set_obj11_cylinder', 'set_obj12_cylinder_corner',
                  'set_obj13_irregular_block', 'set_obj14_irregular_block_cross', 'set_obj15_irregular_block_time',
                  'set_obj16_cylinder_axis']
        }

        self.object_set_id = self.cfg["env"].get("objSet", "0")
        self.used_training_objects = self.object_sets[str(self.object_set_id)]
        self.num_training_objects = len(self.used_training_objects)

        if self.object_set_id == "cross":
            self.obj_init_pos_shift = {
                "org": (0.56, 0.0, 0.36),
                "new": (0.66, 0.01, 0.238), 
            }
        elif self.object_set_id == "ball":
            self.obj_init_pos_shift = {"new": [(0.63, 0.01, 0.25), (0.63, -0.02, 0.25)]}
        else:
            self.obj_init_pos_shift = {
                "org": (0.56, 0.0, 0.36),
                "new": (0.63, 0.01, 0.238)  
            }

        self.obj_init_type = self.cfg["env"].get("objInit", "org")

        if self.object_set_id == "ball":
            self.init_hand_qpos_override_dict = {
            "default" : {
                "joint_0.0": 0.0,
                "joint_1.0": 1.0,
                "joint_2.0": 0.0,
                "joint_3.0": 0.0,
                "joint_12.0": 1.3815,
                "joint_13.0": 0.0868,
                "joint_14.0": 0.1259,
                "joint_15.0": 0.0,
                "joint_4.0": 0.0048,
                "joint_5.0": 1.0,
                "joint_6.0": 0.0,
                "joint_7.0": 0.0,
                "joint_8.0": 0.0,
                "joint_9.0": 1.0,
                "joint_10.0": 0.0,
                "joint_11.0": 0.0
            }}
        else:
            self.init_hand_qpos_override_dict = {
                "default" : {
                    "joint_0.0": 0.0,
                    "joint_1.0": 0.0,
                    "joint_2.0": 0.0,
                    "joint_3.0": 0.0,
                    "joint_12.0": 1.3815,
                    "joint_13.0": 0.0868,
                    "joint_14.0": 0.1259,
                    "joint_15.0": 0.0,
                    "joint_4.0": 0.0048,
                    "joint_5.0": 0.0,
                    "joint_6.0": 0.0,
                    "joint_7.0": 0.0,
                    "joint_8.0": 0.0,
                    "joint_9.0": 0.0,
                    "joint_10.0": 0.0,
                    "joint_11.0": 0.0
                }
            }

        self.hand_init_type = self.cfg["env"].get("handInit", "default")
        self.hand_qpos_init_override = self.init_hand_qpos_override_dict[self.hand_init_type]

        assert self.obj_init_type in self.obj_init_pos_shift

        self.obs_type = self.cfg["env"]["observationType"]
        self.ablation_mode = self.cfg["env"]["ablation_mode"]

        if not (self.obs_type in ["partial_stack", "full_stack", "full_stack_pointcloud", "partial_stack_pointcloud", "full_stack_baoding", "partial_stack_baoding"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.palm_name = "palm"
        self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
                                     "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
                                     "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
                                     "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]

        self.tip_sensor_names = ["link_3.0_tip_fsr",  "link_7.0_tip_fsr",
                                "link_11.0_tip_fsr", "link_15.0_tip_fsr"]
        self.arm_sensor_names = ["link1", "link2", "link3", "link4", "link5", "link6"]

        self.n_stack = self.cfg['env'].get('obs_stack', 4)
        self.n_obs_dim = 85
        self.pc_ablation = self.cfg["env"]["pc_ablation"]  # if True, only disable pointcloud info in observation
        self.is_distillation = self.cfg["env"]["is_distillation"]
        if self.pc_ablation:
            self.num_obs_dict = {
                "full_no_vel": 50,
                "full": 72,
                "full_state": 88,
                "full_contact": 93,
                "partial_contact": 45 + 16 + 24,
                "partial_stack": (45 + 16 + 24) * self.n_stack,
                "full_stack": (45 + 16 + 24) * self.n_stack + 13,
                "full_stack_pointcloud": (45 + 16 + 24) * self.n_stack + 13,
                "partial_stack_cont": (45 + 16 + 24) * self.n_stack,
                "partial_stack_pointcloud": (45 + 16 + 24) * self.n_stack,
            }
        else:
            self.num_obs_dict = {
                "full_no_vel": 50,
                "full": 72,
                "full_state": 88,
                "full_contact": 93,
                "partial_contact": 45+16+24,
                "partial_stack": (45+16+24) * self.n_stack,
                "full_stack": (45+16+24) * self.n_stack+13,
                "nojoint": (16+24) * self.n_stack,
                "notactile": 69 * self.n_stack,
                "full_stack_baoding": (45+16+24) * self.n_stack+13*2,
                "partial_stack_baoding": (45+16+24) * self.n_stack,
                "full_stack_pointcloud": (45+16+24) * self.n_stack+13+32,
                "partial_stack_cont": (45 + 16 + 24) * self.n_stack,
                "partial_stack_pointcloud": (45 + 16 + 24) * self.n_stack,
            }

        if self.obs_type == "full_stack_pointcloud" and not self.pc_ablation:
            pkl_dir = "object_pc_embeddings_{}_pretrain_{}.pkl".format(self.cfg['env']['objSet'], self.cfg['env']['pc_category'])
            with open(pkl_dir, "rb") as f:  # object_pc_embeddings_{}_train.pkl
                self.pc_emb_dict = pickle.load(f)

        self.reward_mode = self.cfg["env"].get("rewardType", "free")
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        self.robot_stiffness = self.cfg["env"].get("robotStiffness", 10.0)

        num_states = 0

        if self.asymmetric_obs:
            if self.obs_type == "full_stack_pointcloud":
                num_states = 101 + 24 + 49 + 16 + 32
                if self.pc_ablation:
                    num_states = 101 + 24 + 49 + 16
            elif self.obs_type == "partial_stack_pointcloud":
                num_states = 101 + 24 + 49 + 16 + self.num_training_objects
            elif self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
                num_states = (66 + 13 * 2 + 22) + 24 + 49 + self.num_training_objects + 16
            else:
                num_states = 101 + 24 + 49 + self.num_training_objects + 16

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        if self.ablation_mode in ["no-tactile", "multi-modality"]:
            self.cfg["env"]["numObservations"] = 276
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 22
        if self.is_distillation:
            self.num_student_obs = (45 + 16 + 24) * self.n_stack

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.object_class_indices_tensor = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.last_obs_buf = torch.zeros((self.num_envs, self.n_obs_dim), device=self.device, dtype=torch.float)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))

        if self.cfg["env"]["legacy_obs"] or not self.headless:
            if self.viewer != None:
                cam_pos = gymapi.Vec3(5.4, 4.05, 0.57)
                cam_target = gymapi.Vec3(4.1, 5.35, 0.20)
                self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            if self.viewer:
                self.debug_contacts = np.zeros((16, 49), dtype=np.float32)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
             dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
             self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_arm_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Contact.
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.spin_axis = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.disable_sets = {
            'A': [0, 3, 6, 12, 13, 14, 9],
            'B': [1, 2, 4, 5, 7, 8, 10, 11]
        } # disable part of fingers.
        self.disable_mode = self.cfg["env"].get("disableSet", '0')
        self.use_disable = False

        if self.disable_mode in self.disable_sets:
            self.use_disable = True
            self.disable_sensor_idxes = torch.tensor(self.disable_sets[self.disable_mode],
                                                     dtype=torch.long, device=self.device)

        if self.rotation_axis == "x":
            self.all_spin_choices = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

        elif self.rotation_axis == "y":
            self.all_spin_choices = torch.tensor([[0.0, -1.0, 0.0]], device=self.device)

        elif self.rotation_axis == "z":
            self.all_spin_choices = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        else:
            assert False, "wrong spin axis"

        # Contact.
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)

        print("Contact Tensor Dimension", self.contact_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.last_actions = torch.zeros((self.num_envs, 22), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            self.object_init_pos = torch.zeros((self.num_envs, 2, 3), dtype=torch.float, device=self.device)
            self.object_init_quat = torch.zeros((self.num_envs, 2, 4), dtype=torch.float, device=self.device)
        else:
            self.object_init_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.object_init_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.relative_scale_tensor = torch.full((self.num_envs, 1), self.relative_scale, device=self.device)

        self.p_gain_val = 100.0
        self.d_gain_val = 4.0
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain_val
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain_val

        self.reset_goal_buf = self.reset_buf.clone()
        self.init_stack_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.last_contacts = torch.zeros((self.num_envs, 16), dtype=torch.float, device=self.device)
        self.contact_thresh = torch.zeros((self.num_envs, 16), dtype=torch.float, device=self.device)
        self.tip_contacts = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        # for the sim2real debugging use
        self.debug_target = []
        self.debug_qpos = []

        self.post_init()

    def get_internal_state(self):
        return self.root_state_tensor[self.object_indices, 3:7]

    def get_internal_info(self, key):
        if key == 'target':
            return self.debug_target
        elif key == 'qpos':
            return self.debug_qpos
        elif key == 'contact':
            return  self.sensed_contacts 
        elif key == 'obj':
            return torch.tensor(self.object_class_indices, dtype=torch.long, device=self.device).reshape(self.num_envs, -1)
        elif key == 'qinit':
            return self.object_init_quat

        return None

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.create_object_asset_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_object_asset_dict(self, asset_root):
        self.object_asset_dict = {}
        print("ENTER ASSET CREATING!")
        for used_objects in self.used_training_objects:
            object_asset_file = self.asset_files_dict[used_objects]
            object_asset_options = gymapi.AssetOptions()

            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            object_asset_options.disable_gravity = True

            goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            self.object_asset_dict[used_objects] = {'obj': object_asset, 'goal': goal_asset}

            self.object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        if self.object_set_id == "ball":
            arm_hand_asset_file = "urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.urdf"
        else:
            arm_hand_asset_file = self.robot_asset_files_dict[self.cfg["env"]["sensor"]]

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)

        # load arm and hand.
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)
        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        print("Num dofs: ", self.num_arm_hand_dofs)
        self.num_arm_hand_actuators = self.num_arm_hand_dofs 

        # Set up each DOF.
        self.actuated_dof_indices = [i for i in range(self.num_arm_hand_dofs)]

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        # This part is very important (damping)
        for i in range(22):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            if i < 6:
                robot_dof_props['velocity'][i] = 1.0
            else:
                robot_dof_props['velocity'][i] = 3.14

            robot_dof_props['effort'][i] = 20.0

            robot_dof_props['friction'][i] = 0.1
            robot_dof_props['stiffness'][i] = 0  
            robot_dof_props['armature'][i] = 0.1

            if i < 6:
                robot_dof_props['damping'][i] = 100.0
            else:
                robot_dof_props['damping'][i] = 0.0 
            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)

        print("DOF_LOWER_LIMITS", robot_lower_qpos)
        print("DOF_UPPER_LIMITS", robot_upper_qpos)

        # Set up default arm position.
        self.default_arm_pos = [0.00, 1.183, -1.541, 3.1416, 2.742, -1.569]  

        for i in range(self.num_arm_hand_dofs):
            if i < 6:
                self.arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            elif self.object_set_id == "ball" and i in [7, 15, 19]:
                self.arm_hand_dof_default_pos.append(1.0)
            else:
                self.arm_hand_dof_default_pos.append(0.0)
            self.arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        arm_hand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            object_start_pose = []
            for i in range(2): # two baoding balls
                object_start_pose_tmp = gymapi.Transform()
                object_start_pose_tmp.p = gymapi.Vec3()

                pose_dx, pose_dy, pose_dz = self.obj_init_pos_shift[self.obj_init_type][i]
                object_start_pose_tmp.p.x = arm_hand_start_pose.p.x + pose_dx
                object_start_pose_tmp.p.y = arm_hand_start_pose.p.y + pose_dy
                object_start_pose_tmp.p.z = arm_hand_start_pose.p.z + pose_dz
                object_start_pose_tmp.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

                object_start_pose.append(object_start_pose_tmp)
        else:
            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3()

            pose_dx, pose_dy, pose_dz = self.obj_init_pos_shift[self.obj_init_type]
            object_start_pose.p.x = arm_hand_start_pose.p.x + pose_dx
            object_start_pose.p.y = arm_hand_start_pose.p.y + pose_dy
            object_start_pose.p.z = arm_hand_start_pose.p.z + pose_dz
            object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.goal_displacement = gymapi.Vec3(0.2, 0.2, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            goal_start_pose.p = object_start_pose[0].p + self.goal_displacement
        else:
            goal_start_pose.p = object_start_pose.p + self.goal_displacement

        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 300
        max_agg_shapes = self.num_arm_hand_shapes + 300

        self.objects = []   # object handles
        self.arm_hands = [] # arm-hand handles
        self.envs = []      # environment pointers

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []

        self.object_class_indices = []

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.object_rb_handles = list(range(arm_hand_rb_count, arm_hand_rb_count + self.object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([arm_hand_start_pose.p.x,
                                           arm_hand_start_pose.p.y,
                                           arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x,
                                           arm_hand_start_pose.r.y,
                                           arm_hand_start_pose.r.z,
                                           arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, robot_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            for rb in range(arm_hand_rb_count):
                self.gym.set_rigid_body_segmentation_id(env_ptr, arm_hand_actor, rb, 2)
            self.hand_indices.append(hand_idx)

            # add object
            obj_class_indice = np.random.randint(0, len(self.used_training_objects), 1)[0]
            select_obj = self.used_training_objects[obj_class_indice]
            # randomize initial quat
            if self.object_set_id == "cross": 
                init_theta = random.uniform(-np.pi / 2, np.pi / 2)
                object_start_pose.r = gymapi.Quat(0, 0, np.cos(init_theta), np.sin(init_theta))

            if self.object_set_id == "ball":
                init_theta = random.uniform(-np.pi, np.pi)
                center_pos = (0.63, 0., 0.25)  
                radius = 0.015

                object_start_pose[0].p.x = arm_hand_start_pose.p.x + (center_pos[0] + radius * np.cos(init_theta))
                object_start_pose[0].p.y = arm_hand_start_pose.p.y + (center_pos[1] + radius * np.sin(init_theta))
                
                object_start_pose[0].p.x = arm_hand_start_pose.p.x + (center_pos[0] - radius * np.cos(init_theta))
                object_start_pose[0].p.y = arm_hand_start_pose.p.y + (center_pos[1] - radius * np.sin(init_theta))

            if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
                object_handle_tmp = []
                object_init_state = []
                object_indices = []
                obj_scale = np.random.uniform(0.95, 1.05)  

                for j in range(len(object_start_pose)):
                    object_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['obj'], object_start_pose[j], "object", i, 0, 0)
                    object_init_state.append([object_start_pose[j].p.x, object_start_pose[j].p.y, object_start_pose[j].p.z,
                                                object_start_pose[j].r.x, object_start_pose[j].r.y, object_start_pose[j].r.z, object_start_pose[j].r.w,
                                                0, 0, 0, 0, 0, 0])
                    object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
                    object_indices.append(object_idx)
                    self.object_class_indices.append(obj_class_indice)

                    # Randomize the object scale
                    self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
                    # Do some friction randomization
                    rand_friction =  np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                    hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
                    for p in hand_props:
                        p.friction = rand_friction
                    self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, hand_props)

                    object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                    for p in object_props:
                        p.friction = rand_friction

                    self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)


                    prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                    for p in prop:
                        p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                    self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                    object_handle_tmp.append(object_handle)
                self.objects.append(object_handle_tmp)
                self.object_indices.append(object_indices)
                self.object_init_state.append(object_init_state)
            
            else:
                object_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['obj'], object_start_pose, "object", i, 0, 0)
                self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                            object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                            0, 0, 0, 0, 0, 0])
                object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
                self.object_indices.append(object_idx)
                self.object_class_indices.append(obj_class_indice)
                if not self.pc_ablation and self.obs_type == "full_stack_pointcloud":
                    self.object_class_pc_buf[i, :] = self.pc_emb_dict[select_obj]

                # Randomize the object scale
                if self.object_set_id == "cross":
                    obj_scale = np.random.uniform(1.0, 1.1)
                else:
                    obj_scale = np.random.uniform(0.95, 1.1)
                self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
                # Do some friction randomization
                rand_friction =  np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction

                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)


                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

                self.objects.append(object_handle)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)
            
        self.object_class_indices_tensor = torch.tensor(self.object_class_indices, dtype=torch.long, device=self.device)

        palm_handles = self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, self.palm_name)
        self.palm_indices = to_torch(palm_handles, dtype=torch.int64)

        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                          for sensor_name in self.contact_sensor_names]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        arm_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                          for sensor_name in self.arm_sensor_names]
        self.arm_handle_indices = to_torch(arm_handles, dtype=torch.int64)

        tip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                       for sensor_name in self.tip_sensor_names]

        self.fingertip_handles = to_torch(tip_handles, dtype=torch.int64)

        self.object_class_indices = to_torch(self.object_class_indices, dtype=torch.int64) 
        self.object_one_hot_vector = F.one_hot(self.object_class_indices, num_classes=self.num_training_objects).float()

        # override!
        self.hand_override_info = [(self.gym.find_actor_dof_handle(env_ptr, arm_hand_actor, finger_name), self.hand_qpos_init_override[finger_name]) for finger_name in self.hand_qpos_init_override]

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]


        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, -1, 13)
        else:
            self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def post_init(self):
        all_qpos = {}

        arm_hand_dof_default_pos = []
        arm_hand_dof_default_vel = []
        for (idx, qpos) in self.hand_override_info:
            print("Hand QPos Overriding: Idx:{} QPos: {}".format(idx, qpos))
            self.arm_hand_default_dof_pos[idx] = qpos
            all_qpos[idx] = qpos

        for i in range(self.num_arm_hand_dofs):
            if i < 6:
                arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            elif i in all_qpos:
                arm_hand_dof_default_pos.append(all_qpos[i])
            else:
                arm_hand_dof_default_pos.append(0.0)
            arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(arm_hand_dof_default_vel, device=self.device)

    def compute_reward(self, actions):
        self.control_error = torch.norm(self.cur_targets - self.arm_hand_dof_pos, dim=1)

        # Lets do some calculation

        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            last_relative_pos = self.last_object_pos[:, 1] - self.last_object_pos[:, 0]
            relative_pos = self.object_pos[:, 1] - self.object_pos[:, 0]
            last_relative_pos = torch.nn.functional.normalize(last_relative_pos, dim=-1)
            relative_pos = torch.nn.functional.normalize(relative_pos, dim=-1)
            zero_pos = torch.zeros_like(relative_pos)
            zero_pos[:, 1] = -1.

            last_dot = torch.tensor([torch.dot(zero_pos[i], last_relative_pos[i]).item() for i in range(zero_pos.shape[0])]).unsqueeze(1).to(self.device)
            last_cross = torch.cross(zero_pos, last_relative_pos)
            last_relative_rot = torch.cat([last_cross, 1+last_dot], dim=-1)
            last_relative_rot = torch.nn.functional.normalize(last_relative_rot, dim=-1)

            dot = torch.tensor([torch.dot(zero_pos[i], relative_pos[i]).item() for i in range(zero_pos.shape[0])]).unsqueeze(1).to(self.device)
            cross = torch.cross(zero_pos, relative_pos)
            relative_rot = torch.cat([cross, 1+dot], dim=-1)
            relative_rot = torch.nn.functional.normalize(relative_rot, dim=-1)
        
        # Generate a normal vector to the spinning axis.
        tmp_vector = self.spin_axis + torch.randn_like(self.spin_axis)  
        vector_1 = torch.cross(self.spin_axis, tmp_vector)
        vector_1 = torch.nn.functional.normalize(vector_1, dim=-1)

        # Generate another vector to form a basis [spin_axis, v1, v2]
        vector_2 = torch.cross(self.spin_axis, vector_1)
        vector_2 = torch.nn.functional.normalize(vector_2, dim=-1)
        
        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(last_relative_rot)).transpose(1, 2)
            forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(relative_rot))
        else:
            inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.last_object_rot)).transpose(1, 2)
            forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.object_rot))

        vector_1_new = torch.bmm(inverse_rotation_matrix, vector_1.unsqueeze(-1))
        vector_1_new = torch.bmm(forward_rotation_matrix, vector_1_new).squeeze()

        rot_vec_coordinate_1 = (vector_1_new * vector_1).sum(-1).reshape(-1, 1)
        rot_vec_coordinate_2 = (vector_1_new * vector_2).sum(-1).reshape(-1, 1)
        rot_vec_coordinate_3 = (vector_1_new * self.spin_axis).sum(-1)

        dev_angle = 3.1415926 / 2 - torch.arccos(rot_vec_coordinate_3)
        dev_angle = torch.abs(dev_angle)

        rot_vec = torch.cat((rot_vec_coordinate_1, rot_vec_coordinate_2), dim=-1)
        rot_vec = torch.nn.functional.normalize(rot_vec, dim=-1)

        inner_prod = rot_vec[:, 0]
        theta_sign = torch.sign(rot_vec[:, 1])
        theta = theta_sign * torch.arccos(inner_prod)

        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = (torch.abs(self.torques) * torch.abs(self.dof_vel_finite_diff)).sum(-1)

        if self.reward_mode == 'finger':
            if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
                init_relative_rot = torch.zeros_like(relative_rot)
                init_relative_rot[:, -1] = 1.
                self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], \
                self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward_finger(
                    torch.tensor(self.spin_coef).to(self.device),
                    torch.tensor(self.aux_coef).to(self.device),
                    torch.tensor(self.main_coef).to(self.device),
                    torch.tensor(self.vel_coef).to(self.device),
                    torch.tensor(self.torque_coef).to(self.device),
                    torch.tensor(self.work_coef).to(self.device),
                    torch.tensor(self.contact_coef).to(self.device),
                    torch.tensor(self.finger_coef).to(self.device),
                    self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
                    self.consecutive_successes,
                    self.max_episode_length, self.fingertip_pos, self.object_pos, relative_rot, self.object_init_pos,
                    init_relative_rot, self.object_linvel,
                    self.object_angvel,
                    self.goal_pos, self.goal_rot, self.finger_contacts, self.tip_contacts, self.contact_coef,
                    self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.control_error,
                    self.control_penalty_scale, self.actions, self.action_penalty_scale,
                    self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.spin_axis,
                    theta * 20, dev_angle * 20, torque_penalty, work_penalty,
                    self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), self.object_set_id
                )
            else:
                self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], \
                self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward_finger(
                    torch.tensor(self.spin_coef).to(self.device),
                    torch.tensor(self.aux_coef).to(self.device),
                    torch.tensor(self.main_coef).to(self.device),
                    torch.tensor(self.vel_coef).to(self.device),
                    torch.tensor(self.torque_coef).to(self.device),
                    torch.tensor(self.work_coef).to(self.device),
                    torch.tensor(self.contact_coef).to(self.device),
                    torch.tensor(self.finger_coef).to(self.device),
                    self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
                    self.consecutive_successes,
                    self.max_episode_length, self.fingertip_pos, self.object_pos, self.object_rot, self.object_init_pos,
                    self.object_init_quat, self.object_linvel,
                    self.object_angvel,
                    self.goal_pos, self.goal_rot, self.finger_contacts, self.tip_contacts, self.contact_coef,
                    self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.control_error,
                    self.control_penalty_scale, self.actions, self.action_penalty_scale,
                    self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.spin_axis,
                    theta * 20, dev_angle * 20, torque_penalty, work_penalty,
                    self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), self.object_set_id
                )

        else:
            raise NotImplementedError

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]  # [num_env, 2, 7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]  # [num_env, 2, 3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]  # [num_env, 2, 4]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, ..., 0:7]
        self.goal_pos = self.goal_states[:, ..., 0:3]
        self.goal_rot = self.goal_states[:, ..., 3:7]

        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        if self.obs_type == "partial_stack":
            self.compute_contact_observations('ps')
        elif self.obs_type == "full_stack":
            self.compute_contact_observations('fs')
        elif self.obs_type == "full_stack_pointcloud":
            self.compute_contact_observations('fspc')
        elif self.obs_type == "partial_stack_pointcloud":
            self.compute_contact_observations('pspc')
        elif self.obs_type == "full_stack_baoding":
            self.compute_contact_observations('fsbd')
        elif self.obs_type == "partial_stack_baoding":
            self.compute_contact_observations('psbd')
        else:
            print("Unknown observations type!")

    def compute_contact_observations(self, mode='full'):
        if mode == 'ps':
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  # 66
                self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
                self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
                self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

                obs_end = 79 
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact
                self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                   obs_end + self.num_actions + 24 + 49 + self.num_training_objects] = self.object_one_hot_vector

                end_pos = obs_end + self.num_actions + 24 + 49 + self.num_training_objects
                self.states_buf[:, end_pos:end_pos + 16] = self.prev_targets[:, 6:22]

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0

            self.last_obs_buf[:, 22:45] = 0

            contacts = self.contact_tensor.reshape(-1, 49, 3).clone() 
            contacts = contacts[:, self.sensor_handle_indices, :]
            tip_contacts = contacts[:, self.fingertip_indices, :]

            contacts = torch.norm(contacts, dim=-1)
            tip_contacts = torch.norm(tip_contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()
            tip_contacts = torch.where(tip_contacts >= 0.5, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            if self.use_disable:
                sensed_contacts[:, self.disable_sensor_idxes] = 0
            # Do some data augmentation to the contact....
            self.sensed_contacts = sensed_contacts
            if self.cfg["env"]["legacy_obs"] or not self.headless:
                if self.viewer:
                    self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += (torch.rand_like(self.last_obs_buf[:, 6:22]) - 0.5) * 2 * 0.06


            self.last_obs_buf[:, 22:23+6] =  0 
            self.last_obs_buf[:, 23+6:23+22] = unscale(self.prev_targets,
                                                       self.arm_hand_dof_lower_limits,
                                                       self.arm_hand_dof_upper_limits)[:, 6:22]

            init_obs_ids = torch.where(self.init_stack_buf == 1)
            self.init_stack_buf[init_obs_ids] = 0
            self.obs_buf[init_obs_ids] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim]), dim=-1)
            self.finger_contacts = gt_contacts
            self.tip_contacts = tip_contacts
            
            if self.is_distillation:
                self.student_obs_buf[:, :] = self.obs_buf.clone()

        elif mode == 'fs':
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  
                self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
                self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
                self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

                obs_end = 79 
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact
                self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                   obs_end + self.num_actions + 24 + 49 + self.num_training_objects] = self.object_one_hot_vector  

                end_pos = obs_end + self.num_actions + 24 + 49 + self.num_training_objects
                self.states_buf[:, end_pos:end_pos + 16] = self.prev_targets[:, 6:22]

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0
            self.last_obs_buf[:, 22:45] = 0

            contacts = self.contact_tensor.reshape(-1, 49, 3).clone() 
            contacts = contacts[:, self.sensor_handle_indices, :] 
            tip_contacts = contacts[:, self.fingertip_indices, :]

            contacts = torch.norm(contacts, dim=-1)
            tip_contacts = torch.norm(tip_contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()
            tip_contacts = torch.where(tip_contacts >= 0.5, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            if self.use_disable:
                sensed_contacts[:, self.disable_sensor_idxes] = 0
            self.sensed_contacts = sensed_contacts
            if self.cfg["env"]["legacy_obs"] or not self.headless:
                if self.viewer:
                    self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += (torch.rand_like(self.last_obs_buf[:, 6:22]) - 0.5) * 2 * 0.06
            self.last_obs_buf[:, 22:23+6] =  0 
            self.last_obs_buf[:, 23+6:23+22] = unscale(self.prev_targets,
                                                       self.arm_hand_dof_lower_limits,
                                                       self.arm_hand_dof_upper_limits)[:, 6:22]

            init_obs_ids = torch.where(self.init_stack_buf == 1)[0]
            
            self.init_stack_buf[init_obs_ids] = 0
            
            self.obs_buf[init_obs_ids, :-13] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim-13]), dim=-1)
            
            self.finger_contacts = gt_contacts
            self.tip_contacts = tip_contacts

            if self.is_distillation:
                self.student_obs_buf[:, :] = self.obs_buf.clone()

            # add object observation
            self.obj_buf[:, :7] = self.object_pose
            self.obj_buf[:, 7:10] = self.object_linvel
            self.obj_buf[:, 10:13] = self.vel_obs_scale * self.object_angvel
            self.obs_buf = torch.cat((self.obs_buf, self.obj_buf), dim=-1)
        
        elif mode == 'fspc':
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  
                self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
                self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
                self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

                obs_end = 79 
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact
                if not self.pc_ablation:
                    if self.cfg["env"]["pc_category"] == "laptop_smallpn_fulldata" or self.cfg["env"]["pc_category"] == "bucket_mediumpn_fulldata":
                        self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                    obs_end + self.num_actions + 24 + 49 + 256] = self.object_class_pc_buf
                    else:
                        self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                    obs_end + self.num_actions + 24 + 49 + 32] = self.object_class_pc_buf  
                if self.pc_ablation:
                    end_pos = obs_end + self.num_actions + 24 + 49  
                else:
                    if self.cfg["env"]["pc_category"] == "laptop_smallpn_fulldata" or self.cfg["env"]["pc_category"] == "bucket_mediumpn_fulldata":
                        end_pos = obs_end + self.num_actions + 24 + 49 + 256
                    else:
                        end_pos = obs_end + self.num_actions + 24 + 49 + 32  
                self.states_buf[:, end_pos:end_pos + 16] = self.prev_targets[:, 6:22]

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0
            self.last_obs_buf[:, 22:45] = 0

            contacts = self.contact_tensor.reshape(-1, 49, 3).clone() 
            contacts = contacts[:, self.sensor_handle_indices, :]
            tip_contacts = contacts[:, self.fingertip_indices, :]

            contacts = torch.norm(contacts, dim=-1)
            tip_contacts = torch.norm(tip_contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()
            tip_contacts = torch.where(tip_contacts >= 0.5, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            if self.use_disable:
                sensed_contacts[:, self.disable_sensor_idxes] = 0
            # Do some data augmentation to the contact....
            self.sensed_contacts = sensed_contacts
            if self.cfg["env"]["legacy_obs"] or not self.headless:
                if self.viewer:
                    self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += (torch.rand_like(self.last_obs_buf[:, 6:22]) - 0.5) * 2 * 0.06

            self.last_obs_buf[:, 22:23+6] =  0 
            self.last_obs_buf[:, 23+6:23+22] = unscale(self.prev_targets,
                                                       self.arm_hand_dof_lower_limits,
                                                       self.arm_hand_dof_upper_limits)[:, 6:22]

            init_obs_ids = torch.where(self.init_stack_buf == 1)
            self.init_stack_buf[init_obs_ids] = 0
            if self.pc_ablation:
                tmp = 13
            else:
                tmp = 13+32
            self.obs_buf[init_obs_ids][:, :-tmp] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim-tmp]), dim=-1)
            self.finger_contacts = gt_contacts
            self.tip_contacts = tip_contacts

            if self.is_distillation:
                self.student_obs_buf[:, :] = self.obs_buf.clone()
            # add object observation
            self.obj_buf[:, :7] = self.object_pose
            self.obj_buf[:, 7:10] = self.object_linvel
            self.obj_buf[:, 10:13] = self.vel_obs_scale * self.object_angvel
            if self.pc_ablation:
                self.obs_buf = torch.cat((self.obs_buf, self.obj_buf), dim=-1)
            else:
                self.obs_buf = torch.cat((self.obs_buf, self.obj_buf, self.object_class_pc_buf), dim=-1)
        elif mode == 'pspc':
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  # 66
                self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
                self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
                self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

                obs_end = 79 
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact

                end_pos = obs_end + self.num_actions + 24 + 49 
                self.states_buf[:, end_pos:end_pos + 16] = self.prev_targets[:, 6:22]

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0
            self.last_obs_buf[:, 22:45] = 0

            contacts = self.contact_tensor.reshape(-1, 49, 3).clone()  
            contacts = contacts[:, self.sensor_handle_indices, :] 
            tip_contacts = contacts[:, self.fingertip_indices, :]

            contacts = torch.norm(contacts, dim=-1)
            tip_contacts = torch.norm(tip_contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()
            tip_contacts = torch.where(tip_contacts >= 0.5, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            if self.use_disable:
                sensed_contacts[:, self.disable_sensor_idxes] = 0
            # Do some data augmentation to the contact....
            self.sensed_contacts = sensed_contacts
            if self.cfg["env"]["legacy_obs"] or not self.headless:
                if self.viewer:
                    self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += (torch.rand_like(self.last_obs_buf[:, 6:22]) - 0.5) * 2 * 0.06

            self.last_obs_buf[:, 22:23+6] =  0 
            self.last_obs_buf[:, 23+6:23+22] = unscale(self.prev_targets,
                                                       self.arm_hand_dof_lower_limits,
                                                       self.arm_hand_dof_upper_limits)[:, 6:22]

            init_obs_ids = torch.where(self.init_stack_buf == 1)
            self.init_stack_buf[init_obs_ids] = 0
            self.obs_buf[init_obs_ids] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim]), dim=-1)
            self.finger_contacts = gt_contacts
            self.tip_contacts = tip_contacts
            if self.is_distillation:
                self.student_obs_buf[:, :] = self.obs_buf.clone()
        elif mode == 'fsbd':  # get observation of two baoding balls in hand. 
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  # 66
                self.states_buf[:, obj_obs_start:obj_obs_start + 7 * 2] = self.object_pose.reshape(-1, 7*2)
                self.states_buf[:, obj_obs_start + 7 * 2:obj_obs_start + 10 * 2] = self.object_linvel.reshape(-1, 3*2)
                self.states_buf[:, obj_obs_start + 10 * 2:obj_obs_start + 13 * 2] = self.vel_obs_scale * self.object_angvel.reshape(-1, 3*2)

                obs_end = 79+13 
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact
                self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                   obs_end + self.num_actions + 24 + 49 + self.num_training_objects] = self.object_one_hot_vector.reshape(-1, 2)[:, :1]  

                end_pos = obs_end + self.num_actions + 24 + 49 + self.num_training_objects
                self.states_buf[:, end_pos:end_pos + 16] = self.prev_targets[:, 6:22]

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0
            self.last_obs_buf[:, 22:45] = 0

            contacts = self.contact_tensor.reshape(-1, 49, 3).clone()  
            contacts = contacts[:, self.sensor_handle_indices, :] 
            tip_contacts = contacts[:, self.fingertip_indices, :]

            contacts = torch.norm(contacts, dim=-1)
            tip_contacts = torch.norm(tip_contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()
            tip_contacts = torch.where(tip_contacts >= 0.5, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            if self.use_disable:
                sensed_contacts[:, self.disable_sensor_idxes] = 0
            # Do some data augmentation to the contact....
            self.sensed_contacts = sensed_contacts
            if self.cfg["env"]["legacy_obs"] or not self.headless:
                if self.viewer:
                    self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += (torch.rand_like(self.last_obs_buf[:, 6:22]) - 0.5) * 2 * 0.06
            self.last_obs_buf[:, 22:23+6] =  0
            self.last_obs_buf[:, 23+6:23+22] = unscale(self.prev_targets,
                                                       self.arm_hand_dof_lower_limits,
                                                       self.arm_hand_dof_upper_limits)[:, 6:22]

            init_obs_ids = torch.where(self.init_stack_buf == 1)
            self.init_stack_buf[init_obs_ids] = 0
            self.obs_buf[init_obs_ids][:, :-13*2] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim-13*2]), dim=-1)
            self.finger_contacts = gt_contacts
            self.tip_contacts = tip_contacts

            if self.is_distillation:
                self.student_obs_buf[:, :] = self.obs_buf.clone()

            # add object observation
            self.obj_buf[:, :7*2] = self.object_pose.reshape(-1, 7*2)
            self.obj_buf[:, 7*2:10*2] = self.object_linvel.reshape(-1, 3*2)
            self.obj_buf[:, 10*2:13*2] = self.vel_obs_scale * self.object_angvel.reshape(-1, 3*2)
            self.obs_buf = torch.cat((self.obs_buf, self.obj_buf), dim=-1)
        elif mode == 'psbd':  # get observation of two baoding balls in hand. 
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  # 66
                self.states_buf[:, obj_obs_start:obj_obs_start + 7 * 2] = self.object_pose.reshape(-1, 7*2)
                self.states_buf[:, obj_obs_start + 7 * 2:obj_obs_start + 10 * 2] = self.object_linvel.reshape(-1, 3*2)
                self.states_buf[:, obj_obs_start + 10 * 2:obj_obs_start + 13 * 2] = self.vel_obs_scale * self.object_angvel.reshape(-1, 3*2)

                obs_end = 79+13 
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact
                self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                   obs_end + self.num_actions + 24 + 49 + self.num_training_objects] = self.object_one_hot_vector.reshape(-1, 2)[:, :1]  

                end_pos = obs_end + self.num_actions + 24 + 49 + self.num_training_objects
                self.states_buf[:, end_pos:end_pos + 16] = self.prev_targets[:, 6:22]

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0
            self.last_obs_buf[:, 22:45] = 0

            contacts = self.contact_tensor.reshape(-1, 49, 3).clone()  
            contacts = contacts[:, self.sensor_handle_indices, :] 
            tip_contacts = contacts[:, self.fingertip_indices, :]

            contacts = torch.norm(contacts, dim=-1)
            tip_contacts = torch.norm(tip_contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()
            tip_contacts = torch.where(tip_contacts >= 0.5, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            if self.use_disable:
                sensed_contacts[:, self.disable_sensor_idxes] = 0
            # Do some data augmentation to the contact....
            self.sensed_contacts = sensed_contacts
            if self.cfg["env"]["legacy_obs"] or not self.headless:
                if self.viewer:
                    self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += (torch.rand_like(self.last_obs_buf[:, 6:22]) - 0.5) * 2 * 0.06
            self.last_obs_buf[:, 22:23+6] =  0 
            self.last_obs_buf[:, 23+6:23+22] = unscale(self.prev_targets,
                                                       self.arm_hand_dof_lower_limits,
                                                       self.arm_hand_dof_upper_limits)[:, 6:22]

            init_obs_ids = torch.where(self.init_stack_buf == 1)
            self.init_stack_buf[init_obs_ids] = 0
            self.obs_buf[init_obs_ids][:, :] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim]), dim=-1)
            self.finger_contacts = gt_contacts
            self.tip_contacts = tip_contacts

        else:
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  
                self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
                self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
                self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

                obs_end = 79 
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact
                self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                   obs_end + self.num_actions + 24 + 49 + self.num_training_objects] = self.object_one_hot_vector

                end_pos = obs_end + self.num_actions + 24 + 49 + self.num_training_objects
                self.states_buf[:, end_pos:end_pos + 16] = self.prev_targets[:, 6:22]

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0
            self.last_obs_buf[:, 22:45] = 0

            contacts = self.contact_tensor.reshape(-1, 49, 3).clone() 
            contacts = contacts[:, self.sensor_handle_indices, :] 
            tip_contacts = contacts[:, self.fingertip_indices, :]

            contacts = torch.norm(contacts, dim=-1)
            tip_contacts = torch.norm(tip_contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()
            tip_contacts = torch.where(tip_contacts >= 0.5, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.clip(contacts / 30.0, 0.0, 1.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
            if self.use_disable:
                sensed_contacts[:, self.disable_sensor_idxes] = 0
            # Do some data augmentation to the contact....

            if self.cfg["env"]["legacy_obs"] or not self.headless:
                if self.viewer:
                    self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += (torch.rand_like(self.last_obs_buf[:, 6:22]) - 0.5) * 2 * 0.06
            self.last_obs_buf[:, 22:23+6] =  0 
            self.last_obs_buf[:, 23+6:23+22] = unscale(self.prev_targets,
                                                       self.arm_hand_dof_lower_limits,
                                                       self.arm_hand_dof_upper_limits)[:, 6:22]

            init_obs_ids = torch.where(self.init_stack_buf == 1)
            self.init_stack_buf[init_obs_ids] = 0
            self.obs_buf[init_obs_ids] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim]), dim=-1)
            self.finger_contacts = gt_contacts
            self.tip_contacts = tip_contacts

    def reset_spin_axis(self, env_ids, init_quat=None):
        env_ids_torch = torch.tensor(env_ids,  device=self.device).long()
        # Reset the init_quat...
        if init_quat is None:
            self.object_init_quat[
                env_ids_torch] = self.root_state_tensor[self.object_indices[env_ids_torch], 3:7]
        else:
            self.object_init_quat[env_ids_torch] = init_quat

        self.object_init_pos[env_ids_torch] = self.root_state_tensor[self.object_indices[env_ids_torch], 0:3]
        # Reset the axis
        self.spin_axis[env_ids_torch] = self.all_spin_choices[torch.randint(0, int(self.all_spin_choices.size(0)),
                                                                            (len(env_ids), ))]
        return

    def update_controller(self):
        previous_dof_pos = self.arm_hand_dof_pos.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        if self.torque_control:
            dof_pos = self.arm_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torques.clone()
            self.torques = torch.clip(self.torques, -20.0, 20.0)
            if self.debug_viz or self.force_debug:
                self.debug_target.append(self.cur_targets[:, 6:].clone())
                self.debug_qpos.append(self.arm_hand_dof_pos[:, 6:].clone())
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        return

    def refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    def reset_idx(self, env_ids, goal_env_ids, is_test=False):
        # generate random values
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        for env_id in env_ids:
            env = self.envs[env_id]
            handle = self.gym.find_actor_handle(env, 'object')
            prop = self.gym.get_actor_rigid_body_properties(env, handle)
            if not is_test:
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
            else:
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
            self.gym.set_actor_rigid_body_properties(env, handle, prop)

            if not is_test:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.arm_hands[env_id])
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.arm_hands[env_id], hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.arm_hands[env_id])
                for p in object_props:
                    p.friction = rand_friction

                if isinstance(self.objects[env_id], list):
                    for obj in self.objects[env_id]:
                        self.gym.set_actor_rigid_shape_properties(self.envs[env_id], obj, object_props)
                else:
                    self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.objects[env_id], object_props)
            else:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.arm_hands[env_id])
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.arm_hands[env_id], hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.arm_hands[env_id])
                for p in object_props:
                    p.friction = rand_friction
                
                if isinstance(self.objects[env_id], list):
                    for obj in self.objects[env_id]:
                        self.gym.set_actor_rigid_shape_properties(self.envs[env_id], obj, object_props)
                else:
                    self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.objects[env_id], object_props)


        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)


        # reset contact
        self.contact_thresh[env_ids] = torch.rand_like(self.contact_thresh[env_ids]) * self.sensor_thresh + 1.0
        self.last_contacts[env_ids] = 0.0

        # reset the pd-gain.
        self.randomize_p_gain_lower = self.p_gain_val * 0.30
        self.randomize_p_gain_upper = self.p_gain_val * 0.60
        self.randomize_d_gain_lower = self.d_gain_val * 0.75
        self.randomize_d_gain_upper = self.d_gain_val * 1.05

        if self.obs_type == 'partial_stack' or self.obs_type == 'full_stack' or self.obs_type == 'full_stack_pointcloud' or self.obs_type == 'partial_stack_pointcloud':
            self.obs_buf[env_ids] = 0
            self.init_stack_buf[env_ids] = 1

        self.p_gain[env_ids] = torch_rand_float(
            self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
            device=self.device).squeeze(1)
        self.d_gain[env_ids] = torch_rand_float(
            self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
            device=self.device).squeeze(1)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, ..., 0:2] + \
                                                                        self.reset_position_noise * rand_floats[:, 0:2].unsqueeze(1)
            self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, ..., self.up_axis_idx] + \
                                                                                        self.reset_position_noise * rand_floats[:, self.up_axis_idx].unsqueeze(1)
        else:
            self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, ..., 0:2] + \
                self.reset_position_noise * rand_floats[:, 0:2]
            self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, ..., self.up_axis_idx] + \
                self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        if not self.use_initial_rotation:
            # legacy codes.
            new_object_rot = randomize_rotation(torch.zeros_like(rand_floats[:, 3]),
                                                    torch.zeros_like(rand_floats[:, 4]), self.x_unit_tensor[env_ids],
                                                    self.y_unit_tensor[env_ids])

        else:
            new_object_rot = randomize_rotation(torch.zeros_like(rand_floats[:, 3]), rand_floats[:, 4],
                                                self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot.clone().unsqueeze(1)
        else:
            self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot.clone()
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            object_indices = torch.unique(torch.cat([self.object_indices[env_ids]], dim=-1).to(torch.int32))
        else:
            object_indices = torch.unique(torch.cat([self.object_indices[env_ids]]).to(torch.int32))

        # reset spinning axis
        if not self.use_initial_rotation:
            self.reset_spin_axis(env_ids)
        else:
            self.reset_spin_axis(env_ids, init_quat=new_object_rot)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        self.arm_hand_dof_pos[env_ids, :] = self.arm_hand_dof_default_pos
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel 
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_default_pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_default_vel

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        for env_id in env_ids:
            for (idx, qpos) in self.hand_override_info:
                self.dof_state[env_id * self.num_arm_hand_dofs + idx, 0] = qpos

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        for env_id in env_ids:
            self.object_init_pos[env_id] = self.root_state_tensor[self.object_indices[env_id], 0:3]
            self.object_init_quat[env_id] = self.root_state_tensor[self.object_indices[env_id], 3:7]

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            self.last_object_pos = self.root_state_tensor[self.object_indices, :3].clone()
        
        self.last_object_rot = self.root_state_tensor[self.object_indices, 3:7].clone()

        self.actions = actions.clone().to(self.device)

        if self.debug_viz or self.force_debug:
            self.debug_qpos = []
            self.debug_target = []

        if self.use_relative_control:

            self.actions = self.actions * self.act_moving_average + self.last_actions * (1.0 - self.act_moving_average)
            self.relative_scale_tensor = torch.full_like(self.relative_scale_tensor, self.relative_scale) * \
                                         (1 + (torch.rand_like(self.relative_scale_tensor) - 0.5) * 0.1)

            targets = self.prev_targets + self.relative_scale_tensor * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.arm_hand_dof_lower_limits[
                                                                              self.actuated_dof_indices],
                                                                          self.arm_hand_dof_upper_limits[
                                                                              self.actuated_dof_indices])
            self.prev_targets = self.cur_targets.clone()
            self.last_actions = self.actions.clone().to(self.device)

        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + \
                                                             (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                self.arm_hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def debug(self, info):
        print(info, self.root_state_tensor[self.object_indices, 3:7])
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()

        # Now we disable this...
        if self.rotation_axis == 'all':
            env_ids = list(torch.where(torch.rand(self.num_envs) < 1 / self.skill_step)[0])  # On average: 50 steps.
            self.reset_spin_axis(env_ids)

        if self.debug_viz or self.force_debug:
            self.debug_qpos = torch.stack(self.debug_qpos, dim=0)
            self.debug_target = torch.stack(self.debug_target, dim=0) # [control_steps, num_envs, 16]

        self.compute_reward(self.actions)

        if not self.cfg["env"]["legacy_obs"]:
            if self.cfg["env"]["pc_mode"] == "cam":
                self.fetch_camera_observations()
            elif self.cfg["env"]["pc_mode"] == "label":
                imagined_mesh, fsr_pc = self.fetch_imagined_pointcloud()
                self.fetch_camera_observations(imagined_mesh, fsr_pc)
            else:
                raise NotImplementedError

        if self.cfg["env"]["legacy_obs"] or not self.headless:
            condition = True if self.viewer else False
        else:
            condition = False
        if condition and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            main_vector = self.spin_axis.clone()
            inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.object_init_quat))
            inverse_rotation_matrix = inverse_rotation_matrix.permute(0, 2, 1)
            forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.object_rot))

            inverse_main_vector = torch.bmm(inverse_rotation_matrix, main_vector.unsqueeze(-1))
            current_main_vector = torch.bmm(forward_rotation_matrix, inverse_main_vector).squeeze()

            for i in range(self.num_envs):

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                objectm = (self.object_pos[i] + current_main_vector[i]).cpu().numpy()
                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objectm[0], objectm[1], objectm[2]], [0.85, 0.1, 0.85])

        # We do some debug visualization.
        if condition:
            for env in range(len(self.envs)):
                for i, contact_idx in enumerate(list(self.sensor_handle_indices)):

                    if self.debug_contacts[env, i] > 0.0:
                        self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(0.0, 1.0, 0.0))
                    else:
                        self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(1.0, 0.0, 0.0))

                import math
                if self.debug_viz:
                    if math.fabs(float(self.spin_axis[env, 0])) > 0.0:
                        color = (0.0, 0.0, 1.0)
                    elif math.fabs(float(self.spin_axis[env, 1])) > 0.0:
                        color = (0.0, 1.0, 0.0)
                    else:
                        color = (1.0, 0.0, 0.0)

                    for i, contact_idx in enumerate(list(self.arm_handle_indices)):
                        self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                    contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                    gymapi.Vec3(*color))


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_hand_reward_finger(
    spin_coef, aux_coef, main_coef, vel_coef, torque_coef, work_coef, contact_coef, finger_coef,
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, fingertip_pos, object_pos, object_rot, object_init_pos, object_init_rot, object_linvel, object_angvel, target_pos, target_rot,
    finger_contacts, tip_contacts, contact_scale: float, dist_reward_scale: float, rot_reward_scale: float,  rot_eps: float,
    control_error, control_penalty_scale: float, actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, main_vector, spinned_theta, dev_theta, torque_penalty, work_penalty,
    max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool, object_set_id: str
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    if object_set_id == "ball":
        object_pos_repeat = torch.cat([object_pos.reshape(-1, 2, 3)[:, 0, :].unsqueeze(1).repeat(1, 4, 1), 
                                        object_pos.reshape(-1, 2, 3)[:, 1, :].unsqueeze(1).repeat(1, 4, 1)], dim=1)
        
        distance = torch.sqrt(((object_pos_repeat - fingertip_pos.repeat(1, 2, 1)) ** 2).sum(-1))
        
    else:
        object_pos_repeat = object_pos.reshape(-1, 1, 3).repeat(1, 4, 1)
        distance = torch.sqrt(((object_pos_repeat - fingertip_pos) ** 2).sum(-1))
    distance_reward = torch.clip(0.1 / (4 * distance + 0.02), 0, 1).mean(-1) * finger_coef

    inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_init_rot)).transpose(1, 2)
    forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_rot))

    inverse_main_vector = torch.bmm(inverse_rotation_matrix, main_vector.unsqueeze(-1))
    current_main_vector = torch.bmm(forward_rotation_matrix, inverse_main_vector).squeeze()
    angle_difference = torch.arccos(torch.sum(main_vector * current_main_vector, dim=-1)) # The cosine similarity.
    object_angvel_norm = torch.norm(object_angvel, dim=-1)

    if object_set_id == "ball":
        quat_diff = quat_mul(object_rot, quat_conjugate(object_init_rot))
    else:
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    spinned_theta = torch.clip(spinned_theta, -3.14, 3.14)
    spin_reward = spin_coef * spinned_theta 
    if object_set_id == "ball":
        vel_reward = vel_coef * torch.norm(object_linvel, dim=-1).sum(dim=-1)
    else:
        vel_reward = vel_coef * torch.norm(object_linvel, dim=-1)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    finger_contact_sum = finger_contacts.sum(dim=-1).float()
    finger_contact_sum = torch.clip(finger_contact_sum, 0.0, 5.0)

    # The hand must hold the object. Otherwise it is penalized.
    contact_reward = finger_contact_sum * contact_coef 

    reward = spin_reward + vel_reward + contact_reward + distance_reward + \
             torque_penalty * torque_coef + work_penalty * work_coef + \
             action_penalty * action_penalty_scale + control_error * control_penalty_scale  

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) > 100.0, torch.ones_like(reset_goal_buf), reset_goal_buf)

    if object_set_id != "non-convex" and object_set_id != "ball":
        deviation = object_pos[:, ..., 0] - 0.59  # 0.59, 0.62
    else:
        deviation = object_pos[:, ..., 0] - 0.58

    if object_set_id == "ball":
        reward = torch.where(goal_dist[:, 0] >= fall_dist, reward + fall_penalty, reward)
        reward = torch.where(goal_dist[:, 1] >= fall_dist, reward + fall_penalty, reward)
        resets = torch.where(goal_dist[:, 0] >= fall_dist, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(goal_dist[:, 1] >= fall_dist, torch.ones_like(reset_buf), resets)
    else:
        reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)
        resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    if object_set_id == "non-convex" or object_set_id == "ball" or object_set_id == "cross_bmr":
        pass
    elif object_set_id == "cross" or object_set_id in ["cross3", "cross5", "cross_t", "cross_y"]:
        resets = torch.where(angle_difference > 0.2 * 3.1415926, torch.ones_like(reset_buf), resets)
    else:
        resets = torch.where(angle_difference > 0.4 * 3.1415926, torch.ones_like(reset_buf), resets)
    
    if object_set_id == "ball":
        pass
    else:
        resets = torch.where(deviation < 0, torch.ones_like(reset_buf), resets)

    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_corand_floatsnsecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) > 100.0, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@torch.jit.script
def randomize_z_rotation(rand0, rand1, y_unit_tensor, z_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, z_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


@torch.jit.script
def jit_identity(x):
    return x
