# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#-
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

from typing import Dict, Any, Tuple

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.pc_utils import generate_normalized_pc, process_robot_pc
from isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
from isaacgymenvs.utils.rotation3d import transformation_matrix_from_quat_trans, rot_matrix_from_quaternion

import torch
import numpy as np
import operator, random
from copy import deepcopy
import sys

import abc
from abc import ABC

import open3d as o3d

import pytorch3d

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

STL_FILE_DICT = {'base_link': "assets/urdf/allegro_hand_description/meshes/base_link.STL",
                 'link_0.0': "assets/urdf/allegro_hand_description/meshes/link_0.0.STL",
                 'link_1.0': "assets/urdf/allegro_hand_description/meshes/link_1.0.STL",
                 'link_2.0': "assets/urdf/allegro_hand_description/meshes/link_2.0.STL",
                 'link_3.0': "assets/urdf/allegro_hand_description/meshes/link_3.0.STL",
                 'link_3.0_tip': "assets/urdf/allegro_hand_description/meshes/modified_tip.STL",
                 'link_4.0': "assets/urdf/allegro_hand_description/meshes/link_0.0.STL",
                 'link_5.0': "assets/urdf/allegro_hand_description/meshes/link_1.0.STL",
                 'link_6.0': "assets/urdf/allegro_hand_description/meshes/link_2.0.STL",
                 'link_7.0': "assets/urdf/allegro_hand_description/meshes/link_3.0.STL",
                 'link_7.0_tip': "assets/urdf/allegro_hand_description/meshes/modified_tip.STL",
                 'link_8.0': "assets/urdf/allegro_hand_description/meshes/link_0.0.STL",
                 'link_9.0': "assets/urdf/allegro_hand_description/meshes/link_1.0.STL",
                 'link_10.0': "assets/urdf/allegro_hand_description/meshes/link_2.0.STL",
                 'link_11.0': "assets/urdf/allegro_hand_description/meshes/link_3.0.STL",
                 'link_11.0_tip': "assets/urdf/allegro_hand_description/meshes/modified_tip.STL",
                 'link_12.0': "assets/urdf/allegro_hand_description/meshes/link_12.0_right.STL",
                 'link_13.0': "assets/urdf/allegro_hand_description/meshes/link_13.0.STL",
                 'link_14.0': "assets/urdf/allegro_hand_description/meshes/link_14.0.STL",
                 'link_15.0': "assets/urdf/allegro_hand_description/meshes/link_15.0.STL",
                 'link_15.0_tip': "assets/urdf/allegro_hand_description/meshes/modified_tip.STL"}

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class Env(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool):
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True and config["env"]["legacy_obs"]:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments
        self.num_observations = config["env"]["numObservations"]
        self.num_states = config["env"].get("numStates", 0)
        self.num_actions = config["env"]["numActions"]

        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

    def set_test_mode(self, is_test=False):
        pass


    @abc.abstractmethod 
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self)-> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations


class VecTask(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.allocate_buffers()
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True


        # Enable camera sensor if obs is composed of visual input
        obs_modality = self.cfg["env"].get("observation", {})
        visual_obs_set = {"rgb", "depth", "pointcloud"}
        if len(visual_obs_set.intersection(set(obs_modality))) > 0:
            if not self.cfg["task"]["enableCameraSensors"]:
                self.cfg["task"]["enableCameraSensors"] = True
                print(f"Visual observation modality exist, set enableCameraSensors to be True")

        if self.cfg["env"]["legacy_obs"] or not self.headless:
            self.set_viewer()
        if not self.cfg["env"]["legacy_obs"]:
            self.load_meshes()
            # Camera
            use_rgb = "rgb" in obs_modality
            use_depth = "depth" in obs_modality
            use_pc = "pointcloud" in obs_modality
            use_camera = use_rgb or use_depth or use_pc
            if use_camera and not self.cfg["task"]["enableCameraSensors"]:
                raise RuntimeError(f"Observation includes visual information but camera sensors are not enable.")
            if use_rgb:
                self.rgb_tensor_list = []
                self.rgb_image_list = []
            if use_depth or use_pc:
                self.depth_tensor_list = []
                self.depth_image_list = []
                self.segmentation_image_list = []
            if use_pc:
                self.norm_pc_tensor: Optional[torch.Tensor] = None
                self.pc_list = []
                self.camera_pose_tensor: Optional[torch.Tensor] = None
                bd = self.cfg["env"]["observation"]["pointcloud"]["bound"]
                self.pc_bound = torch.tensor(bd, dtype=torch.float32, device=self.device)
                num_sample = self.cfg["env"]["observation"]["pointcloud"]["numSample"]
                if self.ablation_mode in ["no-tactile", "aug"]:
                    self.num_point = 680
                elif self.ablation_mode == "cam-tactile":
                    self.num_point = 640
                elif self.ablation_mode == "cam":
                    self.num_point = 512
                else:
                    self.num_point = num_sample

                # simultaneous pc visualization
                self.hand_geometry = o3d.geometry.PointCloud()
                self.object_geometry = o3d.geometry.PointCloud()
            if use_camera:
                self.create_camera()
            self.use_depth = use_depth
            self.use_rgb = use_rgb
            self.use_pc = use_pc

            # Modify env observation space
            if use_camera:
                cam = self.cfg["env"]["camera"]
                space_dict = {"obs": self.obs_space}
                if use_depth:
                    space_dict.update({"depth": spaces.Box(low=0, high=1, shape=(cam["height"], cam["width"], 1))})
                if use_pc:
                    space_dict.update(
                        {"pointcloud": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_point, 6))})
                if self.is_distillation:
                    space_dict.update(
                        {"student_obs": spaces.Box(np.ones(self.num_student_obs) * -np.Inf,
                                                   np.ones(self.num_student_obs) * np.Inf)})
                self.obs_space = spaces.Dict(space_dict)
        else:
            if self.is_distillation:
                space_dict = {"obs": self.obs_space}
                space_dict.update(
                    {"student_obs": spaces.Box(np.ones(self.num_student_obs) * -np.Inf,
                                               np.ones(self.num_student_obs) * np.Inf)})
                self.obs_space = spaces.Dict(space_dict)
        self.obs_dict = {}
        self.num_frame = 0

    def set_viewer(self):
        """Create the viewer."""

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs_dict[self.obs_type]), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        if self.cfg["env"]["pc_category"] == "laptop_smallpn_fulldata" or self.cfg["env"]["pc_category"] == "bucket_mediumpn_fulldata":
            self.object_class_pc_buf = torch.zeros(
                (self.num_envs, 256), device=self.device, dtype=torch.float
            )
        else:
            self.object_class_pc_buf = torch.zeros(
                (self.num_envs, 32), device=self.device, dtype=torch.float
            )
        if self.is_distillation:
            self.student_obs_buf = torch.zeros(
                (self.num_envs, self.num_student_obs), device=self.device, dtype=torch.float)
        if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
            self.obj_buf = torch.zeros(
                (self.num_envs, 13*2), device=self.device, dtype=torch.float
            )
        else:
            self.obj_buf = torch.zeros(
                (self.num_envs, 13), device=self.device, dtype=torch.float
            )
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the privileged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def update_controller(self):
        return


    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    def debug(self, info):
        pass

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.fetch_results(self.sim, True)
            self.update_controller()
            self.gym.simulate(self.sim)

        self.gym.fetch_results(self.sim, True)
        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        if self.ablation_mode in ["no-tactile", "multi-modality"]:
            self.obs_storage = torch.cat([self.obs_buf[:, :45], self.obs_buf[:, 61:130],
                                                self.obs_buf[:, 146:215], self.obs_buf[:, 231:300], 
                                                self.obs_buf[:, 316:]], dim=-1)
        else:
            self.obs_storage = self.obs_buf
        if not self.cfg["env"]["legacy_obs"] or self.is_distillation:
            if 'obs' in self.obs_dict:
                self.obs_dict["obs"]["obs"] = torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)
            else:
                self.obs_dict["obs"] = {'obs': torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)}
        else:
            self.obs_dict["obs"] = torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        if self.is_distillation:
            self.obs_dict["obs"]["student_obs"] = torch.clamp(self.student_obs_buf.clone(), -self.clip_obs, self.clip_obs).to(
                self.rl_device)
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx. 
        Should be implemented in an environment class inherited from VecTask.
        """  
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        if self.ablation_mode in ["no-tactile", "multi-modality"]:
            self.obs_storage = torch.cat([self.obs_buf[:, :45], self.obs_buf[:, 61:130],
                                                self.obs_buf[:, 146:215], self.obs_buf[:, 231:300], 
                                                self.obs_buf[:, 316:]], dim=-1)
        else:
            self.obs_storage = self.obs_buf
        if not self.cfg["env"]["legacy_obs"] or self.is_distillation:
            if 'obs' in self.obs_dict:
                self.obs_dict["obs"]["obs"] = torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)
            else:
                self.obs_dict["obs"] = {'obs': torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)}
            if not self.cfg["env"]["legacy_obs"]:
                cam = self.cfg["env"]["camera"]
                if self.use_depth and not self.use_pc:
                    self.obs_dict["obs"]["depth"] = torch.zeros((self.num_envs, cam["height"], cam["width"], 1), device=self.device, dtype=torch.float)
                if self.use_pc:
                    self.obs_dict["obs"]["pointcloud"] = torch.zeros((self.num_envs, self.num_point, 6), device=self.device,
                                                               dtype=torch.float)
        else:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        if self.is_distillation:
            self.obs_dict["obs"]["student_obs"] = torch.clamp(self.student_obs_buf, -self.clip_obs, self.clip_obs).to(
                self.rl_device)

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        if self.ablation_mode in ["no-tactile", "multi-modality"]:
            self.obs_storage = torch.cat([self.obs_buf[:, :45], self.obs_buf[:, 61:130],
                                                self.obs_buf[:, 146:215], self.obs_buf[:, 231:300], 
                                                self.obs_buf[:, 316:]], dim=-1)
        else:
            self.obs_storage = self.obs_buf
        if not self.cfg["env"]["legacy_obs"] or self.is_distillation:
            if 'obs' in self.obs_dict:
                self.obs_dict["obs"]["obs"] = torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)
            else:
                self.obs_dict["obs"] = {'obs': torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)}
        else:
            self.obs_dict["obs"] = torch.clamp(self.obs_storage, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        if self.is_distillation:
            self.obs_dict["obs"]["student_obs"] = torch.clamp(self.student_obs_buf, -self.clip_obs, self.clip_obs).to(
                self.rl_device)

        return self.obs_dict, done_env_ids

    def create_camera(self):
        obs_modality = self.cfg["env"]["observation"]
        use_rgb = "rgb" in obs_modality
        use_depth = "depth" in obs_modality or "pointcloud" in obs_modality
        use_pc = "pointcloud" in obs_modality

        camera_params = self.cfg["env"]["camera"]
        camera_props = gymapi.CameraProperties()
        camera_props.width = camera_params["width"]
        camera_props.height = camera_params["height"]
        camera_props.horizontal_fov = camera_params["fov"]
        camera_props.enable_tensors = self.cfg["sim"]["use_gpu_pipeline"]
        f = camera_params["width"] / 2 / np.tan(np.deg2rad(camera_params["fov"]) / 2)
        camera_mat = np.array([[f, 0, camera_params["width"] / 2], [0, f, camera_params["height"] / 2], [0, 0, 1]])
        camera_pose_list = []
        if use_pc:
            norm_pc = generate_normalized_pc(camera_params["width"], camera_params["height"], camera_mat)[None, ...]
            self.norm_pc_tensor = to_torch(norm_pc, device=self.device)

        cam_convention_quat = gymapi.Quat.from_euler_zyx(np.radians(90), -np.radians(90), 0).inverse()
        for env_id in range(len(self.envs)):
            camera_handle = self.gym.create_camera_sensor(self.envs[env_id], camera_props)
            if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
                cam_pos = gymapi.Vec3(0.73712237, 0.27106623, 0.40333985)
                cam_quat = gymapi.Quat.from_euler_zyx(*np.radians([1.24906045, 34.36755593, -122.84793373]))
            else:
                cam_pos = gymapi.Vec3(0.74333504, 0.24734959, 0.41341816)
                cam_quat = gymapi.Quat.from_euler_zyx(*np.radians([1.59006776, 34.56966209, -124.50969204]))

            self.gym.set_camera_transform(camera_handle, self.envs[env_id],
                                          gymapi.Transform(cam_pos, cam_quat))  
            ext_quat = cam_quat * cam_convention_quat
            # let's add some noise to camera position...
            cam_pos.x = cam_pos.x + 0.01 * np.random.normal()
            cam_pos.y = cam_pos.y + 0.01 * np.random.normal()
            pose_wxyz = torch.tensor([cam_pos.x, cam_pos.y, cam_pos.z, ext_quat.w, ext_quat.x, ext_quat.y, ext_quat.z],
                                     dtype=torch.float, device=self.device, requires_grad=False)
            camera_pose_list.append(pose_wxyz)

            if use_rgb:
                rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], camera_handle,
                                                                  gymapi.IMAGE_COLOR)
                self.rgb_tensor_list.append(rgb_tensor)
                self.rgb_image_list.append(gymtorch.wrap_tensor(rgb_tensor))
            if use_depth:
                depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], camera_handle,
                                                                    gymapi.IMAGE_DEPTH)
                segmentation_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_id], camera_handle,
                                                                           gymapi.IMAGE_SEGMENTATION)
                self.depth_tensor_list.append(depth_tensor)
                self.depth_image_list.append(gymtorch.wrap_tensor(depth_tensor))
                self.segmentation_image_list.append(gymtorch.wrap_tensor(segmentation_tensor))

        self.camera_pose_tensor = torch.stack(camera_pose_list)

    def load_meshes(self):
        self.meshes = {}
        for name in STL_FILE_DICT.keys():
            if name[-3:] == 'fsr' or name[:4] == 'palm' or name == 'world':
                continue
            try:
                mesh = o3d.io.read_triangle_mesh(STL_FILE_DICT[name])
                self.meshes[name] = mesh
            except:
                continue

    
    def fetch_imagined_pointcloud(self):
        all_rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).reshape(self.num_envs, -1, 13)
        all_pc = []
        fsr_pc = []
        for i in range(self.num_envs):
            meshes = []
            pc_tmp = []
            fsr_pc_tmp = []
            rb_dict = self.gym.get_actor_rigid_body_dict(self.envs[i], self.arm_hands[i])
            
            triggered_contact = list(torch.where(self.obs_buf[i, 45:61])[0].cpu().numpy())
            triggered_contact_name = []
            for idx in triggered_contact:
                triggered_contact_name.append(self.contact_sensor_names[idx])
            contact_box_size = [0.003011, 0.01, 0.02]
            for name, index in rb_dict.items():
                link_state = all_rb_states[i][index]  # position 0:3, rotation 3:7
                rot1 = rot_matrix_from_quaternion(link_state[3:7]).cpu().numpy()
                if name[:4] == 'palm' or name == 'world':
                    continue
                if name in triggered_contact_name:
                    fsr_sample_pts = torch.tensor([[random.uniform(-contact_box_size[0] / 2, contact_box_size[0] / 2) for _ in range(8)],
                                        [random.uniform(-contact_box_size[1] / 2, contact_box_size[1] / 2) for _ in range(8)],
                                        [random.uniform(-contact_box_size[2] / 2, contact_box_size[2] / 2) for _ in range(8)]]).transpose(0, 1)
                    fsr_sample_pts = fsr_sample_pts @ rot1.T
                    fsr_sample_pts = fsr_sample_pts + link_state[:3].cpu().numpy()
                    fsr_pc_tmp.append(fsr_sample_pts)
                    continue
                    
                if name in STL_FILE_DICT:
                    mesh = deepcopy(self.meshes[name])
                else:
                    continue

                if name[-3:] == 'tip':
                    mesh.scale(0.001, center=np.array([0, 0, 0]))
                    rot0 = np.array([[0, -1, 0],
                                     [1, 0, 0],
                                     [0, 0, 1]])
                    mesh.rotate(rot0, center=np.array([0, 0, 0.02]))
                    mesh.rotate(rot1, center=np.array([0, 0, 0.02]))
                    mesh.translate(link_state[:3].cpu().numpy() + np.array([0, 0, -0.02]))
                else:
                    mesh.rotate(rot1, center=np.array([0, 0, 0]))
                    mesh.translate(link_state[:3].cpu().numpy())

                meshes.append(mesh)
                pc_tmp.append(torch.from_numpy(np.asarray(mesh.sample_points_uniformly(number_of_points=8).points)))
            
            all_pc.append(torch.cat(pc_tmp, dim=0))
            if len(fsr_pc_tmp) > 0:
                fsr_pc_tmp = torch.cat(fsr_pc_tmp, dim=0)
                n_tactile = fsr_pc_tmp.shape[0]
                fsr_pc_tmp = torch.nn.functional.pad(fsr_pc_tmp, [0, 0, 0, 128 - n_tactile])
                fsr_pc_tmp[n_tactile:, :] = fsr_pc_tmp[0, :]
            else:
                fsr_pc_tmp = torch.zeros(128, 3)
            fsr_pc.append(fsr_pc_tmp)

        all_pc = torch.stack(all_pc, dim=0)
        fsr_pc = torch.stack(fsr_pc, dim=0)
        one_hot = torch.zeros((all_pc.shape[0], all_pc.shape[1], 3))
        one_hot[:, :, 0] = 1

        all_pc = torch.cat([all_pc, one_hot], dim=-1).cuda().to(torch.float)
        one_hot_fsr = torch.zeros((fsr_pc.shape[0], fsr_pc.shape[1], 3))
        one_hot_fsr[:, :, 2] = 1
        fsr_pc = torch.cat([fsr_pc, one_hot_fsr], dim=-1).cuda().to(torch.float)

        return all_pc, fsr_pc


    def fetch_camera_observations(self, imagined_pc=None, fsr_pc=None):
        if self.use_pc or self.use_depth or self.use_rgb:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            if self.use_depth and not self.use_pc:
                batch_depth_image = torch.stack(self.depth_image_list, dim=0).clone().detach()
                self.obs_dict["obs"]["depth"] = -batch_depth_image.unsqueeze(-1)
            
            if self.use_pc:
                batch_depth_image = torch.stack(self.depth_image_list, dim=0).clone().detach()
                batch_segmentation_image = torch.stack(self.segmentation_image_list, dim=0).clone().detach()

                pc = self.norm_pc_tensor * -batch_depth_image.unsqueeze(-1)
                batch_camera_pose = self.camera_pose_tensor.unsqueeze(1).unsqueeze(2)
                # Let's add some randomization on point cloud.
                pc = (1 + 0.006 * torch.randn_like(pc)) * pc
                pc = pytorch3d.transforms.quaternion_apply(batch_camera_pose[..., 3:], pc)
    
                pc = pc + batch_camera_pose[..., :3] - self.root_state_tensor[self.hand_indices, :3].unsqueeze(1).unsqueeze(1)
                pc = process_robot_pc(pc, self.pc_bound, num_sampled_points=self.num_point, segmentation=batch_segmentation_image)
                if self.obs_type == "full_stack_baoding" or self.obs_type == "partial_stack_baoding":
                    palm_center_offset = torch.tensor([5.3432e-01, -1.5243e-06, 2.0256e-01]).cuda()
                else:
                    palm_center_offset = torch.tensor([5.7225e-01, 1.0681e-04, 1.7850e-01]).cuda()
                if imagined_pc is not None:
                    imagined_pc[:, :, :3] -= palm_center_offset
                    zeros = torch.where(fsr_pc[:, :, :3] == torch.zeros(128, 3).cuda())
                    fsr_pc[zeros[0], :, :] = pc[zeros[0], 0, :].unsqueeze(1)
                    fsr_pc[:, :, :3] -= palm_center_offset
                    pc[:, :, :3] -= palm_center_offset
                    if self.num_point == 808:  # self.ablation_mode in ["multi-modality", "all"]:
                        self.obs_dict["obs"]["pointcloud"] = torch.cat([pc, imagined_pc, fsr_pc], dim=1)
                    elif self.num_point == 680:
                        self.obs_dict["obs"]["pointcloud"] = torch.cat([pc, imagined_pc], dim=1)
                    elif self.num_point == 512:
                        self.obs_dict["obs"]["pointcloud"] = pc
                    elif self.num_point == 640:
                        self.obs_dict["obs"]["pointcloud"] = torch.cat([pc, fsr_pc], dim=1)
                    else:
                        raise NotImplementedError
                else:
                    pc[:, :, :3] -= palm_center_offset
                    self.obs_dict["obs"]["pointcloud"] = pc

            self.gym.end_access_image_tensors(self.sim)
    
    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.cfg["env"]["legacy_obs"] or not self.headless:
            condition = True if self.viewer else False
        else:
            condition = False
        
        if condition:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_' + str(prop_idx) + '_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props 
        # and lastly loop over the ranges of the params 

        for actor, actor_properties in dr_params["actor_params"].items():

            # Loop over all envs as this part is not tensorised yet 
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                # randomise dof_props, rigid_body, rigid_shape properties 
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties  
                #          prop_attrs: 
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl)
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

