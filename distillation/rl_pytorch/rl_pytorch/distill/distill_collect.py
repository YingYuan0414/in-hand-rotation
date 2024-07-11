from datetime import datetime
import os
import time

from gym.spaces import Space
import gym

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import central_value
from rl_games.algos_torch import torch_ext

import wandb


class DistillCollector:
    def __init__(
            self,
            teacher_params,
            student_params,
            vec_env,
            num_transitions_per_env,
            num_learning_epochs,
            num_mini_batches,
            clip_param=0.2,
            gamma=0.998,
            lam=0.95,
            init_noise_std=0.05,
            surrogate_loss_coef=1.0,
            value_loss_coef=1.0,
            bc_loss_coef=1.0,
            entropy_coef=0.0,
            use_l1=True,
            learning_rate=1e-3,
            max_grad_norm=0.5,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=None,
            device='cpu',
            sampler='sequential',
            teacher_log_dir='run',
            student_log_dir='student_run',
            is_testing=False,
            print_log=True,
            apply_reset=False,
            teacher_resume="None",
            vidlogdir='video',
            vid_log_step=1000,
            log_video=False,
            enable_wandb=False,
            bc_warmup=True,
            teacher_data_dir=None,
            worker_id=0,
            warmup_mode=None,
            batch_size=None
    ):
        if not isinstance(vec_env.env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.env.observation_space
        self.action_space = vec_env.env.action_space
        self.state_space = vec_env.env.state_space

        self.device = device
        self.desired_kl = desired_kl
        self.writer = SummaryWriter(log_dir=student_log_dir, flush_secs=10)
        self.enable_wandb = enable_wandb

        self.schedule = schedule
        self.step_size = learning_rate

        # PPO components
        self.vec_env = vec_env
        self.clip_param = clip_param
        self.surrogate_loss_coef = surrogate_loss_coef
        self.value_loss_coef = value_loss_coef
        self.bc_loss_coef = bc_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_l1 = use_l1
        self.bc_warmup = bc_warmup

        self.teacher_config = teacher_config = teacher_params['config']
        self.student_config = student_config = student_params['config']
        self.config = teacher_config

        self.num_actors = teacher_config['num_actors']

        self.is_testing = is_testing
        self.vidlogdir = vidlogdir
        self.log_video = log_video
        self.vid_log_step = vid_log_step
        self.rnn_states = None
        self.clip_actions = self.config.get('clip_actions', True)

        # Basic information about environment.
        self.env_info = self.vec_env.get_env_info()

        self.num_agents = self.env_info.get('agents', 1)
        self.normalize_value = self.config.get('normalize_value', False)
        self.normalize_input = teacher_config['normalize_input']
        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = False # self.central_value_config is not None
        self.value_size = self.env_info.get('value_size',1)
        self.horizon_length = self.config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.multi_gpu = self.config.get('multi_gpu', False)
        self.mixed_precision = self.config.get('mixed_precision', False)
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)

        self.actions_num = self.action_space.shape[0]
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        
        if self.has_central_value:
            if isinstance(self.state_space, gym.spaces.Dict):
                self.state_shape = {}
                for k, v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        # We now define teacher network.
        self.teacher_builder = model_builder.ModelBuilder()
        self.teacher_network = self.teacher_builder.load(teacher_params)
        if isinstance(self.obs_shape, dict):
            self.teacher_obs_shape = self.obs_shape['obs']
        else:
            self.teacher_obs_shape = self.obs_shape
        self.teacher_build_config = {
            'actions_num': self.actions_num,
            'input_shape': self.teacher_obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.teacher_actor_critic = self.teacher_network.build(self.teacher_build_config)
        self.teacher_actor_critic.to(self.device)
        
        if self.has_central_value:
            print('Adding Central Value Network')
            from omegaconf import open_dict
            if 'model' not in self.config['central_value_config']:
                # print(self.config['central_value_config'])
                with open_dict(self.config):
                    self.config['central_value_config']['model'] = {'name': 'central_value'}
            builder = model_builder.ModelBuilder()
            tea_network = builder.load(self.config['central_value_config'])
            # self.config['central_value_config']['network'] = network
            teacher_cv_config = {
                'state_shape': self.state_shape,
                'value_size': self.value_size,
                'ppo_device': self.device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'normalize_value': self.normalize_value,
                'network': tea_network,
                'config': self.central_value_config,
                'writter': self.writer,
                'max_epochs': self.max_epochs,
                'multi_gpu': self.multi_gpu,
            }
            self.teacher_central_value_net = central_value.CentralValueTrain(**teacher_cv_config).to(self.device)

        # We now define student network.
        self.student_builder = model_builder.ModelBuilder()
        self.student_network = self.student_builder.load(student_params)
        self.student_obs_shape = self.obs_shape['student_obs']  # {'obs': self.obs_shape['student_obs'], 'pointcloud': self.obs_shape['pointcloud']}
        
        self.student_build_config = {
            'actions_num': self.actions_num,
            'input_shape': self.student_obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.student_actor_critic = self.student_network.build(self.student_build_config)
        self.student_actor_critic.to(self.device)

        self.optimizer = optim.Adam(
            self.student_actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm

        # Log
        self.teacher_log_dir = teacher_log_dir
        # student Log
        self.student_log_dir = student_log_dir
        self.print_log = print_log
        
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset
        self.teacher_resume = teacher_resume
        self.teacher_data_dir = teacher_data_dir
        self.worker_id = worker_id
        assert teacher_resume is not None
        if not os.path.exists(self.teacher_data_dir):
            os.makedirs(self.teacher_data_dir)

    def teacher_load(self, path):
        checkpoint = torch_ext.load_checkpoint(path)
        self.teacher_actor_critic.load_state_dict(checkpoint['model'])
        self.set_stats_weights(self.teacher_actor_critic, checkpoint)
        env_state = checkpoint.get('env_state', None)
        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)
        self.teacher_actor_critic.eval()

    def set_stats_weights(self, model, weights):
        if self.normalize_input and 'running_mean_std' in weights:
            model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            model.value_mean_std.load_state_dict(weights['reward_mean_std'])

    def get_weights(self):
        state = {}
        state['model'] = self.student_actor_critic.state_dict()
        return state

    def get_full_state_weights(self):
        state = self.get_weights()
        state['optimizer'] = self.optimizer.state_dict()

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def save(self, path):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(path, state)

    def _preproc_obs(self, obs_batch):
        import copy
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def get_teacher_central_value(self, obs_dict):
        return self.teacher_central_value_net.get_value(obs_dict)

    def get_action_values(self, model, obs, mode='teacher'):
        processed_obs = self._preproc_obs(obs['obs'])

        model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states
        }

        with torch.no_grad():
            res_dict = model(input_dict)
            if self.has_central_value and mode == 'teacher':
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                if mode == 'teacher':
                    value = self.get_teacher_central_value(input_dict)
                else:
                    raise NotImplementedError
                res_dict['values'] = value

        return res_dict

    def calc_gradients(self, model, input_dict, prev_actions):
        model.train()
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        batch_dict = {
                'is_train': True,
                'prev_actions': prev_actions,
                'obs': obs_batch,
        }
        
        res_dict = model(batch_dict)
        return res_dict

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.env.get_state()

        self.teacher_load(
            "{}/{}.pth".format(self.teacher_log_dir, self.teacher_resume))
        cur_reward_sum = torch.zeros(
            self.vec_env.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(
            self.vec_env.env.num_envs, dtype=torch.float, device=self.device)

        reward_sum = []
        episode_length = []

        for it in range(self.current_learning_iteration, num_learning_iterations):
            # report_gpu()
            ep_infos = []

            storage = {'obs': [], 'actions': [], 'sigmas': [], 'pointcloud': []}  # , 'pointcloud': []}

            # Rollout
            for i in range(self.num_transitions_per_env):
                if i % 100 == 99:
                    print(i)
                if self.apply_reset:
                    current_obs = self.vec_env.reset()
                    current_states = self.vec_env.get_state()
                
                teacher_obs = current_obs.copy()
                teacher_obs["obs"] = current_obs["obs"]["obs"] 
                
                # Compute the action
                with torch.no_grad():
                    res_dict = self.get_action_values(self.teacher_actor_critic, teacher_obs, mode='teacher')
                    teacher_actions = res_dict['actions']
                    teacher_mus = res_dict['mus']
                    teacher_sigmas = res_dict['sigmas']

                    storage['obs'].extend(current_obs['obs']['student_obs'])
                    storage['actions'].extend(teacher_mus)
                    storage['sigmas'].extend(teacher_sigmas)
                    storage['pointcloud'].extend(current_obs['obs']['pointcloud'])

                    next_obs, rews, dones, infos = self.vec_env.step(torch.clamp(teacher_actions, -1.0, 1.0))
                    next_states = self.vec_env.env.get_state()
                # Record the transition
                current_obs = next_obs
                current_states.copy_(next_states)
                ep_infos.append(infos)

                if self.print_log:
                    cur_reward_sum[:] += rews
                    cur_episode_length[:] += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    reward_sum.extend(
                        cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    episode_length.extend(
                        cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                    if i % 100 == 99:
                        print(np.mean(reward_sum), np.mean(episode_length))
                if i % 200 == 199:
                    for key in storage.keys():
                        storage[key] = torch.stack(storage[key], dim=0)
                        print(storage[key].shape)
                    save_dir = os.path.join(self.teacher_data_dir, "teacher_batch_{}_{}.pt".format(self.worker_id, int((i-199)/200)))
                    torch.save((storage['obs'], storage['actions'], storage['sigmas'], storage['pointcloud']), save_dir)  
                    storage = {'obs': [], 'actions': [], 'sigmas': [], 'pointcloud': []} 
                    reward_sum = []
                    episode_length = []
