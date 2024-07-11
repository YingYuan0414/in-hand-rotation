import os
import time
import numpy as np
import random
import copy
import torch
import yaml

from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import env_configurations
from rl_games.common import experiment
from rl_games.common import tr_helpers

from rl_games.algos_torch import model_builder
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
import rl_games.networks

def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')
class Runner:
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('openloop', lambda **kwargs: a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('openloop', lambda **kwargs: players.OpenloopPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_continuous_collect',
                                             lambda **kwargs: players.PpoPlayerContinuousCollect(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cpu')

        self.player = None

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())
        
        if params["config"].get('multi_gpu', False):
            self.seed += int(os.getenv("LOCAL_RANK", "0"))
        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None
        if self.seed:

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            
            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += int(os.getenv("LOCAL_RANK", "0"))

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_conf):
        self.default_config = yaml_conf['params']
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()

    def run_play(self, args):
        print('Started to play')
        if self.player is None:
            self.player = self.create_player()
            _restore(self.player, args)
            _override_sigma(self.player, args)

        self.player.run()
        self.player.post_run(self)
        self.device = self.player.device

    def reset_player(self):
        if self.player is None:
            return
        self.player.reset()

    def create_player(self):
        if self.params['config']['player_collect']:
            return self.player_factory.create(self.algo_name + '_collect', params=self.params)
        return self.player_factory.create(self.algo_name, params=self.params)

    def build_pose_prediction_dataset(self):
        class Dataset:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.n = self.x.size(0)

            def sample(self, batchsize):
                indices = torch.randint(0, self.n, (batchsize, ))
                return self.x[indices], self.y[indices]

        rnn = self.record_data['rnn'].permute(0, 2, 1, 3) #[T, 2, ENVS, DATA]
        gt = self.record_data['gt']

        rnn = rnn.reshape(rnn.size(0) * rnn.size(1), -1) #rnn.size(2) * rnn.size(3))
        gt = gt.reshape(gt.size(0) * gt.size(1), -1)
        return Dataset(rnn, gt)

    def reset(self):
        self.record_data = None

    def run(self, args):
        load_path = None

        if args['train']:
            self.run_train(args)

        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)

    def set_record_data(self, **kwargs):
        self.record_data = kwargs
