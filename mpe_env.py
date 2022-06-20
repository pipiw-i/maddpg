# -*- coding: utf-8 -*-
# @Time : 2022/5/12 上午10:20
# @Author :  wangshulei
# @FileName: mpe_env.py
# @Software: PyCharm
from mpe.environment import MultiAgentEnv
import mpe.scenarios as scenarios
from utils import space_n_to_shape_n


class mpe_env:
    def __init__(self,
                 mpe_env_name,
                 seed):
        self.mpe_env_name = mpe_env_name
        self.seed = seed
        self.mpe_env = self.env_init()

    def env_init(self):
        scenario = scenarios.load(self.mpe_env_name + '.py').Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                            shared_viewer=True)
        env.seed(self.seed)
        return env

    def get_space(self):
        obs_shape_n = space_n_to_shape_n(self.mpe_env.observation_space)
        act_shape_n = space_n_to_shape_n(self.mpe_env.action_space)
        return obs_shape_n[0][0], act_shape_n[0][0]

    def get_agent_number(self):
        return self.mpe_env.n
