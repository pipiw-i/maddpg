# -*- coding: utf-8 -*-
# @Time : 2022/5/12 上午10:13
# @Author :  wangshulei
# @FileName: mpe_train.py
# @Software: PyCharm
from matplotlib import pyplot as plt
import os
import numpy as np
from RL_algorithm_package.maddpg.policy import maddpg_policy
from RL_algorithm_package.maddpg.mpe_env import mpe_env
import time


SEED = 65535

if __name__ == '__main__':
    # 环境初始化
    save_file = 'test_run1'
    env = mpe_env('simple_spread', seed=SEED)
    obs_dim, action_dim = env.get_space()
    agent_number = env.get_agent_number()
    # policy初始化
    maddpg_agents = maddpg_policy(obs_dim=obs_dim, action_dim=action_dim,
                                  agent_number=agent_number,
                                  actor_learning_rate=10e-4, critic_learning_rate=10e-4,
                                  action_span=0.5, soft_tau=5e-2, log_dir=save_file + '/results')
    maddpg_agents.load_models('run64_1', 5000)
    score = []
    avg_score = []
    for i_episode in range(5000):
        obs_n = env.mpe_env.reset()
        score_one_episode = 0
        for t in range(20):
            env.mpe_env.render()
            time.sleep(0.05)
            action_n = maddpg_agents.get_all_test_action(obs_n)
            new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(action_n)
            score_one_episode += reward_n[0][0]
            obs_n = new_obs_n
        maddpg_agents.logs(score_one_episode, 20 * (i_episode + 1))
        if (i_episode + 1) % 1000 == 0:
            if not os.path.exists(save_file + '/MADDPG_img'):
                os.makedirs(save_file + '/MADDPG_img')
            plt.plot(score)  # 绘制波形
            plt.plot(avg_score)  # 绘制波形
            plt.savefig(save_file + f"/MADDPG_img/MADDPG_score:{i_episode + 1}.png")
            maddpg_agents.save_models(save_file, i_episode + 1)
        score.append(score_one_episode)
        avg = np.mean(score[-100:])
        avg_score.append(avg)
        print(f"i_episode is {i_episode},score_one_episode is {score_one_episode},avg_score is {avg}")
