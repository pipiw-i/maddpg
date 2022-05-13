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
from RL_algorithm_package.maddpg.shared_exp import SharedExp

SEED = 65535


def run_mpe(save_file, actor_learning_rate, critic_learning_rate):
    env = mpe_env('simple_spread', seed=SEED)
    obs_dim, action_dim = env.get_space()
    agent_number = env.get_agent_number()
    # policy初始化
    maddpg_agents = maddpg_policy(obs_dim=obs_dim, action_dim=action_dim,
                                  agent_number=agent_number,
                                  actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                                  action_span=0.5, soft_tau=1e-2, log_dir=save_file + '/results')
    # 经验池初始化
    all_agent_exp = []
    for agent_index in range(agent_number):
        exp = SharedExp(exp_size=51200, batch_size=128, obs_dim=obs_dim, action_dim=action_dim,
                        r_dim=1, done_dim=1, agent_number=agent_number, agent_index=agent_index)
        all_agent_exp.append(exp)
    score = []
    avg_score = []
    # 暖机，得到足够用来学习的经验
    obs_n = env.mpe_env.reset()
    while not all_agent_exp[0].can_learn():
        for t in range(20):
            action_n = maddpg_agents.get_all_action(obs_n)
            # action_n = [np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0])]
            new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(action_n)
            for agent_index in range(agent_number):
                all_agent_exp[agent_index].exp_store(obs_n, action_n, reward_n, new_obs_n, done_n)
            obs_n = new_obs_n

    for i_episode in range(5000):
        obs_n = env.mpe_env.reset()
        score_one_episode = 0
        for t in range(20):
            env.mpe_env.render()
            # 每个智能体得到动作,动作的含义：第一个元素是交流信息，
            # 第二(向右)三(向左)个元素是力，第四(向上)五(向下)个也是力，可以为连续量
            # 实际的计算过程，用二三相减，得到向左或者是向右的力度，然后除质量，得到速度，最后得到位置，四五类似
            action_n = maddpg_agents.get_all_action(obs_n)
            # action_n = [np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0])]
            new_obs_n, reward_n, done_n, info_n = env.mpe_env.step(action_n)
            exp_index_n = []
            for agent_index in range(agent_number):
                all_agent_exp[agent_index].exp_store(obs_n, action_n, reward_n, new_obs_n, done_n)
                # 从三个智能体中的经验分别采样，分别学习，而不是全都学习相同位置的经验
                index, _ = all_agent_exp[agent_index].sample()
                exp_index_n.append(index)
            maddpg_agents.update(all_agent_exp, exp_index_n)
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


if __name__ == '__main__':
    run_mpe('run64', 1e-3, 1e-3)
