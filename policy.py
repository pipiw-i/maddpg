# -*- coding: utf-8 -*-
# @Time : 2022/5/11 下午9:49
# @Author :  wang
# @FileName: policy.py
# @Software: PyCharm
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorboard import summary
from RL_algorithm_package.maddpg.actor_critic_net import actor
from RL_algorithm_package.maddpg.actor_critic_net import critic


class maddpg_policy:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 agent_number,
                 actor_learning_rate,
                 critic_learning_rate,
                 action_span,
                 soft_tau,
                 log_dir,
                 gamma=0.99,
                 actor_name='actor',
                 critic_name='critic',
                 ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_span = action_span
        self.agent_number = agent_number
        self.soft_tau = soft_tau
        self.gamma = gamma
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir + '/maddpg_logs')
        self.actor_pred_list = []
        self.actor_target_list = []
        self.critic_pred_list = []
        self.critic_target_list = []
        for agent_index in range(agent_number):
            actor_pred = actor(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                               actor_learning_rate=actor_learning_rate, agent_index=agent_index,
                               trainable=True, actor_name=actor_name + 'pred',
                               action_span=action_span
                               )
            actor_target = actor(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                                 actor_learning_rate=actor_learning_rate, agent_index=agent_index,
                                 trainable=False, actor_name=actor_name + 'target',
                                 action_span=action_span
                                 )
            critic_pred = critic(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                                 critic_learning_rate=critic_learning_rate, agent_index=agent_index,
                                 trainable=True, critic_name=critic_name + 'pred',
                                 )
            critic_target = critic(obs_dim=obs_dim, act_dim=action_dim, agent_number=agent_number,
                                   critic_learning_rate=critic_learning_rate, agent_index=agent_index,
                                   trainable=False, critic_name=critic_name + 'target',
                                   )
            self.actor_pred_list.append(actor_pred)
            self.actor_target_list.append(actor_target)
            self.critic_pred_list.append(critic_pred)
            self.critic_target_list.append(critic_target)

    def __choose_action(self, s, agent_index):
        s = tf.reshape(s, [1, self.obs_dim])
        a = self.actor_pred_list[agent_index].actor(s)
        # 正式的测试中，可以去掉这个噪声 直接返回动作a[0]即可，这里是训练用，所以要加入噪声，使其能够充分的探索环境
        u = tfp.distributions.Normal(a, 0.2 * self.action_span)
        action = tf.squeeze(u.sample(1), axis=0)[0]
        action = tf.clip_by_value(action, clip_value_min=-self.action_span, clip_value_max=self.action_span)
        return action

    def get_all_action(self, obs_n):
        action_n = [self.__choose_action(obs.astype(np.float32), agent_index) for obs, agent_index in
                    zip(obs_n, range(self.agent_number))]
        action_n = [action.numpy() for action in action_n]
        return action_n

    def soft_param_update(self, target_model, pred_model):
        """
        采用软更新的方式进行参数的更新，不采用DQN中的直接赋值操作，也可以采用别的软更新方式来实现。
        :param pred_model: 预测网络
        :param target_model: 目标网络
        """
        param_target = target_model.get_weights()
        param_pred = pred_model.get_weights()
        param = []
        for i in range(len(param_target)):
            param_target[i] = param_target[i] * (1 - self.soft_tau)
            param_pred[i] = param_pred[i] * self.soft_tau
            param.append(np.add(param_pred[i], param_target[i]))
        target_model.set_weights(param)

    def update(self, all_agent_exp, index):
        exp_n = []
        for agent_index in range(self.agent_number):
            now_agent_index, obs, action, rew, obs_, done = all_agent_exp[agent_index].get_exp_from_index(index)
            exp = [now_agent_index, obs, action, rew, obs_, done]
            exp_n.append(exp)
        # 更新网络
        with tf.GradientTape(persistent=True) as Tape:
            # actor_pred 输入obs 得到的输出列表，用于更新actor网络
            agent_action = []
            # actor_target 输入obs_得到的输出列表,用于更新critic网络
            agent_action_ = []
            for agent_index in range(self.agent_number):
                # 得到当前的状态obs --> exp_n[agent_index][1]
                # 使用actor计算出相应的动作输出，用于更新actor网络
                actor_pred = self.actor_pred_list[agent_index].actor
                obs = exp_n[agent_index][1]
                agent_action.append(actor_pred(obs))
                actor_target = self.actor_target_list[agent_index].actor
                obs_ = exp_n[agent_index][4]
                agent_action_.append(actor_target(obs_))

            for agent_index in range(self.agent_number):
                actor_pred = self.actor_pred_list[agent_index].actor
                critic_pred = self.critic_pred_list[agent_index].critic
                # 得到当前的状态obs --> exp_n[agent_index][1]
                obs = exp_n[agent_index][1]
                # 输入到critic网络，输出预测值，这里当智能体的个数不同的时候，需要修改
                ###############################################################################
                Q_pred = critic_pred([obs, agent_action[0], agent_action[1], agent_action[2]])
                ###############################################################################
                actor_pred_loss = - tf.math.reduce_mean(Q_pred)
                # 用于更新actor网络
                gradients = Tape.gradient(actor_pred_loss, actor_pred.trainable_variables)
                actor_pred.optimizer.apply_gradients(zip(gradients, actor_pred.trainable_variables))
                # 更新critic网络
                # agent_action_
                critic_target = self.critic_target_list[agent_index].critic
                action = exp_n[agent_index][2]
                reward = exp_n[agent_index][3]
                obs_ = exp_n[agent_index][4]
                ###############################################################################
                # 这里当智能体的个数不同的时候，需要修改
                Q_pred_critic = critic_pred([obs, action[:, 0, :], action[:, 1, :], action[:, 2, :]])
                Q_target_critic = reward + self.gamma * critic_target(
                    [obs_, agent_action_[0], agent_action_[1], agent_action_[2]])
                ###############################################################################
                loss_critic = tf.keras.losses.mse(Q_target_critic, Q_pred_critic)
                loss_critic = tf.reduce_mean(loss_critic)
                critic_gradients = Tape.gradient(loss_critic, critic_pred.trainable_variables)
                critic_pred.optimizer.apply_gradients(zip(critic_gradients, critic_pred.trainable_variables))
        for agent_index in range(self.agent_number):
            self.soft_param_update(self.critic_target_list[agent_index].critic,
                                   self.critic_pred_list[agent_index].critic)
            self.soft_param_update(self.actor_target_list[agent_index].actor,
                                   self.actor_pred_list[agent_index].actor)

    def logs(self, score, learn_step):
        # print('... saving logs ...')
        with self.writer.as_default():
            tf.summary.scalar("score", score, learn_step)

    def save_models(self, save_file, episode):
        print('... saving models ...')
        if not os.path.exists(save_file + '/maddpg_model'):
            os.makedirs(save_file + '/maddpg_model')
            os.makedirs(save_file + '/maddpg_model/actor_pred')
            os.makedirs(save_file + '/maddpg_model/actor_target')
            os.makedirs(save_file + '/maddpg_model/critic_pred')
            os.makedirs(save_file + '/maddpg_model/critic_target')
        for agent_index in range(self.agent_number):
            self.actor_pred_list[agent_index].actor.save(
                save_file + f'/maddpg_model/actor_pred/agent_{agent_index}_{episode}.h5')
            self.actor_target_list[agent_index].actor.save(
                save_file + f'/maddpg_model/actor_target/agent_{agent_index}_{episode}.h5')
            self.critic_pred_list[agent_index].critic.save(
                save_file + f'/maddpg_model/critic_pred/agent_{agent_index}_{episode}.h5')
            self.critic_target_list[agent_index].critic.save(
                save_file + f'/maddpg_model/critic_target/agent_{agent_index}_{episode}.h5')

    def load_models(self, save_file, episode):
        print('... loading models ...')
        for agent_index in range(self.agent_number):
            self.actor_pred_list[agent_index].actor = tf.keras.models.load_model(
                save_file + f'/maddpg_model/actor_pred/agent_{agent_index}_{episode}.h5')
            self.actor_target_list[agent_index].actor = tf.keras.models.load_model(
                save_file + f'/maddpg_model/actor_target/agent_{agent_index}_{episode}.h5')
            self.critic_pred_list[agent_index].critic = tf.keras.models.load_model(
                save_file + f'/maddpg_model/critic_pred/agent_{agent_index}_{episode}.h5')
            self.critic_target_list[agent_index].critic = tf.keras.models.load_model(
                save_file + f'/maddpg_model/critic_target/agent_{agent_index}_{episode}.h5')
