# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from newA3C import Game
from game_ac_network import GameACFFNetwork

LOCAL_T_MAX = 20 # repeat step size
ACTION_SIZE = 3 # action size
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
ACTION_SIZE=3

class A3CTrainingThread(object):
  def __init__(self,thread_index,global_network,initial_learning_rate,learning_rate_input,max_global_time_step,device):
    self.learn_rate=0
    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step
    self.local_network = GameACFFNetwork(ACTION_SIZE, thread_index, device)
    self.local_network.prepare_loss(ENTROPY_BETA)

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(self.local_network.total_loss, var_refs,gate_gradients=False,aggregation_method=None,colocate_gradients_with_ops=False)
# #     
    self.apply_gradients = tf.train.RMSPropOptimizer(self.learning_rate_input).apply_gradients(zip(self.gradients ,global_network.get_vars()))
      
    self.sync = self.local_network.sync_from(global_network)
    
    self.game_state = Game()
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    self.prev_local_t = 0

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def process(self, sess, global_t):
    states = []
    actions = []
    rewards = []
    values = []
    temp_reward=0
    terminal_end = False
    sess.run( self.sync )

    start_local_t = self.local_t

    for i in range(0,LOCAL_T_MAX):
#     while True:
#       sleep(100)
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
#       print(pi_)
      action = self.choose_action(pi_)
#       print(action)
      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      temp_action=[0,0,0]
      temp_action[action]=1
      self.game_state.process(temp_action)
      # receive game result
      reward = self.game_state.reward
#       print(self.game_state.terminal)
      terminal = self.game_state.terminal

      self.episode_reward += reward
      temp_reward=self.episode_reward

      # clip reward
      rewards.append(reward )

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()
      
      if terminal:
        terminal_end = True
        print("score={}".format(self.episode_reward))
        print("process:",self.thread_index," learn_rate:",self.learn_rate)
        self.episode_reward = 0
        self.game_state.reset()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate =  self._anneal_learning_rate(global_t)
    self.learn_rate=cur_learning_rate

    sess.run( self.apply_gradients,feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.learning_rate_input: cur_learning_rate} )
      
    diff_local_t = self.local_t - start_local_t
    return diff_local_t,temp_reward
