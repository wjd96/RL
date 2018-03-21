# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import os
from game_ac_network import GameACFFNetwork
from a3c_training_thread import A3CTrainingThread

PARALLEL_SIZE = 1 # parallel thread size
ACTION_SIZE = 3 # action size
MAX_TIME_STEP = 10 * 10**7
THRESHOLD=100
device = "/cpu:0"
initial_learning_rate = 0.001
global_t = 0

global_network = GameACFFNetwork(ACTION_SIZE, -1, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,learning_rate_input, MAX_TIME_STEP,device = device)
  training_threads.append(training_thread)
# prepare session
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
filecount=0
saver = tf.train.Saver()

def train_function(parallel_index):
  global global_t
  global filecount
  training_thread = training_threads[parallel_index]

  while True:
    if global_t > MAX_TIME_STEP:
      break

    diff_global_t,reward = training_thread.process(sess, global_t)
    if reward>THRESHOLD:
        filecount+=1
        os.mkdir("d:/1/copy/"+str(filecount))
        print(reward)
        saver.save(sess,"d:/1/copy/"+str(filecount)+"/"+str(reward)+".ckpt")
    global_t += diff_global_t
      
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))

for t in train_threads:
  t.start() 
  
for t in train_threads:
  t.join()
