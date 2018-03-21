import numpy as np
import tensorflow as tf
from game_ac_network import GameACFFNetwork
from a3c_training_thread import A3CTrainingThread
from newA3C import Game
 
sess = tf.Session()
saver = tf.train.import_meta_graph("d:/1/copy/81/23.ckpt.meta")
saver.restore(sess, "d:/1/copy/81/23.ckpt")
device = "/cpu:0"
graph = tf.get_default_graph() 

game=Game()
while True:
    temp_st=np.reshape(game.s_t,(1,84,84,4))
    s_t=graph.get_tensor_by_name("net_0/s:0")
    pi_=graph.get_tensor_by_name("net_0/action:0")
    pi_t=sess.run(pi_,feed_dict={s_t:[game.s_t]})
#     print(pi_t)
#     pi_, value_ =network.run_policy_and_value(sess, game.s_t)
    action=np.random.choice(range(len(pi_t[0])), p=pi_t[0])
    print(action)
    temp_action=[0,0,0]
    temp_action[action]=1
    game.process(temp_action)
    game.update()
    if game.terminal:
        game.reset()


