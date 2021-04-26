"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""
from simple_emulator import CongestionControl

# We provided a simple algorithms about block selection to help you being familiar with this competition.
# In this example, it will select the block according to block's created time first and radio of rest life time to deadline secondly.
from simple_emulator import BlockSelection

from simple_emulator import constant
import numpy as np;

# for tf version < 2.0
# import tensorflow as tf

# for tf version >= 2.0
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import random

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        temp_p = probs.ravel()
        print(temp_p)
        if np.isnan(temp_p[0]):
            temp_p[0] = random.uniform(0, 1.0)
            temp_p[1] = random.uniform(0,1-temp_p[0])
            temp_p[2] = 1 - temp_p[0] - temp_p[1]
        return np.random.choice(np.arange(probs.shape[1]), p=temp_p)   # return a int

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

EPISODE = 20 # change every EPISODE times
N_F = EPISODE * 3 # speed,losepacket,application_speed
N_A = 3 # +10,0,-10
Lambda_init = 1.0 # random choose
random_counter_init = 100 # decline Lambda after random_counter
MAX_BANDWITH = 15000 # standardlize to 1

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND=False
        self.send_rate = 40.0
        self.cwnd = 5000

        self.counter = 0 # EPISODE counter

        self.result_list = []
        self.last_state = []
        self.Lambda = Lambda_init
        self.random_counter = random_counter_init
        
        for i in range(EPISODE):
            self.last_state.append(40.0/MAX_BANDWITH)
        for i in range(EPISODE):
            self.last_state.append(0)
        for i in range(EPISODE):
            self.last_state.append(40.0/MAX_BANDWITH)

    def estimate_bandwidth(self, cur_time, event_info):
        event_type = event_info["event_type"]
        event_time = cur_time

        # Preparing for the bandwidth estimator's decision
        if event_type == EVENT_TYPE_DROP:
            self.result_list.append(1)
        else:
            self.result_list.append(0)

        self.counter += 1
        if self.counter == EPISODE: # choose action every EPISODE times
            self.counter = 0
            print()
            print("EPISODE:")
            print()

            # declining random rate
            self.random_counter-=1
            if self.random_counter <= 0:
                self.Lambda /=2.0
                if self.Lambda < 0.05:
                    self.Lambda = 0.05
                self.random_counter = 0

            # reward
            r = 0
            for i in range(EPISODE):
                if self.result_list[i] == 0:
                    r += self.send_rate
                else:
                    r += -self.send_rate * EPISODE

            # current_state
            s_ = []
            for i in range(EPISODE):
                s_.append(self.send_rate/MAX_BANDWITH)
            for i in range(EPISODE):
                s_.append(self.result_list[i])
            for i in range(EPISODE):
                s_.append(self.send_rate/MAX_BANDWITH)
            s_array=np.array(s_)

            # choose action and explore
            a = actor.choose_action(s_array)
            if random.random() < self.Lambda:
                a = random.randint(0,2)
            print("action:",a)

            # Estimate the current available bandwidth 
            # and adjust the sending rate according to the output of tensorflow model
            if a == 0:
                self.send_rate += 10.0
            elif a == 1:
                self.send_rate += 0.0
            else:
                self.send_rate += -10.0
                if self.send_rate < 40.0:
                    self.send_rate = 40.0

            # last state
            s = np.array(self.last_state)

            td_error = critic.learn(s, r, s_array)

            # debug
            sums = 0
            for i in range(EPISODE,2*EPISODE):
                sums += self.last_state[i]
            print("last_state:",self.last_state[0] * MAX_BANDWITH,sums)
            sums = 0
            for i in range(EPISODE,2*EPISODE):
                sums += s_[i]
            print("present_state:",s_[0] * MAX_BANDWITH,sums)
            print("r",r)
            print("td_error",td_error)

            if self.last_state[0] == self.send_rate:
                a = 1
            elif self.last_state[0] > self.send_rate:
                a = 2
            else:
                a = 0

            actor.learn(s, a, td_error)

            self.last_state = s_
            self.result_list = []



# Your solution should include packet selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(BlockSelection, RL):

    def select_block(self, cur_time, block_queue):
        '''
        The alogrithm to select the block which will be sended in next.
        The following example is selecting block by the create time firstly, and radio of rest life time to deadline secondly.
        :param cur_time: float
        :param block_queue: the list of Block.You can get more detail about Block in objects/block.py
        :return: int
        '''
        def is_better(block):
            best_block_create_time = best_block.block_info["Create_time"]
            cur_block_create_time = block.block_info["Create_time"]
            # if block is miss ddl
            if (cur_time - cur_block_create_time) >= block.block_info["Deadline"]:
                return False
            if (cur_time - best_block_create_time) >= best_block.block_info["Deadline"]:
                return True
            if best_block_create_time != cur_block_create_time:
                return best_block_create_time > cur_block_create_time
            return (cur_time - best_block_create_time) * best_block.block_info["Deadline"] > \
                   (cur_time - cur_block_create_time) * block.block_info["Deadline"]

        best_block_idx = -1
        best_block= None
        for idx, item in enumerate(block_queue):
            if best_block is None or is_better(item) :
                best_block_idx = idx
                best_block = item

        return best_block_idx

    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm when sender need to send packet.
        """
        return super().on_packet_sent(cur_time)

    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        # estimate the bandwidth
        super().estimate_bandwidth(cur_time, event_info)
        
        # set cwnd or sending rate in sender
        return {
            "cwnd" : self.cwnd,
            "send_rate" : self.send_rate,
        }