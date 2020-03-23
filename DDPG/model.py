'''
-> DDPG is an off-policy learning algorithm

-> Need a replay buffer class

-> Need a class for a target Q network (function of s, a)

-> we will use batch norm

-> the policy is deterministic, how to handle explore exploit? Deterministic policy means outputs the actual action instead of a prob. Will need a way to boud actions to env limit

-> WE use a stochastic policy to find the deterministic policy

-> At each timestep, the actor and critic are updated by sampling a minibatch uniformly from the buffer.

-> Replay buffer can be large, allowing the algo to learn from a large set of uncorrelated transitions. Don't wanna sample subsequent steps, since they might be highly correlated

-> We have 2 actor and 2 critic networks, a target for each.
Updates are soft, according to theta' = tau*theta + (1-tau)*theta', with tau << 1 to improve stability of training/gradual convergence

-> Main advantage of off-policy learning: exploration and learning can be done independently
   mu' = mu + Noraml_dist

-> target actor is just evaluation actor plus some noise

-> they used Ornstein Uhlenbeck(noise process that models motion of Brownian particles), temporal correlation
    (Will need a class for this noise)
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.intiializers import random_uniform

print(tf.test.is_gpu_available())

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[index]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.chkpt_dir = chkpt_dir

        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).\
                    apply_gradients(zip(self.actor_gradients, self.params))
        
    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, self.*input_dims],
                                        name='inputs')
            self.action_gradient = tf.placeholder(tf.float32, shape=[None, self.n_actions])

            f1 = 1 / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims,
                            kernel_initializer=random_uniform(-f1, f1),
                            bias_initializer=random_uniform(-f1, f1))

            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                            kernel_initializer=random_uniform(-f2, f2),
                            bias_initializer=random_uniform(-f2, f2))

            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            f3 = 0.003
            mu = tf.layers.dense(layer2_activation, units=self.n_actions,
                                activation='tanh',
                                kernel_initializer=random_uniform(-f3, f3),
                                bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.actor_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                        feed_dict={self.inputs: inputs,
                                   self.actor_gradients: gradients})

    def save_checkpoint(self):
        print('....saving checkpoint....')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('.... loading checkpoint ....')
        self.saver.restore(self.sess, self.checkpoint_file)


# def Critic(object):
    # def __init__(self, lr)