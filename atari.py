# Implementation of the paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
# Playing Atari with Deep Reinforcement Learning

# !pip install gym-retro
import retro
import tensorflow as tf
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt

from skimage import transform
from skimage.color import rgb2gray
from collections import deque

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
warnings.filterwarnings('ignore') # to ignore warnings while training because of skiimage

env = retro.make(game='SpaceInvaders-Atari2600')
print("Size of our frame is: ", env.observation_space)
print("Action size is: ", env.action_space.n)
# Hot encoded version of our actions
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

def preprocess_frame(frame):
    gray = rgb2gray(frame)

    # Remove the part of frame below the player
    cropped_frame = gray[8:-12, 4:-12]

    # Normalize pixel values
    normalized_frame = cropped_frame/255.0

    # Resize
    preprocess_frame = transform.resize(cropped_frame, [110, 84])

    # 110x84x1 frame
    return preprocess_frame 

# skip 4 frames
stack_size = 4
 # Init deque with zero-images one array for each image
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen = 4)

def stacked_frames(stacked_frames, state, is_new_episode):
     # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        #  Clear our stacked_frames
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen = 4)

        # Because we're in a new episode, copy the same frame 4 times
        for _ in range(4):
            print("HA")
            stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dim specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

# MODEL HYPERPARAMETERS
state_size = [110, 84, 4]
action_size = env.action_space.n # 8 actions
learning_rate = 0.00025

# TRAINING HYPERPARAMS
total_episodes = 50
max_steps = 50000
batch_size = 64

# Exploration params for epsilon greedy strategy
explore_start = 1.0
explore_stop = 0.01   # min exploration probability
decay_rate = 0.00001

# Q Learning hyperparams
gamma = 0.9

# Memory Hyperparams
pretrain_length = batch_size     # No. of experiences stored in the memory when initialized for the first time
memory_size = 1000000       # No. of experiences the memory can keep

# Preprocessing hyperparams
stack_size = 4

# Modify this to false if you just want to see the trained agent
training = False
episode_render = False

# DQN Model
# - Stack of 4 frames as input
# - passes through 3 convnets
# - flattened
# - passes through 2 FC layers
# - Outputs a Q value for each actions

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            #  *state_size equivalent to [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                          filters = 32,
                                          kernel_size = [8,8],
                                          strides = [4,4],
                                          padding = "VALID",
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = "conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """

            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                          filters = 64,
                                          kernel_size = [4,4],
                                          strides = [2,2],
                                          padding = "VALID",
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = "conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """

            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                          filters = 64,
                                          kernel_size = [3,3],
                                          strides = [2,2],
                                          padding = "VALID",
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = "conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs = self.flatten,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      units=self.action_size,
                                      activation=None)
            
            # Q is our predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            # Loss sum(target_Q - pred_Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
# tf.reset_default_graph()
# Instantiate DQN
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Experience Replay
"""
Here we create the Memory object that creates a deque.
"""
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        return [self.buffer[i] for i in index]

# Instantiate memory
memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    #  Get next state, rewards done by taking a random action
    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)

    # env.render()

    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    # If the episode is finished (we're dead 3x)
    if done:
        # episode finished
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # STart new episode
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our new state is now next_state
        state = next_state

# Setup Tensorboard
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

# Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

# Training the agent
# Init weights, init env, init decay rate that we use to reduce epsilon
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # EPSILON GREEDY
    explore_exploit_tradeoff = np.random.rand()
    explore_prob = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_prob > explore_exploit_tradeoff):
        # Make random action(explore)
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]

    else:
        # Get action from Q-network (exploit)
        # estimate Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        # Take biggest Q value (best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_prob

# Saver to save model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Init vars
        sess.run(tf.global_variables_initializer())
        decay_step = 0

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []

            state = env.reset()

            state, stacked_frames = stacl_frames(stacked_frames, state, True)

            while(step < max_steps):
                step += 1

                # Increase decay_step
                decay_step += 1

                # Predict action to take and take it
                action, explore_prob = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                # Perform action and get next_state, reward and done
                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()

                # Add reward to total reward
                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((110, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max steps to end episode
                    step = max_steps

                    # Get total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                            'Total reward: {}'.format(total_reward),
                            'Explore P: {:.4f}'.format(explore_prob),
                            'Training Loss: {:.4f}'.format(loss))
                    
                    rewards_list.append((episode, total_reward))

                    # Store transition <st, at, rt+1, st+1> in memory D
                    memory.add((state, action, reward, next_state, done))

                else:
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add exp to mem
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our currState
                    state = next_state

                # LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q vals for next_state
                target_Qs = sess.run(DQNetwork.output, feed_dict = {
                    DQNetwork.inputs_: next_states_mb
                })

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal: target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                    
                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict = {
                                        DQNetwork.inputs_: states_mb,
                                        DQNetwork.target_Q: targets_mb,
                                        DQNetwork.actions_: actions_mb
                                    })

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict = {
                    DQNetwork.inputs_: states_mb,
                    DQNetwork.target_Q: targets_mb,
                    DQNetwork.actions_: actions_mb
                })
                writer.add_summary(summary, episode)
                writer.flush()

                # Save model every 5 episodes
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")

with tf.Session() as sess:
    total_test_rewards = []
    # Load model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(1):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        print("***********************************")
        print("EPISODE ", episode)

        while True:
            # Reshape state
            state = state.reshape((1, *state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})

            # Take the biggest Q value
            choice = np.argmax(Qs)
            action = possible_actions[choice]

            next_state, reward, done, _ = env.step(action)
            env.render()
            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    env.close()