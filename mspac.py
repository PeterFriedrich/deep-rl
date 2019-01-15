"""
Deep Reinforcement Learning to play Ms-Pacman
"""

"""
Notes:

Get more instances running somehow?
how quickly am i training? get 
need to do more logging, efficiently 
add another layer  sometime?

need proper data logging, save average score every 

"""

# Common imports
import numpy as np
import os
import sys
import datetime

# main libraries
import gym 
import tensorflow as tf 

# plot things 
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
# Animate stuff 
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,
    
def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)

# Environment 
env = gym.make("MsPacman-v0")
obs = env.reset()

# Preprocess the environment 
mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)

img = preprocess_observation(obs)

# Time information
timelog = []
timelog.append(datetime.datetime.now())

""" 
---------------------------------------------------------------------------------------------------------------------------------------
Building the DQN System
---------------------------------------------------------------------------------------------------------------------------------------
"""


# setting up inputs for convolutional network 
input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.variance_scaling_initializer()

def q_network(X_state, name):
    prev_layer = X_state / 128.0 # scale pixel intensities to the [-1.0, 1.0] range.
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        #optional extra hidden 
        hidden_2 = tf.layers.dense(hidden, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        #hidden_3 = tf.layers.dense(hidden_2, n_hidden,
        #                         activation=hidden_activation,
        #                         kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden_2, n_outputs, 
                                  kernel_initializer=initializer)
        #outputs = tf.layers.dense(hidden, n_outputs,
        #                          kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name
    
X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keepdims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()

# replay memory instead of dequeue for random access speed 
class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0
        
    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
    
    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size) # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

        
replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)
saver = tf.train.Saver()

def sample_memories(batch_size):
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)
    
eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

#n_steps = 60000      
n_steps = 1200000
#n_steps = 2000000
""" 2mil step """        
#n_steps = 4000000  # total number of training steps 
#n_steps = 6000000
#n_steps = 
""" 4mil Ended with score of 1660? mean max-q around 250?"""
#n_steps = 8000000
""" 8mil ended with score of 1530, mean max-q like 210 to 150?"""
#n_steps = 12000000
""" 12mil, got even worse, 900"""
#n_steps = 50000 # test 
"""
800k steps, mean max q 85, reward 1110
"""

training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
save_steps = 1000  # save the model every 1,000 training steps
copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.99
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
checkpoint_path = "./checkpoints/my_dqn2.ckpt"
high_q_checkpoint = "./checkpoints/high_dqn2.ckpt"
done = True # env needs to be reset
#STORE_PATH = 'C:\\Users\\Student\\gym\\\data\\TensorBoard'

# track progress 
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0
total_reward = 0.0
high_q = 0.0

# pyplot stuff
q_val_list = []
iter_list = []
reward_list = []
plot_flag = False


# Main training loop 
with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
    #writer = tf.summary.FileWriter(STORE_PATH, sess.graph)
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        print("\rIteration {}  Training step {}/{} ({:.1f})%  Loss {:5f}  Mean Max-Q {:5f}  High_q {:4f} ".format(
            iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q, high_q), end="")
        if done: # game over, start again
            obs = env.reset()
            for skip in range(skip_start): # skip the start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Compute statistics for tracking progress (not shown in the book)
        total_reward += reward 
        total_max_q += q_values.max()
        game_length += 1
        
        # pseudo-epoch set up 
        if iteration % 50000 == 0:
            plot_flag = True 
            
        # Plot and average stuff
        if done:
            mean_max_q = total_max_q / game_length
            
            if mean_max_q > high_q:
                high_q = mean_max_q
                saver.save(sess, high_q_checkpoint)
            
            # append plot point each epoch 
            if plot_flag:
                q_val_list.append(mean_max_q)
                iter_list.append(iteration)
                reward_list.append(total_reward)
                plot_flag = False 
            
            total_max_q = 0.0
            total_reward = 0.0
            game_length = 0
 
            
            
        if iteration < training_start or iteration % training_interval != 0:
            continue # only train after warmup period and at regular intervals
        
        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})

        # Regularly copy the online DQN to the target DQN
        if step % copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)
            
            
# Test the agent 
frames = []
n_max_steps = 10000

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    obs = env.reset()
    for step in range(n_max_steps):
        state = preprocess_observation(obs)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = np.argmax(q_values)

        # Online DQN plays
        obs, reward, done, info = env.step(action)

        img = env.render(mode="rgb_array")
        frames.append(img)

        if done:
            break

# Print times
timelog.append(datetime.datetime.now())
for i in timelog:
    print(i.strftime("%Y-%m-%d %H:%M"))

# plot and save
plt.plot(iter_list, q_val_list)
plt.xlabel('Iterations')
plt.ylabel('Q-Values')
save_fig('2 layer Q-Values after ' + str(iteration) + ' Iterations')

plt.clf()
plt.plot(iter_list, reward_list)
plt.xlabel('Iterations')
plt.ylabel('Rewards')
save_fig('2 layer Rewards after ' + str(iteration) + 'Iterations')

            
while True:                  
    video = plot_animation(frames)
    plt.show()
    
