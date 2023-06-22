# AI-plays-atari-games
## Introduction

Welcome to the AI Playing Atari Classics project! This repository showcases an implementation of an AI system that learns to play classic Atari games using reinforcement learning techniques, specifically **Space invaders** and **Pong**. The goal of this project is to develop an intelligent agent capable of achieving high scores and demonstrating proficient gameplay in a variety of Atari classics.

By leveraging the power of deep learning and reinforcement learning algorithms, the AI agent autonomously learns to navigate and interact within the game environment. Through iterations of training and self-improvement, the agent gradually develops strategies and policies to maximize its rewards and achieve optimal performance.

Feel free to explore the code, experiment with different settings, and observe the agent's learning process. The repository provides tools for training the AI agent, monitoring its progress, and evaluating its gameplay. Additionally, contributions from the community are welcomed to further enhance and expand the project.

## Atari Games Overview

Atari games hold a significant place in the history of video games, having pioneered the industry and captivated audiences during the early years of gaming. Atari, founded in 1972, played a vital role in shaping the landscape of interactive entertainment.

Atari released numerous iconic games across various platforms, including arcade machines, home consoles, and computers. These games became cultural phenomena, captivating players with their simplicity, addictive gameplay, and innovative concepts.

Atari classics span a wide range of genres, from action-packed space shooters like Asteroids and Centipede to sports simulations like Pong and racing games like Indy 500. Each game presented unique challenges, required different strategies, and provided players with hours of entertainment.

While the graphics and sound of Atari games may appear primitive compared to modern titles, their enduring appeal lies in their gameplay mechanics and timeless design principles. They serve as a testament to the creativity and ingenuity of early game developers, and many of these classics continue to be enjoyed and referenced to this day.

Through this AI project, we aim to harness the capabilities of modern machine learning techniques to teach an AI agent to play and excel in these Atari classics. By doing so, we pay homage to the legacy of Atari games while showcasing the advancements in AI and reinforcement learning.

So, let's dive into the world of Atari games and witness the evolution of an AI agent that conquers these beloved classics!

![atari](https://github.com/smuffyy/AI-plays-atari-games/assets/96169914/59d568e8-ea95-48db-8612-a7f56f27676b)


## Requirements

1. Python 3.x

2. Gym and Gym Atari: Install the required dependencies by running pip install gym[atari] autorom[accept-rom-license]

3. TensorFlow and TF-Agents: Install the required dependencies by running pip install tf-agents

## Getting Started
1. Clone the repository: git clone https://github.com/smuffyy/AI-plays-atari-games.git
2. Install the dependencies mentioned in the Requirements section.
3. Run the code: main.py

## Code

we begin by importing the required python dependencies

## Importing dependencies
```
import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import random
import tensorflow as tf
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
import matplotlib as mpl
import matplotlib.animation as animation
import PIL
import os
```
## Loading the atari environment
To load the Atari environment, we use the suite_atari.load() function from the tf_agents.environments module to load the Atari environment. 
The environment_name variable should contain the name of the Atari game you want to load, such as 'SpaceInvaders-v4'. 
The max_episode_steps variable determines the maximum number of steps per episode.
Additionally, the gym_env_wrappers argument specifies the preprocessing wrappers to be applied to the environment, including AtariPreprocessing and FrameStack4.

```
max_episode_steps = 50000 
environment_name = "SpaceInvaders-v4"
env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])
	
tf_env = TFPyEnvironment(env)
print(tf_env.action_spec())
print(tf_env.observation_spec())

```
## Set up initial game state
sets up the initial state of the game by seeding the environment, resetting it, and taking a few predetermined actions (FIRE and LEFT)
```
env.seed(42)
env.reset()
time_step = env.step(np.array(1))  
for _ in range(4):
    time_step = env.step(np.array(3))  
```
# Define a function to plot and display game observations
define the plot_observation() function, which takes an observation from the game and plots it. The code then uses this function to plot and display the initial observation.
```
def plot_observation(obs):
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.figure(figsize=(6, 6))

# Plot and display the initial observation
plot_observation(time_step.observation)
plt.show()
```
![initialgame](https://github.com/smuffyy/AI-plays-Atari-classics---reinforcement-learning/assets/136073275/b5c0d0a3-1b20-4a6c-af93-567ec4ba6b66)
# Model Architecture 

The AI agent in this project utilizes a deep Q-network (DQN) model to learn and make decisions while playing Atari games. The model architecture consists of the following components:
Preprocessing Layer:

A Lambda layer that normalizes the observations by scaling them to the range [0, 1].
Convolutional Layers:

Several convolutional layers are employed to extract meaningful features from the input observations.
The specific configuration of the convolutional layers is as follows:
Convolutional layer 1:
Number of filters: 32
Kernel size: (8, 8)
Stride: 4
Convolutional layer 2:
Number of filters: 64
Kernel size: (4, 4)
Stride: 2
Convolutional layer 3:
Number of filters: 64
Kernel size: (3, 3)
Stride: 1
Fully Connected Layers:

The output of the convolutional layers is flattened and fed into fully connected layers for further processing.
The specific configuration of the fully connected layers is as follows:
Fully connected layer 1:
Number of units: 512
Output Layer:

The final fully connected layer produces Q-values for each possible action, indicating the estimated expected rewards for taking those actions.
By combining convolutional layers for feature extraction and fully connected layers for decision-making, the DQN model learns to approximate the optimal action-value function and make informed decisions while playing Atari games

## Define the model architecture
define the model architecture for the Q-network.
It uses a preprocessing layer to normalize the observations, followed by convolutional layers and fully connected layers. 
The Q-network takes the observation and action specifications from the environment.
```
preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params = [512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params
)
```
## Set up optimizer and training parameters
set up the optimizer, training parameters, and epsilon-greedy exploration schedule. .
```
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=2.5e-4,
    rho=0.95,
    momentum=0.0,
    epsilon=0.000001,
    centered=True
)
train_step = tf.Variable(0)
update_period = 4  # Run a training step every 4 collect steps

epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=250000000 // update_period,
    end_learning_rate=0.01
)
```
## Initialize the DQN agent
initialize the DQN agent using the Q-network, optimizer, and other settings
```
agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000,
    td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
    gamma=0.99,
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step)
)
agent.initialize()
```
## Set up replay buffer for experience replay
sets up the replay buffer, which is used for experience replay during training. 
It creates a TFUniformReplayBuffer with the data specification from the agent's collect data and specifies the batch size and maximum buffer length.
```
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000
)
replay_buffer_observer = replay_buffer.add_batch
```

## Define training metrics
define a list of training metrics to track during training, such as the number of episodes, environment steps, average return, and average episode length.  
also set up logging to display the training metrics during training.
```
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

## Set up logging for training metrics
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)
```

## Set up the collect driver for data collection
set up the collect driver, which is responsible for collecting data using the agent's collect policy. 
It collects data for a specified number of steps and observes the data using the replay buffer and training metrics
```
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period
)
```

## Define a custom observer for progress display during training
This section defines a custom observer, ShowProgress, which tracks the progress of data collection during the initial phase. 
It counts the number of collected trajectories and displays the progress every 100 trajectories. The initial data collection is performed using a random policy
```
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print(f"\r{self.counter}/{self.total}", end="")

### Set up initial data collection with a random policy
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000
)
final_time_step, final_policy_state = init_driver.run()
```

## Create a dataset from the replay buffer for training
create a dataset from the replay buffer for training the agent. 
It specifies the batch size and the number of steps to sample from each trajectory. 
The dataset is also parallelized for improved performance
```
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3
).prefetch(3)
```

## Optimize training for better performance
```
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)
```
## Define a function to train the agent
defines the train_agent() function, which performs the training loop for the agent. It collects data, trains the agent using the collected data, and logs the training loss and metrics. 
The agent is trained for a specified number of iterations
```
def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)

### Train the agent for a specified number of iterations
train_agent(n_iterations=30000)
```

## Collect frames for creating an animation
```
frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

### Run the agent to collect frames for visualization
watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, ShowProgress(30000)],
    num_steps=30000
)
final_time_step, final_policy_state = watch_driver.run()

### Create an animation from the collected frames
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128

mpl.rc('animation', html='jshtml')

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval
    )
    plt.close()
    return anim

### Plot and display the animation
plot_animation(frames)
```
![myAgentPlaysSpaceInvaders-Best](https://github.com/smuffyy/AI-plays-atari-games/assets/96169914/97b405fc-8e91-47cc-a020-ac2782afbd8e)

## Save the trained model
```
saved_model_path = "./saved_model"
tf.saved_model.save(agent.policy, saved_model_path)
```

## define a function to take checkppoints and compute the average
Define a function compute_long_run_average_return() that takes a list of checkpoint steps as input. It iterates over each checkpoint step and performs data collection using the collect_driver. 
It calculates the average discounted reward over multiple trajectories and stores the results in checkpoint_returns.
Then, the code sets up the checkpoint steps and calls the compute_long_run_average_return() function to obtain the long-run average returns at those checkpoints. 
Finally, it plots the long-run average discounted reward at each checkpoint step using matplotlib.
The resulting plot shows how the long-run average discounted reward changes over different checkpoint steps, providing insights into the agent's performance throughout the training process.
```
def compute_long_run_average_return(checkpoint_steps):
    checkpoint_returns = []
    iterator = iter(dataset)  
    for step in checkpoint_steps:
        time_step = None
        policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
        returns = []
        for _ in range(step):
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            trajectories, _ = next(iterator)
            returns.append(trajectories.reward.numpy().mean())
        checkpoint_returns.append(sum(returns) / len(returns))
    return checkpoint_returns

checkpoint_steps = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
checkpoint_returns = compute_long_run_average_return(checkpoint_steps)

### Plot the long-run average discounted reward
plt.figure(figsize=(8, 6))
plt.plot(checkpoint_steps, checkpoint_returns, marker='o')
plt.xlabel('Checkpoint Steps')
plt.ylabel('Long-Run Average Discounted Reward')
plt.title('Long-Run Average Discounted Reward at Checkpoints')
plt.tight_layout()
plt.show()
```
![trainingCurve-SpaceInvaders](https://github.com/smuffyy/AI-plays-atari-games/assets/96169914/7f4c95fc-7bde-4b87-9559-f1dd39277c0b)

# License
The code in this repository is licensed under the MIT License

# Authors 
This article and code has been co-authored and co-developed by
Harinandan Rajkishor and Jesika Davidson

LinkedIn: https://www.linkedin.com/in/harinandan-rajkishor/

Email: harinandanrajkishor@gmail.com
