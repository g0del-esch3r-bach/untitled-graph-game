'''
Example of using Q-Learning or StableBaseline3 to train our custom environment.
'''
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import stable_baselines3 # type: ignore
from stable_baselines3 import DQN # type: ignore
import os
import v0_conj1349_env # Even though we don't use this class here, we should include it here so that it registers the WarehouseRobot environment.

# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the DQN (DQN) algorithm.
def train_sb3():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('conj1349-v0')

    # Use DQN algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = DQN('MultiInputPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
   
    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(f"{model_dir}/DQN_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(render=True):

    env = gym.make('conj1349-v0', render_mode='human' if render else None)

    # Load model
    model = DQN.load('models/DQN_1000000', env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            break

if __name__ == '__main__':
    #train_sb3()
    test_sb3()