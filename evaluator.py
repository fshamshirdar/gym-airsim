import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import random
import math

from memory import SequentialMemory
from util import *

class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path='', window_length=1, max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
        self.memory = SequentialMemory(limit=2*window_length, window_length=window_length)

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):
            deg = random.uniform(-math.pi, math.pi)
            r = 70
            x = r * math.cos(deg)
            y = r * math.sin(deg)
            env.set_goal([x, y])
            print ("new target: ", x, y)

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None
            self.memory.append(observation, None, None, None)

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(self.memory.get_recent_state_and_split(observation))

                observation, reward, done, info = env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1
                self.memory.append(observation, None, None, None)

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
