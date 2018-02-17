import logging

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box

from gym_airsim.envs.my_airsim_client import *
from AirSimClient import *

logger = logging.getLogger(__name__)

class AirSimEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
	#	self.state = (10, 10, 0, 0)
		# self.action_space = spaces.Discrete(3)
		self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
		self.observation_space = spaces.Box(low=0, high=255, shape=(1, 30, 100))
		self.state = np.zeros((1, 30, 100), dtype=np.uint8) 
		self._seed()

		self.client = myAirSimClient()

		# self.goal = [100.0, -200.0] # global xy coordinates
		self.goal = [0.0, 0.0] # global xy coordinates
		self.distance_before = None
		self.steps = 0
		self.no_episode = 0
		self.reward_sum = 0

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def compute_reward(self, goal):
		distance_goal = np.sqrt(np.power((self.goal[0]-goal.x_val),2) + np.power((self.goal[1]-goal.y_val),2))

		r = -1.
		if self.distance_before != None:
			r = r + (self.distance_before - distance_goal)
            
		return r, distance_goal

	def set_goal(self, goal):
		self.goal = goal

	def goal_direction(self, goal, pos):
		pitch, roll, yaw  = self.client.getPitchRollYaw()
		yaw = math.degrees(yaw) 
		pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
		pos_angle = math.degrees(pos_angle) % 360
		goal = math.radians(pos_angle - yaw)  
		return ((math.degrees(goal) - 180) % 360) - 180 

	def _step(self, action):
		self.steps += 1
		collided = self.client.take_continuous_action(action)
		position = self.client.getPosition()
		goal = self.goal_direction(self.goal, position) 

		if collided == True:
			done = True
			reward = -100.0
			distance = np.sqrt(np.power((self.goal[0]-position.x_val),2) + np.power((self.goal[1]-position.y_val),2))
		else: 
			done = False
			reward, distance = self.compute_reward(position)
        
		if distance < 3:
			done = True
			reward = 100.0

#		reward = np.clip(reward, -1., 1.)

		self.distance_before = distance
		self.reward_sum += reward
		if self.reward_sum < -100:
			done = True
        
		info = {"x_pos" : position.x_val, "y_pos" : position.y_val}
		# print ("current: ", info)
		self.state = self.client.getScreenDepthVis(goal)

		return self.state, reward, done, info

	def _reset(self):
		self.client._reset()
		self.steps = 0
		self.reward_sum = 0
		self.no_episode += 1
		self.distance_before = None

		position = self.client.getPosition()
		goal = self.goal_direction(self.goal, position)
		self.state = self.client.getScreenDepthVis(goal)
        
		return self.state

	def _render(self, mode='human', close=False):
		return
