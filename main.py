import argparse
import os
import gym
import gym_airsim
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
# LEARNING_STARTS = 50000
LEARNING_STARTS = 0
LEARNING_FREQ = 4
# FRAME_HISTORY_LEN = 4
FRAME_HISTORY_LEN = 1
# TARGER_UPDATE_FREQ = 10000
TARGET_UPDATE_FREQ = 100
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

parser = argparse.ArgumentParser(description='AirSim Navigation Reinforcement Learning')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
#parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
#                    choices=model_names,
#                    help='model architecture: ' +
#                        ' | '.join(model_names) +
#                        ' (default: resnet18)')
#parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                    help='number of data loading workers (default: 4)')
#parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                    help='number of total epochs to run')
#parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                    help='manual epoch number (useful on restarts)')
#parser.add_argument('-b', '--batch-size', default=256, type=int,
#                    metavar='N', help='mini-batch size (default: 256)')
#parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                    metavar='LR', help='initial learning rate')
#parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                    help='momentum')
#parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                    metavar='W', help='weight decay (default: 1e-4)')
#parser.add_argument('--print-freq', '-p', default=10, type=int,
#                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
#parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                    help='evaluate model on validation set')
#parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                    help='use pre-trained model')
#parser.add_argument('--world-size', default=1, type=int,
#                    help='number of distributed processes')
#parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                    help='url used to set up distributed training')
#parser.add_argument('--dist-backend', default='gloo', type=str,
#                    help='distributed backend')

def main(env):
	global args
	args = parser.parse_args()

	optimizer_spec = OptimizerSpec(
		constructor=optim.RMSprop,
		kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
	)

	exploration_schedule = LinearSchedule(1000000, 0.1)

	dqn_learing(
		env=env,
		q_func=DQN,
		checkpoint_path=args.checkpoint,
		optimizer_spec=optimizer_spec,
		exploration=exploration_schedule,
		stopping_criterion=None,
		replay_buffer_size=REPLAY_BUFFER_SIZE,
		batch_size=BATCH_SIZE,
		gamma=GAMMA,
		learning_starts=LEARNING_STARTS,
		learning_freq=LEARNING_FREQ,
		frame_history_len=FRAME_HISTORY_LEN,
		target_update_freq=TARGET_UPDATE_FREQ,
	   )

if __name__ == '__main__':
	env = gym.make('AirSim-v0')
	env.seed(543)
	# torch.manual_seed(543)

	main(env)
