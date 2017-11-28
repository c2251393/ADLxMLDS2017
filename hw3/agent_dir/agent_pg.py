from agent_dir.agent import Agent

import numpy as np
from itertools import count
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


'''
When running pg:
    observation: np.array
        current RGB screen of game, shape: (210, 160, 3)
    reward: int
        if opponent wins, reward = +1 else -1
    done: bool
        whether reach the end of the episode?
'''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(16, 8, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(8, 4, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(3, 3)

        self.W = nn.Linear(140, 6)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.transpose(2, 3).transpose(1, 2)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = x.view(1, -1)
        x = self.W(x)
        return F.softmax(x)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        self.model = Model()
        self.opt = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.n_episode = args.episode
        self.gamma = args.gamma
        self.prv_state = cu(Variable(torch.zeros(210, 160, 3).float()))

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.prv_state = cu(Variable(torch.zeros(210, 160, 3).float()))


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        start = time.time()

        def finish_episode():
            R = 0
            rewards = []
            for r in self.model.rewards[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)
            print(R)
            rewards = torch.Tensor(rewards)
            for (act, r) in zip(self.model.saved_actions, rewards):
                act.reinforce(r)
            self.opt.zero_grad()
            autograd.backward(self.model.saved_actions, [None for _ in self.model.saved_actions])
            self.opt.step()
            print(time_since(start))

            del self.model.saved_actions[:]
            del self.model.rewards[:]

        self.model.train()
        if USE_CUDA:
            self.model.cuda()

        for episode in range(self.n_episode):
            print("Episode %d" % episode)
            self.prv_state = cu(Variable(torch.zeros(210, 160, 3).float()))
            state = self.env.reset()
            tot_reward = 0
            for t in range(10000):
                action = self.make_action(state, test=False)
                state, reward, done, info = self.env.step(action[0, 0])
                self.model.rewards.append(reward)
                tot_reward += reward
                # if t % 100 == 0:
                    # print(tot_reward)
                    # print(time_since(start))
                if done or abs(tot_reward) >= 3:
                    break

            print(tot_reward)
            print(time_since(start))
            finish_episode()
            torch.save(self.model.state_dict(), "agent_pg.pt")


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # return self.env.get_random_action()
        state = cu(Variable(torch.from_numpy(observation).float()))
        prob = self.model(state - self.prv_state)
        act = prob.multinomial()
        self.model.saved_actions.append(act)
        self.prv_state = state
        return act.data
