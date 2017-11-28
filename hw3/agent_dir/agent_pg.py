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
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(384, 256)

        self.W = nn.Linear(256, 6)
        self.apply(weights_init)
        self.W.weight.data = norm_col_init(
            self.W.weight.data, 0.01)
        self.W.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x):
        x, hidden = x
        x = x.unsqueeze(0)
        x = x.transpose(2, 3).transpose(1, 2)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        # print(hidden[0].size(), hidden[1].size())
        hx, cx = self.lstm(x, hidden)
        return self.W(hx), (hx, cx)

    def init_hidden(self):
        return cu(Variable(torch.zeros(1, 256))), cu(Variable(torch.zeros(1, 256)))


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)

        ##################
        # YOUR CODE HERE #
        ##################
        self.n_episode = args.episode
        self.gamma = args.gamma
        self.episode_len = args.episode_len

        self.model = Model()
        self.opt = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        self.state = cu(Variable(torch.zeros(210, 160, 3).float()))
        self.log_probs = []
        self.rewards = []

        self.hidden = self.model.init_hidden()

        if args.test_pg:
            #you can load your model here
            print('loading trained model :%s.' % args.model)
            state_dict = torch.load(args.model, map_location=lambda storage, location: storage)
            self.model.load_state_dict(state_dict)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.state = cu(Variable(torch.zeros(210, 160, 3).float()))
        self.hidden = self.model.init_hidden()


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
            for i in reversed(range(len(self.rewards))):
                R = self.rewards[i] + self.gamma * R
                self.rewards[i] = R
            rewards = torch.Tensor(self.rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

            policy_loss = 0.0
            for (log_prob, r) in zip(self.log_probs, rewards):
                policy_loss = policy_loss - log_prob * r

            print("Policy loss: ", policy_loss.data[0, 0])

            self.opt.zero_grad()
            policy_loss = cu(policy_loss)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)
            self.opt.step()

            print(time_since(start))
            self.clear_action()

        self.model.train()
        if USE_CUDA:
            self.model.cuda()

        for episode in range(self.n_episode):
            print("Episode %d" % episode)
            self.state = cu(Variable(torch.zeros(210, 160, 3).float()))
            self.hidden = self.model.init_hidden()
            state = self.env.reset()

            tot_reward = 0
            a, b = 0, 0
            elen = 0
            for t in range(self.episode_len):
                action = self.make_action(state, test=False)
                state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)
                if reward > 0:
                    a += 1
                if reward < 0:
                    b += 1
                tot_reward += reward
                if done or a >= 3 or b >= 3:
                    elen = t+1
                    break

            print(tot_reward, a, b, elen)
            print(time_since(start))
            finish_episode()
            torch.save(self.model.state_dict(), "agent_pg.pt")


    def make_action(self, state, test=True):
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
        state = cu(Variable(torch.from_numpy(state).float()))
        d_state = state - self.state

        y, self.hidden = self.model((d_state, self.hidden))
        prob = F.softmax(y)
        log_prob = F.log_softmax(y)

        act = prob.multinomial().data
        log_prob = log_prob.gather(1, cu(Variable(act)))

        self.log_probs.append(log_prob)
        self.state = state
        return act[0, 0]

    def clear_action(self):
        del self.log_probs[:]
        del self.rewards[:]
