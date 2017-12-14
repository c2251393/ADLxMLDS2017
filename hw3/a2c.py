import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 514)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--step_upd', type=int, default=20, help='step to update network (default: 4)')
parser.add_argument('--clip', action='store_true', help='clip grad magnitude for a2c')
parser.add_argument('--a2c', action='store_true', help='run a2c for breakout (agent_dqn)')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training (default: 0.0001)')
parser.add_argument('--episode', type=int, default=10000, help='episode count (default: 10000)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.W1 = nn.Linear(4, 128)
        self.Wa = nn.Linear(128, 2)
        self.Wv = nn.Linear(128, 1)

        self.log_probs = []
        self.entropies = []
        self.values = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.W1(x))
        return self.Wa(x), self.Wv(x)

    def clear_action(self):
        self.log_probs = []
        self.entropies = []
        self.values = []
        self.rewards = []


policy = Policy()
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    y, val = policy(Variable(state))

    prob = F.softmax(y)
    log_prob = F.log_softmax(y)
    entropy = -(log_prob * prob).sum(1)
    # print(prob)

    action = prob.multinomial().data
    log_prob = log_prob.gather(1, Variable(action))

    policy.values.append(val)
    policy.log_probs.append(log_prob)
    policy.entropies.append(entropy)

    return action


def update_network():
    R = 0
    for i in reversed(range(len(policy.rewards))):
        R = policy.rewards[i] + args.gamma * R
        policy.rewards[i] = R

    rewards = torch.Tensor(policy.rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    policy_loss = 0
    for (log_prob, r) in zip(policy.log_probs, rewards):
        policy_loss -= log_prob * r

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    policy.clear_action()

print(env.spec.reward_threshold)
print(env.action_space)

running_reward = 10
all_rewards = []

pts = []

if args.a2c:
    tot_reward = 0
    state = env.reset()
    i_episode = 0
    for epoch in count(1):
        if i_episode >= args.episode:
            break
        for step in range(args.step_upd):
            action = select_action(state)
            state, reward, done, _ = env.step(action[0,0])
            tot_reward += reward
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                i_episode += 1
                all_rewards.append(tot_reward)
                running_reward = running_reward * 0.99 + tot_reward * 0.01
                pts.append((i_episode, running_reward))
                if i_episode % args.log_interval == 0:
                    print('Episode {}\tLast length: {:5.0f}\tAverage length: {:.2f}'.format(
                        i_episode, tot_reward, running_reward))
                if running_reward > env.spec.reward_threshold:
                    print("Solved in %d episodes! Running reward is now %g and "
                          "the last episode runs to %d time steps!" % (i_episode, running_reward, tot_reward))
                tot_reward = 0
                state = env.reset()
                break
        if running_reward > env.spec.reward_threshold:
            break
        policy_loss = 0
        value_loss = 0

        R = torch.zeros(1, 1)
        if not done:
            fstate = torch.from_numpy(state).float().unsqueeze(0)
            _, val = policy(Variable(fstate))
            R = val.data

        policy.values.append((Variable(R)))
        R = (Variable(R))
        gae = (torch.zeros(1, 1))

        for i in reversed(range(len(policy.rewards))):
            R = args.gamma * R + policy.rewards[i]
            advantage = R - policy.values[i]
            # value_loss = value_loss + 0.5 * advantage.pow(2)
            value_loss = value_loss + advantage.pow(2)

            # delta_t = policy.rewards[i] + args.gamma * policy.values[i+1].data \
                        # - policy.values[i].data
            # gae = gae * args.gamma + delta_t
            gae = advantage.data

            policy_loss = policy_loss - policy.log_probs[i] * (Variable(gae)) - 0.01 * policy.entropies[i]

        target = (policy_loss + 0.5 * value_loss)

        optimizer.zero_grad()
        target.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm(policy.parameters(), 40)
        optimizer.step()

        policy.clear_action()

else:
    for i_episode in range(1, args.episode+1):
        state = env.reset()
        # print(state.shape)
        tot_reward = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action[0,0])
            tot_reward += reward
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        all_rewards.append(tot_reward)

        running_reward = running_reward * 0.99 + tot_reward * 0.01
        update_network()
        pts.append((i_episode, running_reward))
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved in %d episodes! Running reward is now %g and "
                  "the last episode runs to %d time steps!" % (i_episode, running_reward, tot_reward))
            break

print(' '.join(map(lambda p: '%d:%.f' % p, pts)))
exit()

fn = 'cartpole.pg.png'
if args.a2c:
    fn = 'cartpole.a2c.png'
x, y = zip(*pts)

fig = plt.figure()
plt.xlabel('episode')
plt.ylabel('reward')
plt.plot(x, y)
fig.savefig(fn)

