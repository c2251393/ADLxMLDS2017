from agent_dir.agent import Agent

from utils import *

'''
When running dqn:
    observation: np.array
        stack 4 last preprocessed frames, shape: (84, 84, 4)
    reward: int
        wrapper clips the reward to {-1, 0, 1} by its sign
        we don't clip the reward when testing
    done: bool
        whether reach the end of the episode?
'''

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 500


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.W1 = nn.Linear(2592, 256)
        self.W2 = nn.Linear(256, 4)
        # self.apply(weights_init)
        # self.W1.weight.data = norm_col_init(
            # self.W1.weight.data, 0.01)
        # self.W1.bias.data.fill_(0)
        # self.W2.weight.data = norm_col_init(
            # self.W2.weight.data, 0.01)
        # self.W2.bias.data.fill_(0)

    def forward(self, x):
        # (B, 84, 84, 4)
        x = x.transpose(2, 3).transpose(1, 2)
        # (B, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = F.relu(self.W1(x))
        x = self.W2(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.n_episode = args.episode
        self.n_warm = args.warm
        self.gamma = args.gamma
        self.episode_len = args.episode_len
        self.batch_size = args.batch_size
        self.step_copy = args.step_copy

        self.model = Model()
        # self.target_model = Model()
        self.opt = optim.RMSprop(self.model.parameters(), lr=args.learning_rate, weight_decay=0.99)
        self.memory = ReplayMemory(args.buffer_size)

        self.state = cu(Variable(torch.zeros(84, 84, 4).float()))
        self.steps_done = 0

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            state_dict = torch.load(args.model, map_location=lambda storage, location: storage)
            self.model.load_state_dict(state_dict)

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
        self.state = torch.zeros(84, 84, 4).float()


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        start = time.time()
        self.model.train()
        # self.target_model.eval()
        if USE_CUDA:
            self.model.cuda()
            # self.target_model.cuda()

        def optimize_model():
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            # print(batch)
            non_final_mask = cu(torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                  batch.next_state))))
            non_final_next_states = cu(Variable(
                torch.stack([torch.from_numpy(s).float()
                             for s in batch.next_state if s is not None]), volatile=True))
            state_batch = cu(Variable(torch.stack([torch.from_numpy(s).float() for s in batch.state])))
            # (batch, 84, 84, 4)
            action_batch = cu(Variable(torch.stack([torch.LongTensor([a]) for a in batch.action])))
            # (batch, 1)
            reward_batch = cu(Variable(torch.stack([torch.FloatTensor([r]) for r in batch.reward])))
            # (batch, 1)
            state_action_values = self.model(state_batch).gather(1, action_batch)
            # (batch, 1)
            next_state_values = cu(Variable(torch.zeros(self.batch_size, 1)))
            next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
            next_state_values.volatile = False

            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            loss = cu(F.smooth_l1_loss(state_action_values, expected_state_action_values))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # if self.steps_done % self.step_copy == 0:
                # self.target_model.load_state_dict(self.model.state_dict())
            return loss.data[0]

        all_loss = []
        running_reward = None

        for episode in range(self.n_episode):
            self.init_game_setting()
            state = self.env.reset()

            tot_reward = 0

            for t in range(self.episode_len):
                action = self.make_action(state, test=False)
                next_state, reward, done, info = self.env.step(action)
                tot_reward += reward
                if done:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                state = next_state

                loss = optimize_model()
                # print(loss)
                all_loss.append(loss)

                if done:
                    break
            if running_reward is None:
                running_reward = tot_reward
            else:
                running_reward = 0.99 * running_reward + 0.01 * tot_reward

            if episode % 100 == 0:
                print("Episode %d" % episode)
                print(time_since(start), running_reward, self.steps_done)
            torch.save(self.model.state_dict(), "agent_dqn.pt")


    def make_action(self, state, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        sample = random.random()
        if self.steps_done > 1e6:
            eps_threshold = 0.1
        else:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            (1 - self.steps_done / 1e6)
        self.steps_done += 1
        # print(eps_threshold)
        if sample > eps_threshold:
            state = torch.from_numpy(state).float()
            y = self.model(cu(Variable(state.unsqueeze(0), volatile=True)))
            act = y.max(1)[1]
            return act.data[0]
        else:
            return self.env.get_random_action()
