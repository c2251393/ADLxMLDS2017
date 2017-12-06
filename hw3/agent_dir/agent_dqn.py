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
EPS_END = 0.05
EPS_DECAY = 1000000


class Model(nn.Module):
    def __init__(self, duel=False):
        super(Model, self).__init__()
        self.duel = duel
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.W1 = nn.Linear(3136, 512)
        self.W2 = nn.Linear(512, 4)
        if self.duel:
            self.Wv = nn.Linear(512, 1)
        self.apply(weights_init)
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
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        x = F.leaky_relu(self.conv3(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = F.relu(self.W1(x))
        if self.duel:
            a = F.relu(self.W2(x))
            v = F.relu(self.Wv(x))
            am = a.mean(1).unsqueeze(1)
            x = a + (v - am)
        else:
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
        self.print_every = args.print_every
        self.n_episode = args.episode
        self.gamma = args.gamma
        self.episode_len = args.episode_len
        self.batch_size = args.batch_size
        self.step_copy = args.step_copy
        self.step_learn = args.step_learn
        self.step_upd = args.step_upd
        self.max_step = args.step_train

        self.ddqn = args.ddqn
        self.duel = args.duel
        self.clip = args.clip

        self.model = Model(self.duel)
        self.target_model = Model()
        self.opt = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)
        self.memory = ReplayMemory(args.buffer_size)

        self.steps_done = 0
        self.act_by_model = 0

        self.model_fn = args.model
        if self.model_fn == '':
            self.model_fn = 'agent_dqn.pt'

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            state_dict = torch.load(args.model, map_location=lambda storage, location: storage)
            self.model.load_state_dict(state_dict)
            if USE_CUDA:
                self.model.cuda()
            self.test_step = 0

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
        self.test_step = 0


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        start = time.time()
        self.model.train()
        self.target_model.eval()
        if USE_CUDA:
            self.model.cuda()
            self.target_model.cuda()

        def optimize_model():
            if len(self.memory) < self.batch_size:
                return
            # print("hi")
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            # print(batch)
            non_final_mask = cu(torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                  batch.next_state))))
            non_final_next_states = cu(Variable(
                torch.stack([torch.from_numpy(s).float()
                             for s in batch.next_state if s is not None]), volatile=True))
            state_batch = np.array(batch.state)
            state_batch = cu(Variable(torch.from_numpy(state_batch).float()))
            # (batch, 84, 84, 4)
            action_batch = np.array(batch.action)
            action_batch = cu(Variable(torch.from_numpy(action_batch).long().unsqueeze(1)))
            # (batch, 1)
            reward_batch = np.array(batch.reward)
            reward_batch = cu(Variable(torch.from_numpy(reward_batch).float().unsqueeze(1)))
            # (batch, 1)
            state_action_values = self.model(state_batch).gather(1, action_batch)
            # (batch, 1)
            next_state_values = cu(Variable(torch.zeros(self.batch_size, 1)))
            if self.ddqn:
                next_state_actions = self.model(non_final_next_states).max(1)[1]
                next_state_values[non_final_mask] = self.target_model(non_final_next_states)[next_state_actions]
            else:
                next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]

            next_state_values.volatile = False

            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            loss = F.mse_loss(state_action_values, expected_state_action_values)
            # print(loss)
            self.opt.zero_grad()
            loss.backward()
            if self.clip:
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.opt.step()

            return loss.data[0]

        running_reward = None

        for episode in range(self.n_episode):
            if self.steps_done > self.max_step:
                break
            self.init_game_setting()
            state = self.env.reset()

            tot_reward = 0

            for t in count(1):
                if self.steps_done > self.max_step:
                    break
                action = self.make_action(state, test=False)
                next_state, reward, done, info = self.env.step(action)
                tot_reward += reward
                if done:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                state = next_state

                if self.steps_done % self.step_upd == 0\
                   and self.steps_done > self.step_learn:
                    loss = optimize_model()

                if self.steps_done % self.step_copy == 0:
                    self.target_model = copy.deepcopy(self.model)
                    self.target_model.eval()

                if done:
                    break

            if running_reward is None:
                running_reward = tot_reward
            else:
                running_reward = 0.99 * running_reward + 0.01 * tot_reward

            if episode % self.print_every == 0:
                print("Episode %d" % episode)
                print(time_since(start))
                print("%.4f %d %d/%d len=%d" %
                      (running_reward,
                       tot_reward,
                       self.act_by_model, self.steps_done, t))
            torch.save(self.model.state_dict(), self.model_fn)


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
        if self.steps_done > EPS_DECAY:
            eps_threshold = EPS_END
        else:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            (1 - self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if test:
            self.test_step += 1
        # if test:
            # eps_threshold = 0.00001
        # if sample > eps_threshold:
        if self.test_step > 2000:
            return self.env.get_random_action()
        if sample > eps_threshold or test:
            self.act_by_model += 1
            state = torch.from_numpy(state).float().unsqueeze(0)
            # state: (1, 84, 84, 4)
            y = self.model(cu(Variable(state)))
            # y: (1, 4)
            act = y.max(1)[1]
            return act.data[0]
        else:
            return self.env.get_random_action()
