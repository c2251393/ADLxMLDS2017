from agent_dir.agent import Agent

from utils import *


'''
When running pg:
    observation: np.array
        current RGB screen of game, shape: (210, 160, 3)
    reward: int
        if opponent wins, reward = +1 else -1
    done: bool
        whether reach the end of the episode?
'''

def shrink(frame):
    '''
    frame: np.array
        current RGB screen of game, shape: (210, 160, 3)
    @output: gray scale np.array: (1, 80, 80)
    '''
    frame = frame[34: 34+160, :160]
    frame = resize(rgb2gray(frame), (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 512)

        self.W = nn.Linear(512, 6)
        self.apply(weights_init)
        self.W.weight.data = norm_col_init(
            self.W.weight.data, 0.01)
        self.W.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.hidden = cu(Variable(torch.zeros(1, 512))), cu(Variable(torch.zeros(1, 512)))

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        # print(hidden[0].size(), hidden[1].size())
        hx, cx = self.lstm(x, self.hidden)
        self.hidden = (hx, cx)
        return self.W(hx)

    def init_hidden(self):
        self.hidden = cu(Variable(torch.zeros(1, 512))), cu(Variable(torch.zeros(1, 512)))


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
        self.n_warm = args.warm
        self.gamma = args.gamma
        self.episode_len = args.episode_len

        self.model = Model()
        self.opt = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        self.state = cu(Variable(torch.zeros(1, 80, 80).float()))
        self.log_probs = []
        self.rewards = []

        self.warmup = True

        if args.test_pg:
            #you can load your model here
            print('loading trained model :%s.' % args.model)
            state_dict = torch.load(args.model, map_location=lambda storage, location: storage)
            self.model.load_state_dict(state_dict)
            self.warmup = False


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.state = torch.zeros(1, 80, 80).float()
        self.model.init_hidden()


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
                if abs(self.rewards[i]) > 0.0:
                    R = 0
                R = self.rewards[i] + self.gamma * R
                self.rewards[i] = R
            rewards = torch.Tensor(self.rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

            policy_loss = 0.0
            for (log_prob, r) in zip(self.log_probs, rewards):
                policy_loss = policy_loss - log_prob * r

            loss = policy_loss.data[0, 0]

            self.opt.zero_grad()
            policy_loss = cu(policy_loss)
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)
            self.opt.step()

            self.clear_action()
            return loss

        self.model.train()
        if USE_CUDA:
            self.model.cuda()

        for episode in range(self.n_episode):
            print("Episode %d" % episode)
            if episode > self.n_warm:
                self.warmup = False
            self.init_game_setting()
            state = self.env.reset()

            tot_reward = 0
            a, b = 0, 0
            elen = 0
            loss = 0
            for t in range(self.episode_len):
                action = self.make_action(state, test=False)
                state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)
                if reward > 0:
                    a += 1
                if reward < 0:
                    b += 1
                tot_reward += reward
                if abs(reward) > 0:
                    loss = finish_episode()
                if done:
                    elen = t+1
                    break

            print(tot_reward, a, b, elen)
            print(loss)
            print(time_since(start))
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
        # return self.env.get_random_action()
        state = torch.from_numpy(shrink(state)).float()
        y = self.model(cu(Variable(state)))
        self.state = state

        prob = F.softmax(y)
        log_prob = F.log_softmax(y)

        if not test and self.warmup:
            act = torch.LongTensor([[self.env.get_random_action()]])
        else:
            act = prob.multinomial().data
        log_prob = log_prob.gather(1, cu(Variable(act)))

        if not test:
            self.log_probs.append(log_prob)
        return act[0, 0]

    def clear_action(self):
        self.log_probs = []
        self.rewards = []
        hx, cx = self.model.hidden
        self.model.hidden = cu(Variable(hx.data)), cu(Variable(cx.data))
