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

'''
frame: np.array
    current RGB screen of game, shape: (210, 160, 3)
@output: single color np.array: (1, 80, 80)
'''
# def shrink(I):
    # I = I[35:195]
    # I = I[::2, ::2, 0]
    # I[I == 144] = 0
    # I[I == 109] = 0
    # I[I != 0] = 1
    # return I.reshape(1, 80, 80)


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py

    Input:
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array
        Grayscale image, shape: (1, 80, 80)

    """
    # y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    # y = y.astype(np.uint8)
    y = 0.2125 * o[:, :, 0] + 0.7154 * o[:, :, 1] + 0.0721 * o[:, :, 2]
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=0)


def shrink(frame):
    frame = frame[35: 35+160, :160]
    return prepro(frame)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.W1 = nn.Linear(80*80, 256)
        self.W2 = nn.Linear(256, 6)
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.W1(x))
        x = self.W2(x)
        return x


class ModelGAE(nn.Module):
    def __init__(self):
        super(ModelGAE, self).__init__()
        self.W1 = nn.Linear(80*80, 256)
        self.Wa = nn.Linear(256, 6)
        self.Wv = nn.Linear(256, 1)
        self.apply(weights_init)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.W1(x))
        return self.Wa(x), self.Wv(x)


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.W1 = nn.Linear(2048, 256)
        self.W2 = nn.Linear(256, 6)
        self.apply(weights_init)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.W1(x))
        x = self.W2(x)
        return x


class ModelGAE2(nn.Module):
    def __init__(self):
        super(ModelGAE2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.W1 = nn.Linear(2048, 256)
        self.Wa = nn.Linear(256, 6)
        self.Wv = nn.Linear(256, 1)
        self.apply(weights_init)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.W1(x))
        return self.Wa(x), self.Wv(x)


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
        self.print_every = args.print_every
        self.n_episode = args.episode
        self.gamma = args.gamma
        self.episode_len = args.episode_len
        self.update_every = args.update_every
        self.var_reduce = args.var_reduce
        self.gae = args.gae
        self.step_upd = args.step_upd
        self.max_step = args.step_train

        if not self.gae:
            if args.cnn:
                self.model = Model2()
            else:
                self.model = Model()
        else:
            if args.cnn:
                self.model = ModelGAE2()
            else:
                self.model = ModelGAE()

        self.opt = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)

        self.state = np.zeros((1, 80, 80))
        self.log_probs = []
        self.rewards = []
        if self.gae:
            self.values = []
            self.entropies = []

        self.model_fn = args.model
        if self.model_fn == '':
            self.model_fn = 'agent_pg.pt'

        if args.test_pg:
            #you can load your model here
            print('loading trained model :%s.' % args.model)
            state_dict = torch.load(args.model, map_location=lambda storage, location: storage)
            self.model.load_state_dict(state_dict)
            if USE_CUDA:
                self.model.cuda()


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.state = np.zeros((1, 80, 80))
        self.clear_action()


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        start = time.time()
        if self.gae:
            self.train_gae()
            return

        def optimize_model():
            R = 0
            for i in reversed(range(len(self.rewards))):
                if abs(self.rewards[i]) > 0.0:
                    R = 0
                R = self.rewards[i] + self.gamma * R
                self.rewards[i] = R
            rewards = torch.Tensor(self.rewards)
            if self.var_reduce:
                rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

            policy_loss = 0.0
            for (log_prob, r) in zip(self.log_probs, rewards):
                policy_loss -= log_prob * r

            loss = policy_loss.data[0, 0]

            self.opt.zero_grad()
            policy_loss = cu(policy_loss)
            policy_loss.backward()
            self.opt.step()

            self.clear_action()
            return loss

        self.model.train()
        if USE_CUDA:
            self.model.cuda()
        running_reward = None

        for episode in range(1, self.n_episode+1):
            self.init_game_setting()
            state = self.env.reset()

            tot_reward = 0
            a, b = 0, 0
            for t in range(self.episode_len):
                action = self.make_action(state, test=False)
                state, reward, done, info = self.env.step(action)
                self.rewards.append(reward)
                if reward > 0:
                    a += 1
                if reward < 0:
                    b += 1
                tot_reward += reward
                if done:
                    break

            if running_reward is None:
                running_reward = tot_reward
            else:
                running_reward = 0.99 * running_reward + 0.01 * tot_reward

            if episode % self.update_every == 0:
                loss = optimize_model()
                print("Episode %d" % episode)
                print(time_since(start))
                print("reward %.4f %d:%d len=%d" % (running_reward, a, b, t))
                torch.save(self.model.state_dict(), self.model_fn)


    def train_gae(self):
        start = time.time()
        self.model.train()
        if USE_CUDA:
            self.model.cuda()
        running_reward = None

        def optimize_model():
            policy_loss = 0
            value_loss = 0

            R = cu(Variable(torch.zeros(1, 1)))
            gae = cu(torch.zeros(1, 1))

            self.values.append(cu(Variable(torch.zeros(1, 1))))

            for i in reversed(range(len(self.rewards))):
                R = self.gamma * R + self.rewards[i]
                advantage = R - self.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                delta_t = self.rewards[i] + self.gamma * self.values[i+1].data \
                            - self.values[i].data
                gae = gae * self.gamma + delta_t

                policy_loss = policy_loss - self.log_probs[i] * cu(Variable(gae)) - 0.01 * self.entropies[i]

            target = (policy_loss + 0.5 * value_loss)
            loss = target.data[0, 0]

            self.opt.zero_grad()
            target.backward()
            self.opt.step()

            self.clear_action()

            return loss

        self.init_game_setting()
        state = self.env.reset()
        tot_reward, a, b = 0, 0, 0

        episode = 0

        for epoch in range(self.max_step):
            action = self.make_action(state, test=False)
            state, reward, done, info = self.env.step(action)
            self.rewards.append(reward)

            if reward > 0:
                a += 1
            if reward < 0:
                b += 1
            tot_reward += reward

            if done:
                episode += 1
                if running_reward is None:
                    running_reward = tot_reward
                else:
                    running_reward = 0.99 * running_reward + 0.01 * tot_reward
                if episode % self.print_every == 0:
                    print("Episode %d" % episode)
                    print(time_since(start))
                    print("%.4f %d:%d" % (running_reward, a, b))

                self.init_game_setting()
                state = self.env.reset()
                tot_reward, a, b = 0, 0, 0

            if epoch % self.step_upd == 0:
                optimize_model()
                torch.save(self.model.state_dict(), self.model_fn)

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
        state = shrink(state)
        d_state = state - self.state
        if self.gae:
            y, val = self.model(cu(Variable(torch.from_numpy(d_state).float())))
        else:
            y = self.model(cu(Variable(torch.from_numpy(d_state).float())))
        self.state = state

        prob = F.softmax(y)
        log_prob = F.log_softmax(y)
        entropy = -(log_prob * prob).sum(1)
        act = prob.multinomial().data
        log_prob = log_prob.gather(1, cu(Variable(act)))

        if not test:
            self.log_probs.append(log_prob)
            if self.gae:
                self.values.append(val)
                self.entropies.append(entropy)
        return act[0, 0]

    def clear_action(self):
        self.log_probs = []
        self.rewards = []
        if self.gae:
            self.values = []
            self.entropies = []
