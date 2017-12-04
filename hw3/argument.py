def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--episode', type=int, default=10000000, help='episode count (default: 1000000)')
    parser.add_argument('--episode_len', type=int, default=10000, help='episode length (default: 10000)')

    parser.add_argument('--warm', type=int, default=200, help='warmup length (default: 200)')
    parser.add_argument('--model', type=str, default='', help='model file')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--buffer_size', type=int, default=10000, help='buffer size for dqn training (default: 100000)')
    parser.add_argument('--step_copy', type=int, default=10000, help='step to copy (default: 10000)')

    parser.add_argument('--var_reduce', action='store_true', help='variance reduce')
    parser.add_argument('--off_policy', action='store_true', help='off policy learning')
    parser.add_argument('--gae', action='store_true', help='generalized advantage estimation')

    parser.add_argument('--ddqn', action='store_true', help='Double DQN')
    parser.add_argument('--duel', action='store_true', help='Duel DQN')

    parser.add_argument('--clip', action='store_true', help='clip grad magnitude for DQN')
    return parser
