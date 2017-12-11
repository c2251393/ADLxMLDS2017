def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--episode', type=int, default=10000000, help='episode count (default: 1000000)')
    parser.add_argument('--episode_len', type=int, default=10000, help='episode length (default: 10000)')
    parser.add_argument('--print_every', type=int, default=100, help='print progress every episode (default: 100)')

    parser.add_argument('--model', type=str, default='', help='model file')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--buffer_size', type=int, default=10000, help='buffer size for dqn training (default: 100000)')
    parser.add_argument('--step_copy', type=int, default=1000, help='step to copy (default: 1000)')
    parser.add_argument('--step_learn', type=int, default=10000, help='learn until this many steps (default: 10000)')
    parser.add_argument('--step_upd', type=int, default=4, help='step to update network (default: 4)')
    parser.add_argument('--step_train', type=int, default=int(1e7), help='total number of updates length (default: 1e7)')

    parser.add_argument('--cnn', action='store_true', help='use cnn model')
    parser.add_argument('--update_every', type=int, default=10, help='step to update network in PG (default: 10)')
    parser.add_argument('--var_reduce', action='store_true', help='variance reduce')
    parser.add_argument('--gae', action='store_true', help='generalized advantage estimation')

    parser.add_argument('--ddqn', action='store_true', help='Double DQN')
    parser.add_argument('--duel', action='store_true', help='Duel DQN')

    parser.add_argument('--clip', action='store_true', help='clip grad magnitude for DQN')

    parser.add_argument('--a2c', action='store_true', help='run a2c for breakout (agent_dqn)')
    return parser
