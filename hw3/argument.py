def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
    parser.add_argument('--episode', type=int, default=1000, help='episode count (default: 1000)')
    return parser
