from argparse import ArgumentParser

def main(args):
    output = ''
    output += f'BLOCK_SIZE = {args.size}\n'
    output += f'SPEED = {args.speed}\n'
    output += f'MAX_MEM = {args.memory}\n'
    output += f'BATCH_SIZE = {args.batch_size}\n'
    output += f'LR = {args.learning_rate}\n'
    output += f'GAMMA = {args.gamma}\n'
    with open('conf/config.py', 'w') as file:
        file.write(output)

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--size',
        '-s',
        help="The size of each block in the game (default = 8)",
        default=8
    )
    parser.add_argument(
        '--speed',
        '-spd',
        help="The speed of the game (default = 60)",
        default=60
    )
    parser.add_argument(
        '--memory',
        '-m',
        help="The memory size used in training (default = 100000)",
        default=100_000
    )
    parser.add_argument(
        '--batch_size',
        '-bs',
        help="The batch size used in training (default = 1000)",
        default=1000
    )
    parser.add_argument(
        '--learning_rate',
        '-lr',
        help="The learning rate used in training (default = 0.001)",
        default=0.001
    )
    parser.add_argument(
        '--gamma',
        '-g',
        help="The gamma used in training (default = 0.9)",
        default=0.9
    )
    args = parser.parse_args()
    main(args)