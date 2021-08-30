import argparse

def argsParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ratio', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='cave')
    parser.add_argument('--device',type=str, default='cuda:0')

    args = parser.parse_args()

    return args
