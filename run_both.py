import argparse
import sys

# Your argparse setup
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--Connection", help="Connection scheme (fc or lc)", type=str)
parser.add_argument("-s", "--Seed", help="Seed number, should be an integer", type=int)
parser.add_argument("-fs", "--FilterSize", help="Filter size, should be an integer", type=int)
parser.add_argument("-p", "--Padding", help="Padding size, should be an integer", type=int)
args = parser.parse_args()

# Set up sys.argv for train.py
train_args = ['train.py']
if args.Connection:
    train_args.extend(['-c', args.Connection])
if args.Seed is not None:
    train_args.extend(['-s', str(args.Seed)])
if args.FilterSize is not None:
    train_args.extend(['-fs', str(args.FilterSize)])
if args.Padding is not None:
    train_args.extend(['-p', str(args.Padding)])

sys.argv = train_args

# Execute train.py in the current namespace
with open('train.py', 'r') as f:
    exec(f.read())

print("train.py completed, variables like run_dir are now available")

# Reset argv for test.py and execute it
sys.argv = ['test.py']
with open('test.py', 'r') as f:
    exec(f.read())
