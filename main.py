import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Boosted Autoencoders")

    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help="Mode to run: train or evaluate")

    args = parser.parse_args()

    if args.mode == 'train':
        subprocess.run(['python', 'train.py'])
    elif args.mode == 'evaluate':
        subprocess.run(['python', 'evaluate.py'])


if __name__ == '__main__':
    main()
