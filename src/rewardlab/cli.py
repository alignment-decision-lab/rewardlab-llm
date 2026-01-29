import argparse
from rewardlab.eval.runner import run_demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    args = parser.parse_args()
    run_demo()

if __name__ == "__main__":
    main()
