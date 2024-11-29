import sys
import argparse


def check_metric(metric: str) -> any: # XXX should be an enum or something probably
    pass

def rbstar_main():
    parser = argparse.ArgumentParser(description="RBStar CLI")
    parser.add_argument("-m, --metric", type=str, help="Specify the desired metric.")
    parser.add_argument("--observation", type=str, help="Path to the observation.")
    parser.add_argument("--reference", type=str, help="Path to the reference.")
    parser.add_argument("-p, --phi", type=float, default="0.95", help="Persistence parameter.")
    args = parser.parse_args() 


if __name__ == "__main__":
    rbstar_main() 
