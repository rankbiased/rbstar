import sys
import argparse
from enum import Enum


class Metric(Enum):
    RBP = 'RBP'
    RBO = 'RBO'
    RBA = 'RBA'
    RBR = 'RBR'


def check_metric(metric: str) -> Metric:
    try:
        return Metric[metric.upper()]
    except KeyError:
        raise ValueError("Provided metric not recognized.")


def rbstar_main():
    parser = argparse.ArgumentParser(description="RBStar CLI")
    parser.add_argument("-m", "--metric", type=str, help="Specify the desired metric.", required=True)
    parser.add_argument("--observation", type=str, help="Path to the observation.", required=True)
    parser.add_argument("--reference", type=str, help="Path to the reference.", required=True)
    parser.add_argument("-p", "--phi", type=float, default="0.95", help="Persistence parameter.")
    args = parser.parse_args() 
    
    metric_string = check_metric(args.metric)
    print("Computing", metric_string, "𝜙 =", args.phi)


if __name__ == "__main__":
    rbstar_main() 
