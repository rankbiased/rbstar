import sys
import argparse


def check_metric(metric: str) -> str: # XXX should be an enum or something probably
    if metric.upper() == 'RBP':
        return  'RBP'
    elif metric.upper() == 'RBO':
        return 'RBO'
    elif metric.upper() == 'RBA':
        return 'RBA'
    elif metric.upper() == 'RBR':
        return 'RBR'
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
