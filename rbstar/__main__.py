import sys
import argparse
from enum import Enum
from statistics import mean
from typing import Dict, Tuple, Callable

from rbstar.rb_metrics import RBMetric

class Metric(Enum):
    RBP = 'RBP'
    RBO = 'RBO'
    RBA = 'RBA'
    RBR = 'RBR'

def compute_metrics(metric_fn: Callable, observations: Dict, references: Dict) -> Tuple[float, float]:
    """Compute metric for all matching query IDs between observations and references"""
    results = [(metric_fn(obs, ref)) 
              for qid, obs in observations.items()
              if (qid in references and (ref := references[qid]))]
    
    if not results:
        return 0.0, 0.0
        
    lbs, ubs = zip(*results)
    return mean(lbs), mean(ubs)

def rbstar_main():
    parser = argparse.ArgumentParser(
        description="RBStar CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--metric",
        type=str,
        choices=[m.name.lower() for m in Metric] + [m.name for m in Metric],
        help="Specify the metric to use (RBP, RBO, RBA, or RBR)",
        required=True
    )
    parser.add_argument(
        "-o", "--observation",
        type=argparse.FileType('r'),
        help="Path to the observation file",
        required=True
    )
    parser.add_argument(
        "-r", "--reference",
        type=argparse.FileType('r'),
        required=True,
        help="Path to the reference file"
    )
    parser.add_argument(
        "-p", "--phi",
        type=float,
        default=0.95,
        help="Persistence parameter",
        choices=[x/100 for x in range(1, 101)],  # Allow any value between 0.01 and 1.00
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print additional statistics"
    )
    
    args = parser.parse_args()
    metric = Metric[args.metric.upper()]
    print(f"Computing {metric.value} with φ = {args.phi}")

    # Create RBMetric instance
    rb_metric = RBMetric(phi=args.phi)

    try:
        # Load data based on metric type
        if metric in [Metric.RBP, Metric.RBR]:
            from rbstar.util import TrecHandler, QrelHandler
            
            trec_handler = TrecHandler()
            qrel_handler = QrelHandler()
            
            if metric == Metric.RBP:
                trec_handler.read(args.observation.name)
                qrel_handler.read(args.reference.name)
                if args.verbose:
                    print("\nObservation file statistics:")
                    trec_handler.print_stats()
                    print("\nReference file statistics:")
                    qrel_handler.print_stats()
                observations = trec_handler.to_rbranking_dict()
                references = qrel_handler.to_rbset_dict()
                
                def compute_rbp(obs, ref):
                    rb_metric._observation = obs
                    rb_metric._reference = ref
                    return rb_metric.rb_precision()
                    
                lb, ub = compute_metrics(compute_rbp, observations, references)
                
            else:  # RBR
                qrel_handler.read(args.observation.name)
                trec_handler.read(args.reference.name)
                if args.verbose:
                    print("\nObservation file statistics:")
                    qrel_handler.print_stats()
                    print("\nReference file statistics:")
                    trec_handler.print_stats()
                observations = qrel_handler.to_rbset_dict()
                references = trec_handler.to_rbranking_dict()
                
                def compute_rbr(obs, ref):
                    rb_metric._observation = obs
                    rb_metric._reference = ref
                    return rb_metric.rb_recall()
                    
                lb, ub = compute_metrics(compute_rbr, observations, references)
        else:
            # For RBO and RBA: both are rankings
            from rbstar.util import TrecHandler
            
            trec_handler = TrecHandler()
            trec_handler.read(args.observation.name)
            if args.verbose:
                print("\nObservation file statistics:")
                trec_handler.print_stats()
            observations = trec_handler.to_rbranking_dict()
            
            trec_handler = TrecHandler()  # Create new instance for reference
            trec_handler.read(args.reference.name)
            if args.verbose:
                print("\nReference file statistics:")
                trec_handler.print_stats()
            references = trec_handler.to_rbranking_dict()
            
            def compute_metric(obs, ref):
                rb_metric._observation = obs
                rb_metric._reference = ref
                return rb_metric.rb_overlap() if metric == Metric.RBO else rb_metric.rb_alignment()
                
            lb, ub = compute_metrics(compute_metric, observations, references)

        # Check if we have any matching query IDs
        matching_qids = set(observations.keys()) & set(references.keys())
        if not matching_qids:
            print("\nError: No matching query IDs found between observation and reference files")
            print(f"Observation queries: {list(observations.keys())}")
            print(f"Reference queries: {list(references.keys())}")
            sys.exit(1)

        print(f"Mean lower bound: {lb:.4f}")
        print(f"Mean upper bound: {ub:.4f}") 
        print(f"Mean residual: {(ub - lb):.4f}")

    except (ValueError, TypeError, AssertionError) as e:
        print(f"\nError processing data: {str(e)}")
        print("Please check that your input files are in the correct format for the chosen metric:")
        print(f"- {metric.value} expects:")
        if metric == Metric.RBP:
            print("  * Observation: TREC run file")
            print("  * Reference: QREL file")
        elif metric == Metric.RBR:
            print("  * Observation: QREL file")
            print("  * Reference: TREC run file")
        else:
            print("  * Both files: TREC run files")
        sys.exit(1)

if __name__ == "__main__":
    rbstar_main()
