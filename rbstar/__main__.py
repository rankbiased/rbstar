import sys
import argparse
from enum import Enum
from statistics import mean, quantiles
from typing import Dict, Tuple, Callable
from pathlib import Path

from rbstar.util import Range
from rbstar.rb_metrics import RBMetric, MetricResult

class Metric(Enum):
    RBP = 'RBP'
    RBO = 'RBO'
    RBA = 'RBA'
    RBR = 'RBR'

def compute_metrics(metric_fn: Callable, observations: Dict, references: Dict, verbose: bool = False, perquery = False) -> MetricResult:
    """Compute metric for all matching query IDs between observations and references"""
    results = {
        qid: metric_fn(obs, references[qid])
        for qid, obs in observations.items()
        if qid in references
    }.items()
    
    if not results:
        return MetricResult(0.0, 0.0)
    
    if verbose:
        # Calculate statistics
        lbs = [result.lower_bound for _, result in results]
        ubs = [result.upper_bound for _, result in results]
        residuals = [result.residual for _, result in results]
        
        print("\n=== Metric Distribution Statistics ===")
        print("Lower Bounds:")
        if lbs:
            p = quantiles(lbs, n=10)  # Deciles
            print(f"  P0={min(lbs):.4f}, P10={p[0]:.4f}, P50={p[4]:.4f}, P90={p[8]:.4f}, P100={max(lbs):.4f}")
        
        print("Upper Bounds:")
        if ubs:
            p = quantiles(ubs, n=10)
            print(f"  P0={min(ubs):.4f}, P10={p[0]:.4f}, P50={p[4]:.4f}, P90={p[8]:.4f}, P100={max(ubs):.4f}")
            
        print("Residuals:")
        if residuals:
            p = quantiles(residuals, n=10)
            print(f"  P0={min(residuals):.4f}, P10={p[0]:.4f}, P50={p[4]:.4f}, P90={p[8]:.4f}, P100={max(residuals):.4f}")
        print()
        
    results_list = [result for _, result in results]

    if perquery:
        print("\n=== Per-Query Metric Values ===")
        print("qid\tlower\tupper\tresidual")
        for qid, result in sorted(results):
            print(f"{qid}\t{result.lower_bound:.4f}\t{result.upper_bound:.4f}\t{result.residual:.4f}")

    return MetricResult(
        mean([r.lower_bound for r in results_list]),
        mean([r.upper_bound for r in results_list])
    )

def rbstar_main():
    parser = argparse.ArgumentParser(
        description="RBStar CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--metric",
        type=str,
        choices=[*[m.name.lower() for m in Metric], *[m.name for m in Metric]],
        help="Specify the metric to use (RBP, RBO, RBA, or RBR)",
        required=True
    )
    parser.add_argument(
        "-o", "--observation",
        type=str,
        help="Path to the observation file",
        required=True
    )
    parser.add_argument(
        "-r", "--reference",
        type=str,
        required=True,
        help="Path to the reference file"
    )
    parser.add_argument(
        "-p", "--phi",
        type=float,
        default=0.95,
        help="Persistence parameter",
        choices=[Range(0, 1)],  # Allow any value between 0 and 1
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print additional statistics"
    )
    parser.add_argument(
        "-q", "--perquery",
        action="store_true",
        help="Print per-query metric values"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    # Convert strings to Path objects
    obs_path = Path(args.observation)
    ref_path = Path(args.reference)
    
    # Validate paths exist
    if not obs_path.exists():
        print(f"Error: Observation file not found: {obs_path}")
        sys.exit(1)
    if not ref_path.exists():
        print(f"Error: Reference file not found: {ref_path}")
        sys.exit(1)

    metric = Metric[args.metric.upper()]
    print(f"Computing {metric.value} with φ = {args.phi}")

    # Create RBMetric instance
    rb_metric = RBMetric(phi=args.phi)

    try:
        # Load data based on metric type
        if metric in [Metric.RBP, Metric.RBR]:
            from rbstar.util import TrecHandler, QrelHandler
                    
            if metric == Metric.RBP:
                trec_handler = TrecHandler()
                qrel_handler = QrelHandler()

                trec_handler.read(str(obs_path))
                qrel_handler.read(str(ref_path))
                if verbose := args.verbose:
                    print("\n=== Input File Statistics ===")
                    print("Observation file (TREC run):")
                    trec_handler.print_stats()
                    print("\nReference file (QREL):")
                    qrel_handler.print_stats()
                observations = trec_handler.to_rbranking_dict()
                references = qrel_handler.to_rbset_dict()
                
                def compute_rbp(obs, ref):
                    rb_metric._observation = obs
                    rb_metric._reference = ref
                    return rb_metric.rb_precision()
                    
                result = compute_metrics(compute_rbp, observations, references, args.verbose, args.perquery)
                
            else:  # RBR
                ref_handler = TrecHandler()
                obs_handler = TrecHandler()

                obs_handler.read(str(obs_path))
                ref_handler.read(str(ref_path))
                if verbose := args.verbose:
                    print("\n=== Input File Statistics ===")
                    print("Observation file (QREL):")
                    obs_handler.print_stats()
                    print("\nReference file (TREC run):")
                    ref_handler.print_stats()
                observations = obs_handler.to_rbset_dict()
                references = ref_handler.to_rbranking_dict()
                
                def compute_rbr(obs, ref):
                    rb_metric._observation = obs
                    rb_metric._reference = ref
                    return rb_metric.rb_recall()
                    
                result = compute_metrics(compute_rbr, observations, references, args.verbose, args.perquery)
        else:
            # For RBO and RBA: both are rankings
            from rbstar.util import TrecHandler
            
            trec_handler = TrecHandler()
            trec_handler.read(str(obs_path))
            observations = trec_handler.to_rbranking_dict()
            if verbose := args.verbose:
                print("\n=== Input File Statistics ===")
                print("Observation file (TREC run):\n")
                print(f"Read {len(trec_handler)} documents for {len(observations)} queries")
                print(f"Average documents per query: {len(trec_handler)/len(observations):.1f}")
            
            trec_handler = TrecHandler()  # Create new instance for reference
            trec_handler.read(str(ref_path))
            references = trec_handler.to_rbranking_dict()
            if verbose := args.verbose:
                print("\nReference file (TREC run):\n")
                print(f"Read {len(trec_handler)} documents for {len(references)} queries")
            
            def compute_metric(obs, ref):
                rb_metric._observation = obs
                rb_metric._reference = ref
                return rb_metric.rb_overlap() if metric == Metric.RBO else rb_metric.rb_alignment()
                
            result = compute_metrics(compute_metric, observations, references, args.verbose, args.perquery)

        # Check if we have any matching query IDs
        matching_qids = set(observations.keys()) & set(references.keys())
        if not matching_qids:
            print("\nError: No matching query IDs found between observation and reference files")
            print(f"Observation queries: {list(observations.keys())}")
            print(f"Reference queries: {list(references.keys())}")
            sys.exit(1)

        if args.json:
            import json
            results = {
                "metric": metric.value,
                "phi": args.phi,
                **result.to_dict()
            }
            print(json.dumps(results))
        else:
            print(f"\n=== Final Metric Results ({len(matching_qids)} obs/refs) ===")
            print(f'Mean score    : {result.lower_bound:>8.4f}')
            print(f'Mean residual : {result.residual:>8.4f}')
            print(f'Mean max score: {result.upper_bound:>8.4f}') 

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
