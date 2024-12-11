import sys
import argparse
from statistics import mean, quantiles
from typing import Dict, Tuple, List
from pathlib import Path
from multiprocessing import Pool, cpu_count

from rbstar.util import Range
from rbstar.rb_metrics import RBMetric, MetricResult
from rbstar.metric_computer import Metric, MetricComputer

def compute_metrics(metric_computer, observations: Dict[str, Dict], references: Dict, 
                   verbose: bool = False, perquery = False) -> Dict[str, Tuple[MetricResult, Dict]]:
    """Compute metric for all matching query IDs between observations and references"""
    results = {}
    
    # Determine number of CPU cores to use (leave one free for system)
    n_cores = max(1, cpu_count() - 1)
    
    for run_name, obs_dict in observations.items():
        # Prepare query processing tasks
        tasks = [
            (None, qid, obs, references[qid])
            for qid, obs in obs_dict.items()
            if qid in references
        ]
        
        if not tasks:
            results[run_name] = (MetricResult(0.0, 0.0), {})
            continue
            
        # Process queries in parallel
        run_results = {}
        with Pool(n_cores) as pool:
            for qid, result in pool.imap_unordered(metric_computer, tasks):
                run_results[qid] = result
        
        if verbose:
            print(f"\n=== Metric Distribution Statistics for {run_name} ===")
            # Calculate statistics
            lbs = [result.lower_bound for result in run_results.values()]
            ubs = [result.upper_bound for result in run_results.values()]
            residuals = [result.residual for result in run_results.values()]
            
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
            
        if perquery:
            print(f"\n=== Per-Query Metric Values for {run_name} ===")
            print("qid\tlower\tupper\tresidual")
            for qid, result in sorted(run_results.items()):
                print(f"{qid}\t{result.lower_bound:.4f}\t{result.upper_bound:.4f}\t{result.residual:.4f}")

        results_list = list(run_results.values())
        results[run_name] = (
            MetricResult(
                mean([r.lower_bound for r in results_list]),
                mean([r.upper_bound for r in results_list])
            ),
            {qid: result.to_dict() for qid, result in run_results.items()}
        )
    
    return results

def output_latex_table(results: Dict[str, Tuple[MetricResult, Dict]], metric: str, phi: float):
    """Output results in LaTeX table format"""
    print("\n% LaTeX table")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print(f"System & {metric}-{phi:.2f} \\\\")
    print("\\midrule")
    
    # Sort runs by score, highest to lowest
    sorted_runs = sorted(results.items(), key=lambda x: x[1][0].lower_bound, reverse=True)
    
    for run_name, (result, _) in sorted_runs:
        # Escape underscores in run names for LaTeX
        run_name_escaped = run_name.replace("_", "\\_")
        print(f"{run_name_escaped} & {result.lower_bound:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")

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
        action='append',
        help="Path(s) to the observation file(s)",
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
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output results in LaTeX table format"
    )
    
    args = parser.parse_args()
    
    # Convert strings to Path objects
    obs_paths = [Path(p) for p in args.observation]
    ref_path = Path(args.reference)
    
    # Validate paths exist
    for obs_path in obs_paths:
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
            
            # First read the reference file since it's shared across all observations
            if metric == Metric.RBP:
                qrel_handler = QrelHandler()
                qrel_handler.read(str(ref_path))
                if args.verbose:
                    print("\nReference file (QREL):")
                    qrel_handler.print_stats()
                references = qrel_handler.to_rbset_dict()
                
                # Now process each observation file
                observations = {}
                for obs_path in obs_paths:
                    trec_handler = TrecHandler()
                    trec_handler.read(str(obs_path))
                    print(f"DEBUG: Processing {obs_path} with run name {trec_handler.run_name}")  # Debug print
                    observations[trec_handler.run_name] = trec_handler.to_rbranking_dict()
                    if args.verbose:
                        print(f"\n=== Input File Statistics for {trec_handler.run_name} ===")
                        print("Observation file (TREC run):")
                        trec_handler.print_stats()
                
                metric_computer = MetricComputer(rb_metric, metric)
                results = compute_metrics(metric_computer, observations, references, args.verbose, args.perquery)
                
            else:  # RBR
                ref_handler = TrecHandler()
                ref_handler.read(str(ref_path))
                if args.verbose:
                    print("\nReference file (TREC run):")
                    ref_handler.print_stats()
                references = ref_handler.to_rbranking_dict()
                
                observations = {}
                for obs_path in obs_paths:
                    obs_handler = TrecHandler()
                    obs_handler.read(str(obs_path))
                    observations[obs_handler.run_name] = obs_handler.to_rbset_dict()
                    if args.verbose:
                        print(f"\n=== Input File Statistics for {obs_handler.run_name} ===")
                        print("Observation file (QREL):")
                        obs_handler.print_stats()
                
                metric_computer = MetricComputer(rb_metric, metric)
                results = compute_metrics(metric_computer, observations, references, args.verbose, args.perquery)
        else:
            # For RBO and RBA: both are rankings
            from rbstar.util import TrecHandler
            
            # First read the reference file
            trec_handler = TrecHandler()
            trec_handler.read(str(ref_path))
            references = trec_handler.to_rbranking_dict()
            if args.verbose:
                print("\nReference file (TREC run):")
                print(f"Read {len(trec_handler)} documents for {len(references)} queries")
            
            # Now process each observation file
            observations = {}
            for obs_path in obs_paths:
                trec_handler = TrecHandler()
                trec_handler.read(str(obs_path))
                observations[trec_handler.run_name] = trec_handler.to_rbranking_dict()
                if args.verbose:
                    print(f"\n=== Input File Statistics for {trec_handler.run_name} ===")
                    print("Observation file (TREC run):")
                    print(f"Read {len(trec_handler)} documents for {len(observations[trec_handler.run_name])} queries")
                    print(f"Average documents per query: {len(trec_handler)/len(observations[trec_handler.run_name]):.1f}")
            
            metric_computer = MetricComputer(rb_metric, metric)
            results = compute_metrics(metric_computer, observations, references, args.verbose, args.perquery)

        # Check if we have any matching query IDs for each run
        for run_name, obs_dict in observations.items():
            matching_qids = set(obs_dict.keys()) & set(references.keys())
            if not matching_qids:
                print(f"\nError: No matching query IDs found between observation {run_name} and reference files")
                print(f"Observation queries: {list(obs_dict.keys())}")
                print(f"Reference queries: {list(references.keys())}")
                sys.exit(1)

        if args.json:
            import json
            json_results = {
                "metric": metric.value,
                "phi": args.phi,
                "runs": {
                    run_name: {
                        **result[0].to_dict(),
                        **({"per_query": result[1]} if args.verbose else {})
                    }
                    for run_name, result in results.items()
                }
            }
            print(json.dumps(json_results, indent=2))
        elif args.latex:
            output_latex_table(results, metric.value, args.phi)
        else:
            for run_name, (result, _) in results.items():
                matching_qids = set(observations[run_name].keys()) & set(references.keys())
                print(f"\n=== Final Metric Results for {run_name} ({len(matching_qids)} obs/refs) ===")
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
