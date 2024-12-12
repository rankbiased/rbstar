import sys
import argparse
from statistics import mean, quantiles
from typing import Dict, Tuple, List
from pathlib import Path
from multiprocessing import Pool, cpu_count
import json

from rbstar.util import Range
from rbstar.rb_metrics import RBMetric, MetricResult
from rbstar.metric_computer import Metric, MetricComputer

def compute_query_tasks(observations: Dict[str, Dict], references: Dict) -> Dict[str, List[Tuple[None, str, Dict, Dict]]]:
    """
    Prepares query tasks for computation.

    Args:
        observations: Dictionary of observed rankings by run name.
        references: Dictionary of reference rankings.

    Returns:
        A dictionary mapping run names to lists of query tasks.
    """
    tasks = {}
    for run_name, obs_dict in observations.items():
        tasks[run_name] = [
            (None, qid, obs, references[qid])
            for qid, obs in obs_dict.items()
            if qid in references
        ]
    return tasks

def compute_metrics_for_run(metric_computer, tasks: List[Tuple]) -> Dict[str, MetricResult]:
    """
    Compute metrics for a single run using multiprocessing.

    Args:
        metric_computer: MetricComputer instance.
        tasks: List of query tasks.

    Returns:
        A dictionary mapping query IDs to MetricResult objects.
    """
    run_results = {}
    n_cores = max(1, cpu_count() - 1)
    with Pool(n_cores) as pool:
        for qid, result in pool.imap_unordered(metric_computer, tasks):
            run_results[qid] = result
    return run_results

def calculate_statistics(results: Dict[str, MetricResult], verbose: bool):
    """
    Calculates and prints statistics for a set of results if verbose is enabled.

    Args:
        results: Dictionary of query results by query ID.
        verbose: Whether to print statistics.
    """
    if not verbose:
        return

    print("\n=== Metric Distribution Statistics ===")
    lbs = [result.lower_bound for result in results.values()]
    ubs = [result.upper_bound for result in results.values()]
    residuals = [result.residual for result in results.values()]

    def print_stats(name, values):
        if values:
            p = quantiles(values, n=10)  # Deciles
            print(f"{name}:")
            print(f"  P0={min(values):.4f}, P10={p[0]:.4f}, P50={p[4]:.4f}, P90={p[8]:.4f}, P100={max(values):.4f}")

    print_stats("Lower Bounds", lbs)
    print_stats("Upper Bounds", ubs)
    print_stats("Residuals", residuals)

    print()

def aggregate_results(run_results: Dict[str, MetricResult]) -> MetricResult:
    """
    Aggregates per-query results into a single MetricResult.

    Args:
        run_results: Dictionary of per-query MetricResult objects.

    Returns:
        Aggregated MetricResult.
    """
    results_list = list(run_results.values())
    return MetricResult(
        mean([r.lower_bound for r in results_list]),
        mean([r.upper_bound for r in results_list])
    )

def output_results(results: Dict[str, Tuple[MetricResult, Dict]], metric: str, phi: float, args):
    """
    Outputs results in the desired format (JSON, LaTeX, or plain text).

    Args:
        results: Final results by run name.
        metric: Metric name.
        phi: Persistence parameter.
        args: Parsed command-line arguments.
    """
    if args.json:
        json_results = {
            "metric": metric,
            "phi": phi,
            "runs": {
                run_name: {
                    **result[0].to_dict(),
                    **({"per_query": result[1]} if args.perquery else {})
                }
                for run_name, result in results.items()
            }
        }
        print(json.dumps(json_results, indent=2))
    elif args.latex:
        print("\n% LaTeX table")
        print("\\begin{tabular}{lc}")
        print("\\toprule")
        print(f"System & {metric}-{phi:.2f} \\\\")
        print("\\midrule")

        sorted_runs = sorted(results.items(), key=lambda x: x[1][0].lower_bound, reverse=True)
        for run_name, (result, _) in sorted_runs:
            run_name_escaped = run_name.replace("_", "\\_")
            print(f"{run_name_escaped} & {result.lower_bound:.3f} \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")
    else:
        if args.perquery: # we'll do plaintext per-query output
            for run_name, (_, rdict) in results.items():
                print(f"\n=== Per-Query Metric Results for {run_name} ===")
                print(f"qid\tlower\tupper\tresidual")
                for qid, result in rdict.items():
                    lb = result["lower_bound"]
                    ub = result["upper_bound"]
                    res = result["residual"]
                    print(f"{qid}\t{lb:.4f}\t{ub:.4f}\t{res:.4f}")
            
        for run_name, (result, _) in results.items():
            print(f"\n=== Final Metric Results for {run_name} ===")
            print(f'Mean score    : {result.lower_bound:>8.4f}')
            print(f'Mean residual : {result.residual:>8.4f}')
            print(f'Mean max score: {result.upper_bound:>8.4f}')

def rbstar_main():
    """
    Entry point for the RBStar CLI tool.
    """
    parser = argparse.ArgumentParser(
        description="RBStar CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-m", "--metric", type=str, choices=[m.name.lower() for m in Metric], required=True, help="Specify the metric to use")
    parser.add_argument("-o", "--observation", type=str, action='append', required=True, help="Path(s) to the observation file(s)")
    parser.add_argument("-r", "--reference", type=str, required=True, help="Path to the reference file")
    parser.add_argument("-p", "--phi", type=float, default=0.95, help="Persistence parameter", choices=[Range(0, 1)])
    parser.add_argument("-v", "--verbose", action="store_true", help="Print additional statistics")
    parser.add_argument("-q", "--perquery", action="store_true", help="Print per-query metric values")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--latex", action="store_true", help="Output results in LaTeX table format")
    args = parser.parse_args()

    # Validate paths
    obs_paths = [Path(p) for p in args.observation]
    ref_path = Path(args.reference)

    for obs_path in obs_paths:
        if not obs_path.exists():
            sys.exit(f"Error: Observation file not found: {obs_path}")
    if not ref_path.exists():
        sys.exit(f"Error: Reference file not found: {ref_path}")

    metric = Metric[args.metric.upper()]
    rb_metric = RBMetric(phi=args.phi)

    # Load data
    from rbstar.util import TrecHandler, QrelHandler

    if metric in [Metric.RBP, Metric.RBR]:
        qrel_handler = QrelHandler() if metric == Metric.RBP else TrecHandler()
        qrel_handler.read(str(ref_path))
        references = qrel_handler.to_rbset_dict() if metric == Metric.RBP else qrel_handler.to_rbranking_dict()

        observations = {}
        for obs_path in obs_paths:
            trec_handler = TrecHandler()
            trec_handler.read(str(obs_path))
            observations[trec_handler.run_name] = (
                trec_handler.to_rbset_dict() if metric == Metric.RBR else trec_handler.to_rbranking_dict()
            )

    # Compute metrics
    metric_computer = MetricComputer(rb_metric, metric)
    query_tasks = compute_query_tasks(observations, references)
    results = {}
    for run_name, tasks in query_tasks.items():
        if not tasks:
            results[run_name] = (MetricResult(0.0, 0.0), {})
            continue

        run_results = compute_metrics_for_run(metric_computer, tasks)
        calculate_statistics(run_results, args.verbose)
        results[run_name] = (
            aggregate_results(run_results),
            {qid: result.to_dict() for qid, result in run_results.items()}
        )

    # Output results
    output_results(results, metric.value, args.phi, args)

if __name__ == "__main__":
    rbstar_main()
