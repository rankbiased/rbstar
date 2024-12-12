<p align="center">
  <img align="center" src="docs/static/rbstar.png" width="400px" />
</p>
<p align="left">

# RBStar

[![Python Package](https://github.com/rankbiased/rbstar/actions/workflows/python.yml/badge.svg)](https://github.com/rankbiased/rbstar/actions/workflows/python.yml)

RBStar is a Python library and command-line tool that implements a family of rank-biased effectiveness metrics for information retrieval evaluation. These metrics are designed to evaluate ranked lists and set-based relevance judgments, offering robust and flexible options for diverse evaluation needs.


## Features

- **Multiple Metrics**:
  - **RBP** (Rank-Biased Precision): User persistence-based ranking precision.
  - **RBO** (Rank-Biased Overlap): Similarity measure for ranked lists.
  - **RBA** (Rank-Biased Alignment): A hybrid metric for ranking alignment.
  - **RBR** (Rank-Biased Recall): Set-based recall with rank bias.

- **Flexible Input Handling**:
  - Supports TREC-style run files
  - Handles qrels (relevance judgments)
  - Works with both ranked lists and set-based data

- **Efficient Computation**:
  - Parallelized query-level metric computation.
  - Handles large-scale evaluations effectively.

- **Tie Handling**: Explicit support for tied rankings through group-based calculations

## Installation

You can install RBStar via pip:

```bash
pip install rbstar
```

## Quick Start

Here is a quick example to get started with RBStar:

```python
from rbstar import RBMetric, RBRanking, RBSet

# Create a metric instance with persistence parameter
metric = RBMetric(phi=0.8)

# Example with rankings
ranking1 = RBRanking()
ranking1.append([1, 2])  # Tied elements
ranking1.append([3])

ranking2 = RBRanking()
ranking2.append([2, 3])
ranking2.append([1])

# Calculate RBO score
metric._observation = ranking1
metric._reference = ranking2
lb, ub = metric.rb_overlap()
print(f"RBO bounds: [{lb}, {ub}]")
```

## Command Line Usage

RBStar also includes a command-line interface for easy metric computation.

```bash
python -m rbstar -m RBO --observation run.txt --reference other-run.txt --phi 0.95
```

### Arguments:
- `-m, --metric`: Metric to compute (RBP, RBO, RBA, RBR).
- `-o, --observation`: Path to the observation file.
- `-r, --reference`: Path to the reference file.
- `-p, --phi`: Persistence parameter (default: 0.95).
- `-v, --verbose`: Enable verbose output with detailed statistics.
- `-q, --perquery`: Output per-query metric values.
- `--json`: Output results in JSON format.
- `--latex`: Output results in LaTeX table format.

## Examples

### Basic Usage with Default Parameters

```bash
python -m rbstar -m RBP -o observation.trec -r reference.qrel
```

### Generate LaTeX Output

```bash
python -m rbstar -m RBO -o observation.trec -r reference.qrel --latex
```

### JSON Output with Verbose Statistics

```bash
python -m rbstar -m RBA -o observation.trec -r reference.qrel --json -v
```

## Metrics Overview

### RBP (Rank-Biased Precision)
- Focuses on user persistence in evaluating precision.
- Handles both complete and incomplete rankings.
- Returns lower and upper effectiveness bounds.

### RBO (Rank-Biased Overlap)
- Measures similarity between ranked lists.
- Handles incomplete rankings and non-conjoint sets.
- Based on set overlap at each rank position.

### RBA (Rank-Biased Alignment)
- Combines properties of RBP and RBR.
- Symmetric: RBA(A, B) = RBA(B, A).
- Handles ties through weight sharing.

### RBR (Rank-Biased Recall)
- A set-based effectiveness measure.
- Accounts for both positive and negative elements.
- Handles incomplete judgments effectively.

## Integration with Custom Scripts

RBStar is designed for flexibility. You can integrate it with custom pipelines and data preprocessing workflows using the Python API:

```python
from rbstar import RBMetric

# Instantiate the metric
metric = RBMetric(phi=0.85)

# Add custom ranking data
# Example usage with your own ranking data
# metric.compute_metrics(...)  # Customize for your input format
```


## Citation

If you use RBStar in your research, please cite:
```
@inproceedings{corsi2024rbstar,
  title={RBStar: A Family of Rank-Biased Effectiveness Metrics},
  author={TK},
  booktitle={Proceedings of SIGIR '24},
  year={2024},
  doi={10.1145/3626772.3657700}
}
```

## License

RBStar is licensed under the MIT License.
