<p align="center">
  <img align="center" src="docs/static/rbstar.png" width="400px" />
</p>
<p align="left">

# RBStar

RBStar is a Python library that implements a family of rank-biased effectiveness metrics for information retrieval evaluation. These metrics are designed to handle both ranked lists and set-based relevance judgments.

## Features

- **Multiple Metrics**: Implements several rank-biased metrics:
  - RBP (Rank-Biased Precision)
  - RBO (Rank-Biased Overlap)
  - RBA (Rank-Biased Alignment)
  - RBR (Rank-Biased Recall)

- **Flexible Input Handling**:
  - Supports TREC-style run files
  - Handles qrels (relevance judgments)
  - Works with both ranked lists and set-based data

- **Tie Handling**: Explicit support for tied rankings through group-based calculations

## Installation

```bash
pip install rbstar
```

## Quick Start

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

```bash
python -m rbstar -m RBO --observation run.txt --reference qrels.txt --phi 0.95
```

Arguments:
- `-m, --metric`: Metric to compute (RBP, RBO, RBA, RBR)
- `--observation`: Path to observation file
- `--reference`: Path to reference file
- `-p, --phi`: Persistence parameter (default: 0.95)

## Metrics Overview

### RBP (Rank-Biased Precision)
- Evaluates effectiveness considering user persistence
- Handles both complete and incomplete rankings
- Returns lower and upper effectiveness bounds

### RBO (Rank-Biased Overlap)
- Measures similarity between two rankings
- Handles incomplete rankings and non-conjoint sets
- Based on set overlap at each rank position

### RBA (Rank-Biased Alignment)
- Novel metric combining properties of RBP and RBR
- Symmetric: RBA(A,B) = RBA(B,A)
- More nuanced than RBO for misalignment types
- Handles ties through weight sharing

### RBR (Rank-Biased Recall)
- Set-based effectiveness measure
- Considers both positive and negative elements
- Handles incomplete judgments

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

MIT License
