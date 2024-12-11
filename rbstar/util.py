from typing import NamedTuple
from collections import defaultdict
from pathlib import Path
from .rb_ranking import RBRanking 
from .rb_set import RBSet
from dataclasses import dataclass


class Range:
    """
    Helper class for checking PHI ranges.
    https://stackoverflow.com/a/12117089
    """
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
    def __repr__(self) -> str:
        return "[" + str(self.start) + "," + str(self.end) + "]"

# Use the ScoredDoc and Qrel types from ir_measures, but extend ScoredDoc
# with a rank attribute. 
# https://github.com/terrierteam/ir_measures/blob/main/ir_measures/util.py
@dataclass
class Qrel:
    query_id: str
    doc_id: str
    relevance: int
    iteration: str = '0'

@dataclass
class ScoredDoc:
    query_id: str
    doc_id: str
    score: float
    rank: int
    run_name: str

class QrelHandler:
    """
    Handles reading qrels and conversion to RBStar types.
    TODO: Currently only handles binary, treating any qrel > 0 as relevant,
    and anything <= 0 as non-rel. See POSITIVE_CUTOFF in rb_set.py
    """
    def __init__(self):
        self._data = []

    def read(self, path: Path | str) -> None:
        """
        Read qrels file at path into _data for later processing.
        
        Args:
            path: Path to qrels file
            
        Raises:
            AssertionError: If handler already contains data
            ValueError: If no valid qrels were read
        """
        if self._data:
            raise AssertionError("Cannot read into non-empty QrelHandler")

        path = Path(path)
        # Read entire file into memory first
        with path.open() as f:
            lines = f.readlines()

        # Process all lines at once
        for line in lines:
            qid, _, docid, rel = line.split()
            self._data.append(Qrel(qid, docid, int(rel)))
                
        if not self._data:
            raise ValueError(f"No valid qrels found in {path}")

    def print_stats(self) -> None:
        """Print statistics about qrels data."""
        query_counts = defaultdict(int)
        rel_counts = defaultdict(int)
        for qrel in self._data:
            query_counts[qrel.query_id] += 1
            rel_counts[qrel.relevance] += 1
            
        print(f"\nRead {len(self._data)} qrels for {len(query_counts)} queries")
        print(f"Average qrels per query: {len(self._data)/len(query_counts):.1f}")
        print("Relevance level distribution:")
        for rel, count in sorted(rel_counts.items()):
            print(f"  Level {rel}: {count} qrels")

    def to_rbset_dict(self) -> dict[str, RBSet]:
        """
        Convert qrels to dictionary mapping query IDs to RBSets.
        
        Returns:
            Dict mapping query IDs to corresponding RBSets
        """
        # defaultdict automatically creates a new RBSet() when accessing a new key,
        # eliminating the need for try-catch when adding to a new query_id
        rbsets = defaultdict(RBSet)
        for qrel in self._data:
            rbsets[qrel.query_id].add(qrel.doc_id, qrel.relevance)
        return dict(rbsets)

class TrecHandler:
    """
    Handles reading TREC runs and conversion to RBStar types
    """
    def __init__(self):
        self._data = []
        self._run_name = None

    @property
    def run_name(self) -> str:
        return self._run_name

    def read(self, path: Path | str) -> None:
        """
        Read TREC run file at path into handler.
        
        Args:
            path: Path to TREC run file
            
        Raises:
            AssertionError: If handler already contains data
            ValueError: If no valid run data was read or run names are inconsistent
        """
        assert not self._data, "Handler already contains data"
        path = Path(path)
        
        # Read entire file into memory first
        with path.open() as f:
            lines = f.readlines()

        # Process all lines at once
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                qid, _, docid, rank, score, run_name = line.split()
            except ValueError as e:
                raise ValueError(f"Error parsing line {line_num} in {path}: {line}\n{str(e)}")
            
            self._data.append(ScoredDoc(qid, docid, float(score), int(rank), run_name))
            if self._run_name is None:
                self._run_name = run_name
            elif self._run_name != run_name:
                raise ValueError(f"Inconsistent run names: {self._run_name} != {run_name}")
                    
        if not self._data:
            raise ValueError(f"No valid run data found in {path}")

    def print_stats(self) -> None:
        """Print statistics about run data."""
        print(f"\nRun name: {self._run_name}")
        query_counts = defaultdict(int)
        rank_stats = defaultdict(list)
        score_stats = defaultdict(list)
        for doc in self._data:
            query_counts[doc.query_id] += 1
            rank_stats[doc.query_id].append(doc.rank)
            score_stats[doc.query_id].append(doc.score)
            
        print(f"\nRead {len(self._data)} documents for {len(query_counts)} queries")
        print(f"Average documents per query: {len(self._data)/len(query_counts):.1f}")
        print("Rank ranges per query:")
        for qid in sorted(query_counts.keys()):
            min_rank = min(rank_stats[qid])
            max_rank = max(rank_stats[qid])
            min_score = min(score_stats[qid])
            max_score = max(score_stats[qid])
            print(f"  Query {qid}: ranks {min_rank}-{max_rank}, scores {min_score:.3f}-{max_score:.3f}")

    def to_rbset_dict(self) -> dict[str, RBSet]:
        """
        Convert run data to dictionary mapping query IDs to RBSets.
        
        Returns:
            Dict mapping query IDs to corresponding RBSets
        """
        rbsets = defaultdict(RBSet)
        for doc in self._data:
            rbsets[doc.query_id].add(doc.doc_id, 1)
        return dict(rbsets)

    def to_rbranking_dict(self) -> dict[str, RBRanking]:
        """
        Convert TREC-style ranking data to dictionary of RBRanking objects.
        
        Returns:
            Dict mapping query IDs to corresponding RBRanking objects
        """
        rankings = defaultdict(list)
        
        # Group documents by query_id
        for doc in self._data:
            rankings[doc.query_id].append((doc.doc_id, doc.score))
            
        # Convert to dictionary of RBRankings
        rbrankings  = {}
        for qid, docs in rankings.items():
            # Sort by score (descending) and then by docid (ascending) for consistent tie-breaking
            sorted_docs = sorted(docs, key=lambda x: (-x[1], x[0]))
            # Extract just the document IDs in ranked order
            rbrankings[qid] = RBRanking([[doc_id] for doc_id, _ in sorted_docs])
            
        return rbrankings
