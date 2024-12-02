from typing import NamedTuple
from collections import defaultdict
from pathlib import Path
from .rb_ranking import RBRanking 
from .rb_set import RBSet

# Use the ScoredDoc and Qrel types from ir_measures, but extend ScoredDoc
# with a rank attribute. 
# https://github.com/terrierteam/ir_measures/blob/main/ir_measures/util.py
class Qrel(NamedTuple):
    query_id: str
    doc_id: str
    relevance: int
    iteration: str = '0'

class ScoredDoc(NamedTuple):
    query_id: str
    doc_id: str
    score: float
    rank: int

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
        """
        if self._data:
            raise AssertionError("Cannot read into non-empty QrelHandler")

        path = Path(path)
        with path.open() as f:
            for line in f:
                qid, _, docid, rel = line.split()
                self._data.append(Qrel(qid, docid, int(rel)))

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

    def read(self, path: Path | str) -> None:
        """
        Read TREC run file at path into handler.
        
        Args:
            path: Path to TREC run file
            
        Raises:
            AssertionError: If handler already contains data
        """
        if self._data:
            raise AssertionError("Cannot read into non-empty TrecHandler")

        path = Path(path)
        with path.open() as f:
            for line in f:
                qid, _, docid, rank, score, _ = line.strip().split()
                self._data.append(ScoredDoc(qid, docid, float(score), int(rank)))

    def to_rbset_dict(self) -> dict[str, RBSet]:
        """
        Convert run data to dictionary mapping query IDs to RBSets.
        
        Returns:
            Dict mapping query IDs to corresponding RBSets
        """
        rbsets = defaultdict(RBSet)
        for doc in self._data:
            print(doc)
            rbsets[doc.query_id].add(doc.doc_id, 1)

        # Validate all RBSets
        for rbset in rbsets.values():
            print(rbset)
            rbset.validate()

        return dict(rbsets)
