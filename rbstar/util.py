from typing import NamedTuple
from rb_ranking import RBRanking
from rb_set import RBSet

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
    def __init__(self) -> None:
        self._data = None

    def read(self, path: str) -> list:
        """
        Take a file path; expect a qrels file. Read the qrels file into
        _data for exploding later.
        """
        assert len(self._data) == 0, (
            "Error: Trying to read into a non-empty QrelHandler" )

        with open(path) as inf:
            for line in inf:
                qid, _, docid, rel = line.split()
                self._data.append(Qrel(qid, docid, rel))

    def to_rbset_dict(self) -> dict:
        """
        Convert the qrels in self._data to a dictionary of query_id -> RBSet
        pairs. 
        """
        output = dict()
        for element in self._data:
            qid = element.query_id
            did = element.doc_id
            rel = element.relevance
            try:
                output[qid].add(did, rel)
            except:
                output[qid] = RBSet()
                output[qid].add(did, rel)
        return output
            

class TrecHandler:
    """
    Handles reading TREC runs and conversion to RBStar types
    """
    def __init__(self) -> None:
        self._data = []

    def read(self, path: str) -> None:
        """
        Take a file path; expect a TREC-formatted run file. Read the run file
        into a list for exploding later.
        """
        assert len(self._data) == 0, (
            "Error: Trying to read into a non-empty TrecHandler" )
        with open(path) as inf:
            for line in inf:
                qid, _, docid, rank, score, _ = line.strip().split()
                self._data.append(ScoredDoc(qid, docid, float(score), int(rank)))


    def to_rbset_dict(self) -> dict:
        """
        Converts the input data into a dictionary of query_id -> RBSet pairs.
        """
        

        rbset_dict = {}
        
        for element in self._data:
            qid = element.query_id
            did = element.doc_id

            if qid not in rbset_dict:
                rbset_dict[qid] = RBSet()
            rbset_dict[qid].add(did, 1)
        
        # Validate each RBSet
        for rbset in rbset_dict.values():
            rbset.validate()

        return rbset_dict
