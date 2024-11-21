
from typing import NamedTuple

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

def read_trec_run(path: str) -> list:
    """
    Take a file path; expect a TREC-formatted run file. Read the run file into
    a list for exploding later.
    """
    run_data = []
    with open(path) as inf:
        for line in inf:
            qid, _, docid, rank, score, run = line.strip().split()
            run_data.append(ScoredDoc(qid, docid, score, rank))
    return run_data

def read_qrels(path: str) -> list:
    """
    Take a file path; expect a qrels file. Read the qrels file into a list for
    exploding later.
    """
    qrels_data = []
    with open(path) as inf:
        for line in inf:
            qid, _, docid, rel = line.split()
            qrels_data.append(Qrel(qid, docid, rel))
    return qrels_data


