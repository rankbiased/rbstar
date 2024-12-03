import pytest
from rbstar.util import TrecHandler
import os


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """
    Create and cleanup temporary files for mock TREC runs.
    """
    # Setup
    mock_file_1 = "mock_run.txt"
    mock_file_2 = "mock_run_duplicates.txt"

    mock_data_1 = [
        "101 Q0 doc1 1 1.5 run1",
        "101 Q0 doc2 2 0.4 run1", 
        "102 Q0 doc3 1 2.0 run1",
    ]

    mock_data_2 = [
        "101 Q0 doc1 1 1.5 run1",
        "101 Q0 doc2 2 0.4 run1",
        "101 Q0 doc1 3 0.3 run1",  # Duplicate doc1 with different score
    ]

    # Write mock data to temporary files
    with open(mock_file_1, "w") as f:
        f.write("\n".join(mock_data_1))

    with open(mock_file_2, "w") as f:
        f.write("\n".join(mock_data_2))

    yield mock_file_1, mock_file_2

    # Teardown
    if os.path.exists(mock_file_1):
        os.remove(mock_file_1)
    if os.path.exists(mock_file_2):
        os.remove(mock_file_2)


def test_read_and_convert(setup_and_teardown):
    mock_file_1, _ = setup_and_teardown
    handler = TrecHandler()

    handler.read(mock_file_1)

    rbset_dict = handler.to_rbset_dict()

    # Check the contents of the RBSet for query "101"
    rbset_101 = rbset_dict["101"]
    assert "doc1" in rbset_101.positive_set()
    assert "doc2" in rbset_101.positive_set()

    # Check the contents of the RBSet for query "102"
    rbset_102 = rbset_dict["102"]
    assert "doc3" in rbset_102.positive_set()
    assert "doc3" not in rbset_102.negative_set()


def test_no_duplicate_queries(setup_and_teardown):
    _, mock_file_2 = setup_and_teardown
    handler = TrecHandler()

    handler.read(mock_file_2)
    # Validate that duplicates cause an error
    with pytest.raises(AssertionError):
        handler.to_rbset_dict()
