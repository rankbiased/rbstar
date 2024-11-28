import unittest
from util import TrecHandler
import os


class TestTrecHandler(unittest.TestCase):
    def setUp(self):
        """
        Create temporary files for mock TREC runs before each test.
        """
        self.mock_file_1 = "mock_run.txt"
        self.mock_file_2 = "mock_run_duplicates.txt"

        # Mock data for testing
        self.mock_data_1 = [
            "101 Q0 doc1 1 1.5 run1",
            "101 Q0 doc2 2 0.4 run1",
            "102 Q0 doc3 1 2.0 run1",
        ]

        self.mock_data_2 = [
            "101 Q0 doc1 1 1.5 run1",
            "101 Q0 doc2 2 0.4 run1",
            "101 Q0 doc1 3 0.3 run1",  # Duplicate doc1 with different score
        ]

        # Write mock data to temporary files
        with open(self.mock_file_1, "w") as f:
            f.write("\n".join(self.mock_data_1))

        with open(self.mock_file_2, "w") as f:
            f.write("\n".join(self.mock_data_2))

    def tearDown(self):
        """
        Clean up temporary files after each test.
        """
        if os.path.exists(self.mock_file_1):
            os.remove(self.mock_file_1)
        if os.path.exists(self.mock_file_2):
            os.remove(self.mock_file_2)

    def test_read_and_convert(self):
        handler = TrecHandler()

        handler.read(self.mock_file_1)

        rbset_dict = handler.to_rbset_dict()

        # Check the contents of the RBSet for query "101"
        rbset_101 = rbset_dict["101"]
        self.assertIn("doc1", rbset_101.positive_set())
        self.assertIn("doc2", rbset_101.positive_set())

        # Check the contents of the RBSet for query "102"
        rbset_102 = rbset_dict["102"]
        self.assertIn("doc3", rbset_102.positive_set())
        self.assertNotIn("doc3", rbset_102.negative_set())

    def test_no_duplicate_queries(self):
        handler = TrecHandler()

        handler.read(self.mock_file_2)
        # Validate that duplicates cause an error
        with self.assertRaises(AssertionError):
            handler.to_rbset_dict()

if __name__ == "__main__":
    unittest.main()
