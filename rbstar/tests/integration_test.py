import subprocess
import tempfile
from pathlib import Path
import json

def test_rbstar_with_rbp_metric():
    """
    Integration test for RBStar using RBP metric with sample data.
    """
    # Create temporary files for observation and reference
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        observation_file = tmpdir_path / "observation.trec"
        reference_file = tmpdir_path / "reference.qrel"

        # Write sample data
        observation_file.write_text("""101 Q0 DOC1 1 1.0 run1
101 Q0 DOC2 2 0.8 run1
102 Q0 DOC3 1 0.9 run1
102 Q0 DOC4 2 0.7 run1
""")
        
        reference_file.write_text("""101 0 DOC1 1
101 0 DOC2 0
102 0 DOC3 1
102 0 DOC4 0
""")

        # Run the RBStar program
        result = subprocess.run([
            "python", "rbstar/__main__.py", "-m", "rbp", "-o", str(observation_file), "-r", str(reference_file), "--json"
        ], capture_output=True, text=True)

        # Assert the program ran successfully
        assert result.returncode == 0, f"Program failed with error: {result.stderr}"

        # Parse the JSON output
        output = json.loads(result.stdout)

        # Check that the metric results are present
        assert "metric" in output
        assert output["metric"] == "RBP"
        assert "runs" in output
        assert "run1" in output["runs"]

        run_results = output["runs"]["run1"]
        assert "lower_bound" in run_results
        assert "upper_bound" in run_results
        assert "residual" in run_results


def test_rbstar_with_latex_output():
    """
    Integration test for RBStar with LaTeX output enabled.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        observation_file = tmpdir_path / "observation.trec"
        reference_file = tmpdir_path / "reference.qrel"

        # Write sample data
        observation_file.write_text("""101 Q0 DOC1 1 1.0 run1
101 Q0 DOC2 2 0.8 run1
102 Q0 DOC3 1 0.9 run1
102 Q0 DOC4 2 0.7 run1
""")

        reference_file.write_text("""101 0 DOC1 1
101 0 DOC2 0
102 0 DOC3 1
102 0 DOC4 0
""")

        # Run the RBStar program
        result = subprocess.run([
            "python", "rbstar/__main__.py", "-m", "rbp", "-o", str(observation_file), "-r", str(reference_file), "--latex"
        ], capture_output=True, text=True)

        # Assert the program ran successfully
        assert result.returncode == 0, f"Program failed with error: {result.stderr}"

        # Verify LaTeX output
        assert "\\begin{tabular}" in result.stdout
        assert "run1" in result.stdout

def test_rbstar_invalid_metric():
    """
    Integration test for RBStar with an invalid metric argument.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        observation_file = tmpdir_path / "observation.trec"
        reference_file = tmpdir_path / "reference.qrel"

        # Write sample data
        observation_file.write_text("""101 Q0 DOC1 1 1.0 run1
""")
        reference_file.write_text("""101 0 DOC1 1
""")

        # Run the RBStar program with an invalid metric
        result = subprocess.run([
            "python", "rbstar/__main__.py", "-m", "invalid_metric", "-o", str(observation_file), "-r", str(reference_file)
        ], capture_output=True, text=True)

        # Assert the program exits with an error
        assert result.returncode != 0, "Program should have failed due to invalid metric"
        assert "invalid choice" in result.stderr

if __name__ == "__main__":
    test_rbstar_with_rbp_metric()
    test_rbstar_with_latex_output()
    test_rbstar_invalid_metric()

    print("All tests passed!")