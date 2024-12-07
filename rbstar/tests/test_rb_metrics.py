import pytest
from rbstar.rb_metrics import RBMetric, MetricResult
from rbstar.rb_ranking import RBRanking
from rbstar.rb_set import RBSet
from hypothesis import given, strategies as st

@pytest.fixture
def simple_ranking():
    ranking = RBRanking()
    # Add single elements
    for i in range(1, 25):
        ranking.append([i])
    return ranking

@pytest.fixture
def tied_ranking():
    ranking = RBRanking()
    ranking.append([1, 2])
    ranking.append([3, 4])
    return ranking

@pytest.fixture
def simple_set():
    set_obj = RBSet()
    set_obj.add_positive(1)
    set_obj.add_positive(2)
    set_obj.add_negative(3)
    return set_obj

class TestRBPrecision:
    def test_perfect_match(self, simple_ranking, simple_set):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._observation = simple_ranking
        rb_metric._reference = simple_set
        result = rb_metric.rb_precision()
        assert result.lower_bound == pytest.approx(0.744, rel=1e-3), f"Lower bound {result.lower_bound} does not match expected value 0.744"
        assert result.upper_bound == pytest.approx(0.744, rel=1e-3), f"Upper bound {result.upper_bound} does not match expected value 0.744"

    def test_no_match(self, simple_ranking):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._observation = simple_ranking
        ref_set = RBSet()
        ref_set.add_negative(1)
        ref_set.add_negative(2)
        ref_set.add_negative(3)
        rb_metric._reference = ref_set
        result = rb_metric.rb_precision()
        assert result.lower_bound == pytest.approx(0.0), f"Lower bound {result.lower_bound} should be 0.0 for no matches"
        assert result.upper_bound == pytest.approx(0.0), f"Upper bound {result.upper_bound} should be 0.0 for no matches"

    def test_partial_match(self, simple_ranking, simple_set):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._observation = simple_ranking
        rb_metric._reference = simple_set
        result = rb_metric.rb_precision()
        assert result.lower_bound < result.upper_bound, f"Lower bound {result.lower_bound} should be less than upper bound {result.upper_bound}"
        assert result.lower_bound > 0, f"Lower bound {result.lower_bound} should be greater than 0"
        assert result.upper_bound < 1, f"Upper bound {result.upper_bound} should be less than 1"

class TestRBRecall:
    def test_perfect_match(self, simple_ranking):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._reference = simple_ranking
        obs_set = RBSet()
        obs_set.add_positive(1)
        obs_set.add_positive(2)
        obs_set.add_positive(3)
        rb_metric._observation = obs_set
        result = rb_metric.rb_recall()
        assert result.lower_bound == pytest.approx(1.0), f"Lower bound {result.lower_bound} should be 1.0 for perfect match"
        assert result.upper_bound == pytest.approx(1.0), f"Upper bound {result.upper_bound} should be 1.0 for perfect match"

    def test_empty_observation(self, simple_ranking):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._reference = simple_ranking
        rb_metric._observation = RBSet()
        result = rb_metric.rb_recall()
        assert result.lower_bound == pytest.approx(0.0), f"Lower bound {result.lower_bound} should be 0.0 for empty observation"
        assert result.upper_bound == pytest.approx(0.0), f"Upper bound {result.upper_bound} should be 0.0 for empty observation"

class TestRBAlignment:
    def test_identical_rankings(self, simple_ranking):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._observation = simple_ranking
        rb_metric._reference = simple_ranking
        result = rb_metric.rb_alignment()
        assert result.lower_bound == pytest.approx(1.0, abs=0.005), f"Lower bound {result.lower_bound} should be 1.0 for identical rankings"
        assert result.upper_bound == pytest.approx(1.0, abs=0.0001), f"Upper bound {result.upper_bound} should be 1.0 for identical rankings"

    def test_reversed_rankings(self, simple_ranking):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._observation = simple_ranking
        reversed_ranking = RBRanking()
        reversed_ranking.append([3])
        reversed_ranking.append([2])
        reversed_ranking.append([1])
        rb_metric._reference = reversed_ranking
        result = rb_metric.rb_alignment()
        assert result.lower_bound < 0.5, f"Lower bound {result.lower_bound} should be less than 0.5 for reversed rankings"
        assert result.upper_bound < 0.5, f"Upper bound {result.upper_bound} should be less than 0.5 for reversed rankings"

    def test_tied_rankings(self, tied_ranking):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._observation = tied_ranking
        rb_metric._reference = tied_ranking
        result = rb_metric.rb_alignment()
        assert result.lower_bound == pytest.approx(1.0), f"Lower bound {result.lower_bound} should be 1.0 for identical tied rankings"
        assert result.upper_bound == pytest.approx(1.0), f"Upper bound {result.upper_bound} should be 1.0 for identical tied rankings"

class TestRBOverlap:
    def test_identical_rankings(self, simple_ranking):
        rb_metric = RBMetric(phi=0.8)
        rb_metric._observation = simple_ranking
        rb_metric._reference = simple_ranking
        result = rb_metric.rb_overlap()
        assert result.lower_bound == pytest.approx(1.0, abs=0.001), f"Lower bound {result.lower_bound} should be 1.0 for identical rankings"
        assert result.upper_bound == pytest.approx(1.0, abs=0.0001), f"Upper bound {result.upper_bound} should be 1.0 for identical rankings"

    def test_disjoint_rankings(self):
        rb_metric = RBMetric(phi=0.8)
        ranking1 = RBRanking()
        ranking1.append([1, 2, 3])
        ranking2 = RBRanking()
        ranking2.append([4, 5, 6])
        rb_metric._observation = ranking1
        rb_metric._reference = ranking2
        result = rb_metric.rb_overlap()
        assert result.lower_bound == pytest.approx(0.0), f"Lower bound {result.lower_bound} should be 0.0 for disjoint rankings"
        assert result.upper_bound > 0, f"Upper bound {result.upper_bound} should be greater than 0 for potential extended matches"

    def test_partial_overlap(self):
        rb_metric = RBMetric(phi=0.8)
        ranking1 = RBRanking()
        ranking1.append([1, 2])
        ranking1.append([3])
        ranking2 = RBRanking()
        ranking2.append([2, 3])
        ranking2.append([4])
        rb_metric._observation = ranking1
        rb_metric._reference = ranking2
        result = rb_metric.rb_overlap()
        assert 0 < result.lower_bound < result.upper_bound < 1, f"Expected 0 < {result.lower_bound} < {result.upper_bound} < 1 for partial overlap"

def test_invalid_inputs():
    rb_metric = RBMetric(phi=0.8)
    with pytest.raises(AssertionError):
        rb_metric._observation = "invalid"
        rb_metric._reference = simple_ranking
        rb_metric.rb_precision()

    with pytest.raises(AssertionError):
        rb_metric._observation = simple_ranking
        rb_metric._reference = "invalid"
        rb_metric.rb_recall()

class TestRBOFromSIGIRAP24:
    """Test cases for Rank-Biased Overlap from the paper's comparison table."""

    def test_perfect_match(self):
        ref = RBRanking()
        obs = RBRanking()
        for i in range(1, 11):
            ref.append([i])
            obs.append([i])

        # Test phi=0.6
        metric = RBMetric(phi=0.6)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_overlap()
        assert result.lower_bound >= 0.99, f"Lower bound {result.lower_bound} should be at least 0.99 for phi=0.6"
        assert result.upper_bound == pytest.approx(1.00, abs=1e-5), f"Upper bound {result.upper_bound} should be 1.00 for phi=0.6"

        # Test phi=0.7
        metric = RBMetric(phi=0.7)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_overlap()
        assert result.lower_bound >= 0.99, f"Lower bound {result.lower_bound} should be at least 0.99 for phi=0.7"
        assert result.upper_bound == pytest.approx(1.00, abs=1e-5), f"Upper bound {result.upper_bound} should be 1.00 for phi=0.7"

        # Test phi=0.8
        metric = RBMetric(phi=0.8)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_overlap()
        assert result.lower_bound >= 0.969, f"Lower bound {result.lower_bound} should be at least 0.97 for phi=0.8"
        assert result.upper_bound == pytest.approx(1.00, abs=1e-5), f"Upper bound {result.upper_bound} should be 1.00 for phi=0.8"

    def test_adjacent_swaps(self):
        ref = RBRanking()
        obs = RBRanking()
        for i in range(1, 11):
            ref.append([i])
        # [2,1,4,3,6,5,8,7,10,9]
        for i in [2,1,4,3,6,5,8,7,10,9]:
            obs.append([i])
        
        metric = RBMetric(phi=0.8)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_overlap()
        assert result.lower_bound == pytest.approx(0.699, rel=1e-3), f"Lower bound {result.lower_bound} should be 0.699 for adjacent swaps"
        assert result.upper_bound == pytest.approx(0.699, rel=1e-3), f"Upper bound {result.upper_bound} should be 0.699 for adjacent swaps"

    def test_half_reversed(self):
        ref = RBRanking()
        obs = RBRanking()
        for i in range(1, 11):
            ref.append([i])
        # [5,4,3,2,1,10,9,8,7,6]
        for i in [5,4,3,2,1,10,9,8,7,6]:
            obs.append([i])
            
        metric = RBMetric(phi=0.8)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_overlap()
        assert result.lower_bound == pytest.approx(0.458, rel=1e-3), f"Lower bound {result.lower_bound} should be 0.458 for half reversed ranking"
        assert result.upper_bound == pytest.approx(0.458, rel=1e-3), f"Upper bound {result.upper_bound} should be 0.458 for half reversed ranking"

class TestRBAFromSIGIRAP24:
    """Test cases for Rank-Biased Alignment from the paper's comparison table."""

    def test_perfect_match(self):
        ref = RBRanking()
        obs = RBRanking()
        for i in range(1, 11):
            ref.append([i])
            obs.append([i])

        # Test phi=0.6
        metric = RBMetric(phi=0.6)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_alignment()
        assert result.lower_bound >= 0.99, f"Lower bound {result.lower_bound} should be at least 0.99 for phi=0.6"
        assert result.upper_bound == pytest.approx(1.00, abs=1e-5), f"Upper bound {result.upper_bound} should be 1.00 for phi=0.6"

        # Test phi=0.7
        metric = RBMetric(phi=0.7)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_alignment()
        assert result.lower_bound == pytest.approx(0.97, rel=1e-2), f"Lower bound {result.lower_bound} should be 0.97 for phi=0.7"
        assert result.upper_bound == pytest.approx(1.00, rel=1e-5), f"Upper bound {result.upper_bound} should be 1.00 for phi=0.7"

        # Test phi=0.8
        metric = RBMetric(phi=0.8)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_alignment()
        assert result.lower_bound == pytest.approx(0.89, rel=1e-2), f"Lower bound {result.lower_bound} should be 0.89 for phi=0.8"
        assert result.upper_bound == pytest.approx(1.00, rel=1e-5), f"Upper bound {result.upper_bound} should be 1.00 for phi=0.8"

    @given(st.lists(st.integers(min_value=1, max_value=100), 
                    min_size=20, max_size=50, unique=True))
    def test_perfect_match_property(self, elements):
        rb_metric = RBMetric(phi=0.6)
        # Create a ranking from random elements
        test_ranking = RBRanking()
        for element in elements:
            test_ranking.append([element])
            
        rb_metric._observation = test_ranking
        rb_metric._reference = test_ranking
        result = rb_metric.rb_alignment()
        
        # For identical rankings, lower and upper bounds should be equal
        assert result.lower_bound == pytest.approx(result.upper_bound, abs=1e-3), f"Lower bound {result.lower_bound} should approximately equal upper bound {result.upper_bound} for identical rankings"
        # Value should be between 0 and 1
        assert 0.99 <= result.lower_bound <= 1, f"Lower bound {result.lower_bound} should be between 0 and 1"
        assert result.lower_bound == pytest.approx(1.0, abs=1e-3), f"Lower bound {result.lower_bound} should be 1.0 for perfect match"

    def test_adjacent_swaps(self):
        ref = RBRanking()
        obs = RBRanking()
        for i in range(1, 11):
            ref.append([i])
        # [2,1,4,3,6,5,8,7,10,9]
        for i in [2,1,4,3,6,5,8,7,10,9]:
            obs.append([i])
        
        metric = RBMetric(phi=0.8)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_alignment()
        assert result.lower_bound == pytest.approx(0.887, rel=1e-3), f"Lower bound {result.lower_bound} should be 0.887 for adjacent swaps"
        assert result.upper_bound == pytest.approx(0.887, rel=1e-3), f"Upper bound {result.upper_bound} should be 0.887 for adjacent swaps"

    def test_complete_reversal(self):
        ref = RBRanking()
        obs = RBRanking()
        for i in range(1, 11):
            ref.append([i])
        # [10,9,8,7,6,5,4,3,2,1]
        for i in range(10, 0, -1):
            obs.append([i])
            
        metric = RBMetric(phi=0.8)
        metric._observation = obs
        metric._reference = ref
        result = metric.rb_alignment()
        assert result.lower_bound == pytest.approx(0.733, rel=1e-3), f"Lower bound {result.lower_bound} should be 0.733 for complete reversal"
        assert result.upper_bound == pytest.approx(0.733, rel=1e-3), f"Upper bound {result.upper_bound} should be 0.733 for complete reversal"