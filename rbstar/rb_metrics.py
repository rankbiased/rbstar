import math
from rbstar.rb_ranking import RBRanking
from rbstar.rb_set import RBSet
from dataclasses import dataclass
from typing import Dict

RB_EPS = 1e-6 # The floating point epsilon value

@dataclass
class MetricResult:
    lower_bound: float
    upper_bound: float
    
    @property
    def residual(self) -> float:
        return self.upper_bound - self.lower_bound
        
    def to_dict(self) -> Dict:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "residual": self.residual
        }
    
class RBMetric:

    def __init__(self, phi: float = 0.95) -> None:
        assert 0 < phi < 1, "phi must be between 0 and 1 inclusive"
        self._phi = phi
        self._observation = None  # Will hold either RBRanking or RBSet
        self._reference = None    # Will hold either RBRanking or RBSet


    def __validate_data(self) -> None:
        """
        Calls the validation check on both the observation and the reference;
        Any additional validation can be done here too.
        """
        self._observation.validate()
        self._reference.validate()

    def __calculate_rank_weights(self, ranking: RBRanking) -> dict:
        """
        Given a ranking (containing groups of tied elements), compute the
        fand assign the weight of each element
        """
        assert isinstance(ranking, RBRanking), (
            "ranking needs to be a RBRanking type.")
        
        weights = dict()
        weight = 1 - self._phi
        rank = 1  # Initialize rank counter
        # 1. Iterate each group of elements
        for group in ranking:
            group_weight = 0.0
            # 2. Get the total weight
            for element in group:
                group_weight += weight
                weight = weight * self._phi
            # 3. Set the per-element weight as the average
            group_size = len(group)
            for element in group:
                weights[element] = (rank, group_weight / group_size)
            # 4. Increase the rank according to the group size
            rank += group_size  # Increment rank by size of group
        return weights


    def rb_precision(self) -> MetricResult:
        """
        Metric: ranking | set 
        Computes: Rank-Biased Precision score for self._observation, a ranking,
        against self._reference, a set.
        Returns: A [lower, upper] bound on the RBP score; upper - lower is
        the residual, capturing the extent of unknownness due to missing data
        in the reference set.
        """
        assert isinstance(self._observation, RBRanking), (
            "RBP requires self._observation to be an RBRanking type" )
 
        assert isinstance(self._reference, RBSet), (
            "RBP requires self._reference to be an RBSet type" )
        
        observation_weights = self.__calculate_rank_weights(self._observation)
        
        # 1. Score based on what we know to be positive; that is, accumulate
        # the weights of the positive elements
        relevant_set = self._reference.positive_set()
        lb_score = 0.0
        for element in relevant_set:
            if element in observation_weights:
                lb_score += observation_weights[element][1]

        # 2. Score the upper bound by reducing score for known non-rel elements;
        # that is, start with an upper bound score of 1.0, and then remove the
        # weight associated with each non relevant document
        nonrelevant_set = self._reference.negative_set()
        ub_score = 1.0
        for element in nonrelevant_set:
            if element in observation_weights:
               ub_score -= observation_weights[element][1]

        # We now have the RBP score, and the upper-bound score; the residual
        # can be computed via ub_score - lb_score
        return MetricResult(lb_score, ub_score)

    def rb_recall(self) -> MetricResult:
        """
        Computes the Rank-Biased Recall (RBR) score between an observation set and a reference ranking.
        
        RBR measures how well a set matches a ranking by assigning geometrically decreasing weights
        to elements in the reference ranking. The weight of each element in position i of the reference
        is (1-φ)φ^(i-1), where φ is the persistence parameter that controls how quickly the weights decay.
        
        For example, with φ=0.95:
        - First element has weight 0.05
        - Second element has weight 0.0475 
        - Third element has weight 0.0451
        And so on...
        
        The RBR score is the sum of the weights for elements from the reference that appear in the 
        observation set. This makes RBR particularly suitable for evaluating first-phase retrieval 
        systems that need to identify a set of candidates for a more expensive second-phase ranker.
        
        Key properties:
        - Missing high-ranked elements from the reference is more costly than missing low-ranked ones
        - The score increases as more elements from the reference are included in the observation
        - Returns both a lower and upper bound to account for uncertainty in incomplete rankings
        
        Args:
            self: The RBMetric instance containing:
                - self._observation: An RBSet containing the observation set
                - self._reference: An RBRanking containing the reference ranking
                - self._phi: The persistence parameter (default 0.95)
                
        Returns:
            A tuple of (lower_bound, upper_bound) for the RBR score.
            The difference between bounds represents the residual uncertainty.
            
        Raises:
            AssertionError: If observation is not an RBSet or reference is not an RBRanking
            
        Example:
            If reference=[A,B,C,D] and observation={A,C}:
            - A contributes weight (1-φ)
            - C contributes weight (1-φ)φ^2
            - Total score = (1-φ)(1 + φ^2)
        """
        assert isinstance(self._observation, RBSet), (
            "RBR requires self._observation to be an RBSet type" )
 
        assert isinstance(self._reference, RBRanking), (
            "RBR requires self._reference to be an RBRanking type" )
 
        reference_weights = self.__calculate_rank_weights(self._reference)
        lb_score = 0.0
        residual = 0.0  # Initialize residual to 0
        # Compute the weight of the ranking one beyond the length of our
        # reference -- this is for residual computation
        next_weight = (1 - self._phi) * self._phi**(len(reference_weights) - 1)

        # Iterate over the documents in the observation and tally up the
        # weights for each element; if an element is not present in the
        # reference, we assume that would appear directly after the last
        # element; this is how the residual is computed below.
        for element in self._observation.pos_iter():
            if element in reference_weights:
                lb_score += reference_weights[element][1]
            else:
                residual += next_weight
                next_weight = next_weight * self._phi

        # We return the RBR score, and the upper-bound score
        return MetricResult(lb_score, lb_score + residual)
    
    def __extract_missing(self, ranking: RBRanking, weights: dict) -> RBRanking:
        """
        Helper: Given a ranking, and a dictionary of element weights, return
        a new ranking that preserves only the elements from the input ranking
        that **do not** appear in the dictionary.
        """
        tail = RBRanking()
        for group in ranking:
            new_group = [element for element in group if element not in weights]
            if new_group:
                tail.append(new_group)
        return tail

    def __rb_alignment_scorer(self, obs_weight: dict, ref_weight: dict) -> float:
        """
        Helper: Computes the base RBA between two dictionaries containing
        per-element weights. Only uses the documents in the intersection to
        compute the final score.
        """
        score = 0.0
        # iterate one set, look up the other. We only want to tally up the
        # weight for elements in the intersection.
        for element in obs_weight:
            if element in ref_weight:
                weight_obs = obs_weight[element][1]
                weight_ref = ref_weight[element][1]
                score += math.sqrt(weight_obs * weight_ref)
        return score

    def rb_alignment(self) -> MetricResult:
        """
        Computes the Rank-Biased Alignment (RBA) score between two rankings.
        
        RBA is a novel ranking correlation metric that combines properties of RBP and RBR
        to provide a more nuanced measure of ranking similarity. The score considers both:
        1. The position of elements in the observation ranking
        2. The position of elements in the reference ranking
        
        For each element e that appears in both rankings, its contribution is:
        (1-φ)/φ * φ^((rank_obs(e) + rank_ref(e))/2)
        
        where rank_obs(e) and rank_ref(e) are the positions of element e in the 
        observation and reference rankings respectively.
        
        Key properties:
        - Symmetric: RBA(A,B) = RBA(B,A)
        - Bounded: Scores are between 0 (disjoint) and 1 (identical rankings)
        - Top-weighted: Misalignments at higher ranks cost more than at lower ranks
        - Handles partial rankings: Can compare rankings of different lengths
        - More nuanced than RBO: Distinguishes between different types of misalignments
            - Example: Piecewise rearrangements score better than complete reversals
        
        Implementation details:
        - Computes base score from elements present in both rankings
        - Calculates residuals for elements present in only one ranking
        - Adds tail residual for potential matches beyond observed prefixes
        - Handles ties by sharing weights within tied groups
        
        Args:
            self: The RBMetric instance containing:
                - self._observation: First ranking to compare (RBRanking)
                - self._reference: Second ranking to compare (RBRanking)
                - self._phi: Persistence parameter controlling weight decay
                
        Returns:
            tuple[float, float]: A (lower_bound, upper_bound) pair where:
                - lower_bound: Base RBA score from observed elements
                - upper_bound: Maximum possible score if rankings were extended
                - upper_bound - lower_bound: Residual uncertainty
                
        Raises:
            AssertionError: If either input is not an RBRanking type
            
        Example:
            For rankings A=[1,2,3] and B=[2,1,3] with φ=0.7:
            - Element 1: avg_rank=(2+1)/2=1.5, weight=0.3*0.7^0.5
            - Element 2: avg_rank=(1+2)/2=1.5, weight=0.3*0.7^0.5
            - Element 3: avg_rank=(3+3)/2=3.0, weight=0.3*0.7^2
            Base score = sum of weights
        
        Note:
            The choice of φ affects the degree of top-weightedness. For rankings of 
            length k, φ = √(f)^(1/k) gives a score ratio of f between perfect alignment
            and complete reversal.
        """
        assert isinstance(self._observation, RBRanking), (
            "RBR requires self._observation to be an RBRanking type" )
 
        assert isinstance(self._reference, RBRanking), (
            "RBR requires self._reference to be an RBRanking type" )

        # 1. Compute the "base" RBA via the intersection of the lists
        obs_weights = self.__calculate_rank_weights(self._observation)
        ref_weights = self.__calculate_rank_weights(self._reference)
        base_score = self.__rb_alignment_scorer(obs_weights, ref_weights)

        # 2. Compute the upper bound score by extending each of obs and
        # ref using the most productive tail so that both obs and ref
        # account for all of the items in their union
        obs_tail = self.__extract_missing(self._reference, obs_weights)
        ref_tail = self.__extract_missing(self._observation, ref_weights)

        # 3. Recompute the weights based on the new tails
        obs_weights = self.__calculate_rank_weights(self._observation + obs_tail)
        obs_weights = self.__calculate_rank_weights(self._reference + ref_tail)
        
        # 4. Recompute RBA - now we have residuals
        ub_score = self.__rb_alignment_scorer(obs_weights, ref_weights)
        
        # 5. Finally, add on a tail residual for everything that may have
        # appeared beyond the end of the most optimistic union; this assumes
        # everything following is perfectly matched
        rank_obs = len(obs_weights)
        ub_score += self._phi ** rank_obs
        
        # As usual, return the base and the upper bound score
        return MetricResult(base_score, ub_score)
        

    def __rb_overlap_combine(self, obs_val: float, ref_val: float) -> float:
        """
        Helper: Determines the value given according to the selected
        tie-breaking scheme at hand.
        See: Corsi & Urbano, SIGIR 2024: https://doi.org/10.1145/3626772.3657700
        """
        return obs_val * ref_val

    def __rb_overlap_tail_min(self, depth: int, overlap: int) -> float:
        """
        Helper: Computes the tail sum from depth to infinity with a fixed
        overlap.
        See: Eqn 11 of Webber et al: https://doi.org/10.1145/1852102.1852106
        """
        tail = (1 - self._phi) / self._phi * overlap * math.log(1.0 / (1.0 - self._phi))
        weight = (1 - self._phi) / self._phi
        for rank in range(1, depth + 1):
            weight = weight * self._phi
            term = weight * overlap / rank
            tail = tail - term
        return tail

    def __rb_overlap_tail_max(self, depth: int, overlap: int) -> float:
        """
        Helper: Computes the tail sum of the geometric sequence. Requires
        depth and overlap to be equal, as we're assuming all elements match
        up to this point.
        """
        assert (depth == overlap), (
            "Depth and Overlap must be equal to compute the RBO tail maximum.")
        return self._phi ** depth

    def __rb_overlap_scorer(self, obs: RBRanking, ref: RBRanking) -> tuple[float, int, int]:
        """
        Helper: Given an observation and a reference, both RBRankings, compute
        the RBO score, returning the base score, the overlap, and depth.
        This is not the outward facing RBO function as it needs to handle
        both lower and upper bounds, but this is the RBO computation itself.
        """

        # Set up data to handle iteration of groups
        obs_group_len = len(obs)
        ref_group_len = len(ref)
        idx_obs_group = 0
        idx_ref_group = 0
        # and within groups
        idx_obs = 0
        idx_ref = 0
       
        # Set up dictionaries that handle the "inclusion at depth d" counts;
        # both dictionaries contain the union of elements from obs and ref
        # at the beginning of the main loop below
        obs_count = dict()
        ref_count = dict()
        for group in obs:
            for item in group:
                obs_count[item] = 0
                ref_count[item] = 0
        for group in ref:
            for item in group:
                obs_count[item] = 0
                ref_count[item] = 0

        # prepare for the main loop
        weight = 1 - self._phi
        score = 0.0
        depth = 0
        olap = 0

        # Initialize indices before the loop
        cur_obs_idx = 0
        cur_ref_idx = 0

        # main loop - we process until both groups are exhausted
        while idx_obs_group < obs_group_len and idx_ref_group < ref_group_len:
            
            obs_group = obs[idx_obs_group]
            ref_group = ref[idx_ref_group]
    
            # get a status on all active items - the union of both groups
            active = set(obs_group + ref_group)
            old_olap = 0
            for item in active:
                old_olap += self.__rb_overlap_combine(obs_count[item], ref_count[item])
            
            # shift to a new status
            cur_obs_idx += 1
            cur_ref_idx += 1

            for item in obs_group:
                obs_count[item] = cur_obs_idx / len(obs_group)
            for item in ref_group:
                ref_count[item] = cur_ref_idx / len(ref_group)

            new_olap = 0
            for item in active:
                new_olap += self.__rb_overlap_combine(obs_count[item], ref_count[item])

            olap += (new_olap - old_olap)

            depth += 1
            contrib = olap / depth * weight
            score += contrib
            weight *= self._phi

            # We need to move to a new group now
            if cur_obs_idx == len(obs_group):
                cur_obs_idx = 0
                idx_obs_group += 1

            if cur_ref_idx == len(ref_group):
                cur_ref_idx = 0
                idx_ref_group += 1
               
        return (score, olap, depth)

    def rb_overlap(self) -> MetricResult:
        """
        Computes the Rank-Biased Overlap (RBO) score between two rankings.
        
        RBO is a top-weighted correlation metric that measures similarity between two rankings
        while accommodating:
        1. Different ranking lengths
        2. Rankings that are not permutations of each other
        3. Incomplete rankings with missing elements
        
        The score is based on the overlap between prefixes of the two rankings at different depths,
        with geometrically decreasing weights controlled by the persistence parameter φ:
        
        RBO = (1-φ)/φ * sum(φ^d * |overlap_at_depth_d| / d)
        
        where d is the depth and overlap_at_depth_d is the number of common elements 
        in the first d positions of both rankings.
        
        Key properties:
        - Symmetric: RBO(A,B) = RBO(B,A)
        - Bounded: Scores are between 0 (disjoint) and 1 (identical rankings)
        - Top-weighted: Disagreements at higher ranks cost more than at lower ranks
        - Handles ties: Elements in the same group/tier are considered tied
        
        Implementation details:
        - Processes both rankings group by group to handle ties
        - Computes base score from available prefixes
        - Calculates residuals to bound the possible scores if rankings were extended
        - Handles extrapolation tails for elements seen in one ranking but not the other
        
        Args:
            self: The RBMetric instance containing:
                - self._observation: First ranking to compare (RBRanking)
                - self._reference: Second ranking to compare (RBRanking)
                - self._phi: Persistence parameter controlling weight decay
                
        Returns:
            tuple[float, float]: A (lower_bound, upper_bound) pair where:
                - lower_bound: Minimum possible RBO score given available data
                - upper_bound: Maximum possible RBO score if rankings were extended
                - upper_bound - lower_bound: Residual indicating score uncertainty
                
        Raises:
            AssertionError: If either input is not an RBRanking type
            
        Example:
            For rankings A=[1,2,3] and B=[1,3,2] with φ=0.8:
            - At depth 1: overlap=1/1, contribution=(1-0.8)*1=0.2
            - At depth 2: overlap=1/2, contribution=(1-0.8)*0.8*0.5=0.08
            - At depth 3: overlap=3/3, contribution=(1-0.8)*0.64*1=0.128
            Base score = 0.408 (plus residual for possible extensions)
        """
        assert isinstance(self._observation, RBRanking), (
              "RBO requires self._observation to be an RBRanking type" )

        assert isinstance(self._reference, RBRanking), (
              "RBO requires self._reference to be an RBRanking type" )

        # For a set of items that have been seen in each ranking so we can
        # compute the completion tails
        obs_seen = set()
        ref_seen = set()
        for group in self._observation:
            for elem in group:
                obs_seen.add(elem)
        for group in self._reference:
            for elem in group:
                ref_seen.add(elem)

        
        # form the tails for later
        obs_tail = self.__extract_missing(self._reference, obs_seen)
        ref_tail = self.__extract_missing(self._observation, ref_seen)

        # get the lb RBO score
        (rbo_base, overlap, depth) = self.__rb_overlap_scorer(self._reference,
                                                           self._observation)
        base_tail = self.__rb_overlap_tail_min(depth, overlap)
        rbo_base += base_tail

        # get the ub score now
        (rbo_uppr, olap, depth) = self.__rb_overlap_scorer(self._reference + ref_tail,
                                                           self._observation + obs_tail)
        uppr_tail = self.__rb_overlap_tail_max(depth, olap)
        rbo_uppr += uppr_tail

        return MetricResult(rbo_base, rbo_uppr)
