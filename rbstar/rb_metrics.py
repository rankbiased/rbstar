import math
from rbstar.rb_ranking import RBRanking
from rbstar.rb_set import RBSet

RB_EPS = 1e-6 # The floating point epsilon value

class RBMetric:

  def __init__(self, phi: float = 0.95) -> None:
        self._phi = phi
        self._observation = None # XXX This should be a collection
        self._reference = None   # XXX This should be a collection


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
            rank += group_size
        return weights


    def rb_precision(self) -> tuple[float, float]:
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
        return (lb_score, ub_score)

    def rb_recall(self) -> tuple[float, float]:
        """
        Metric: set | ranking
        Computes: Rank-Biased Recall score for self._observation, a set,
        against self._reference, a ranking.
        Returns: A [lower, upper] bound on the RBR score; upper - lower is
        the residual, capturing the extent of unknownness due to missing data
        in the reference set.
        """
        assert isinstance(self._observation, RBSet), (
            "RBR requires self._observation to be an RBSet type" )
 
        assert isinstance(self._reference, RBRanking), (
            "RBR requires self._reference to be an RBRanking type" )
 
        reference_weights = self.__calculate_rank_weights(self._reference)
        lb_score = 0.0
        # Compute the weight of the ranking one beyond the length of our
        # reference -- this is for residual computation
        next_weight = (1 - self._phi) * phi**(len(reference_weights) - 1)

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
        return (lb_score, lb_score + residual)
    
    def __extract_missing(self, ranking: RBRanking, weights: dict) -> RBRanking:
        """
        Helper: Given a ranking, and a dictionary of element weights, return
        a new ranking that preserves only the elements from the input ranking
        that **do not** appear in the dictionary.
        """
        tail = RBRanking()
        for group in ranking:
            new_group = [element in group if element not in weights]
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
        for element in obs_weights:
            if element in ref_weight:
                weight_obs = obs_weight[element][1]
                weight_ref = ref_weight[element][1]
                score += math.sqrt(weight_obs * weight_ref)
        return score

    def rb_alignment(self) -> tuple[float, float]:
        """
        Metric: ranking | ranking
        Computes: Rank-Biased Alignment score for self._observation, a ranking,
        against self._reference, another ranking.
        Returns: A [lower, upper] bound on the RBA score; upper - lower is
        the residual, capturing the extent of unknownness due to missing data
        in the reference set.
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
        ub_score = self.__rb_alignment_base(obs_weights, ref_weights)
        
        # 5. Finally, add on a tail residual for everything that may have
        # appeared beyond the end of the most optimistic union; this assumes
        # everything following is perfectly matched
        rank_obs = len(obs_weights)
        ub_score += self._phi ** rank_obs
        
        # As usual, return the base and the upper bound score
        return (base_score, ub_score)
        

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
        obs_group_len = obs.get_count()
        ref_group_len = ref.get_count()
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

        # main loop - we process until both groups are exhausted
        while idx_obs_group < obs_group_len and idx_ref_group < ref_group_len:
            
            obs_group = obs.get_group(idx_obs_group)
            ref_group = ref.get_group(idx_ref_group)
    
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

    def rb_overlap(self) -> tuple[float, float]:
        """
        Metric: ranking | ranking
        Computes: Rank-Biased Overlap score for self._observation, a ranking,
        against self._reference, another ranking.
        Returns: A [lower, upper] bound on the RBO score; upper - lower is
        the residual, capturing the extent of unknownness due to missing data
        in the reference set.
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
        (rbo_base, olap, depth) = self.__rb_overlap_scorer(self._reference,
                                                           self._observation)
        base_tail = self.__rb_overlap_tail_min(depth, overlap)
        rbo_base += base_tail

        # get the ub score now
        (rbo_uppr, olap, depth) = self.__rb_overlap_scorer(self._reference + ref_tail,
                                                           self._observation + obs_tail)
        uppr_tail = self.__rb_overlap_tail_max(depth, olap)
        rbo_uppr += uppr_tail

        return (rbo_base, rbo_uppr)


