import math


class RBMetric:

  def __init__(self, phi: float = 0.95) -> None:
        self._phi = phi
        self._observation = None # XXX This should be a collection
        self._reference = None   # XXX This should be a collection


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
        
        # 1. Score based on what we know to be positive
        relevant_set = self._reference.positive_set()
        lb_score = 0.0
        for element in relevant_set:
            if element in observation_weights:
                lb_score += observation_weights[element][1]

        # 2. Score the upper bound by reducing score for known non-rel elements
        nonrelevant_set = self._reference.negative_set()
        ub_score = 1.0
        for element in nonrelevant_set:
            if element in observation_weights:
               ub_score -= observation_weights[element][1]

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

        return (lb_score, lb_score + residual)

    def __rb_alignment_base(self, obs_weight: dict, ref_weight: dict) -> float:
        """
        Computes the base RBA between two dictionaries containing per-element
        weights.
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

    def __extract_tail(self, ranking: RBRanking, weights: dict) -> RBRanking:
        """
        Given a ranking, and a dictionary of element weights, return a new
        ranking that preserves only the elements from the input ranking that
        do not appear in the dictionary.
        """
        tail = RBRanking()
        for group in ranking:
            new_group = [element in group if element not in weights]
            if new_group:
                tail.append(new_group)
        return tail

    def rb_alignment(self) -> tuple[float, float]:
        """
        Metric: ranking | ranking
        Computes: Rank-Biased Alignment score for self._observation, a ranking,
        against self._reference, another ranking.
        Returns: A [lower, upper] bound on the RBR score; upper - lower is
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
        base_score = self.__rb_alignment_base(obs_weights, ref_weights)

        # 2. Compute a maximally productive tail for each list
        obs_tail = self.__extract_tail(self._reference, obs_weights)
        ref_tail = self.__extract_tail(self._observation, ref_weights)

        # 3. Recompute the weights based in the new tails
        obs_weights = self.__calculate_rank_weights(self._observation + obs_tail)
        obs_weights = self.__calculate_rank_weights(self._reference + ref_tail)
        
        # 4. Recompute RBA - now we have residuals
        ub_score = self.__rb_alignment_base(obs_weights, ref_weights)
        
        # 5. Finally, add on a tail residual for everything that may have
        # appeared beyond the end of the most optimistic intersection

        # XXX Check this
        return (base_score, ub_score)
        
   
