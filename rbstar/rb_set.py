
class RBSet:
    """
    Implements the "set" - stores a set of positive and negative elements,
    respectively. These sets are actually lists, and should be composed of
    any basic type that can be compared; strings or integers are the
    recommended types.
    """

    def __init__(self, positive: list, negative: list) -> None:
        self._positive = positive
        self._negative = negative

    def pos_iter(self) -> Any:
        """
        Iterator for the positive elements
        """
        for element in self._positive:
            yield element

    def neg_iter(self) -> Any:
        """
        Iterator for the negative elements
        """
        for element in self._negative:
            yield element

    def positive_set(self) -> set:
        """
        Returns the positive observations as a set
        """
        return set(self._positive)

    def negative_set(self) -> set:
        """
        Returns the negative observations as a set
        """
        return set(self._negative)


