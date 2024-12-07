from typing import Any


POSITIVE_CUTOFF = 1 # XXX

class RBSet:
    """
    Implements the "set" - stores a set of positive and negative elements,
    respectively. These sets are actually lists, and should be composed of
    any basic type that can be compared; strings or integers are the
    recommended types.
    """

    def __init__(self, positive: list = None, negative: list = None) -> None:
        self._positive = positive if positive is not None else []
        self._negative = negative if negative is not None else []


    def add(self, elem: Any, rel: int) -> None:
        if rel >= POSITIVE_CUTOFF: 
            self.add_positive(elem)
        else:
            self.add_negative(elem)

    def add_positive(self, elem: Any) -> None:
        if self._positive and not isinstance(elem, type(self._positive[0])):
            print(self._positive)
            raise TypeError(f"Cannot add {type(elem)} to positive list containing {type(self._positive[0])}")
        self._positive.append(elem)

    def add_negative(self, elem: Any) -> None:
        if self._negative and not isinstance(elem, type(self._negative[0])):
            raise TypeError(f"Cannot add {type(elem)} to negative list containing {type(self._negative[0])}")
        self._negative.append(elem)

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
    
    def validate(self) -> None:
        """
        Validate the groups to ensure that:
            - We have no duplicate elements
            - ** Add conditions as necessary
        """
        element_set = set(self._positive).union(set(self._negative))
        element_count = len(self._positive) + len(self._negative)
        # If the length of the set union is different to the number of total
        # elements, then something has gone wrong and we bail out
        assert len(element_set) == element_count, (
            "Error: RBSet cannot contain duplicates. len(element_set) = {}, element_count = {}".format(len(element_set), element_count) )

    def __str__(self) -> str:
        """
        Pretty print representation of the RBSet showing positive and negative elements.
        """
        pos_str = f"Positive elements ({len(self._positive)}): {sorted(self._positive)}"
        neg_str = f"Negative elements ({len(self._negative)}): {sorted(self._negative)}"
        return f"{pos_str}\n{neg_str}"

