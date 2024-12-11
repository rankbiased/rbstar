from typing import Any, Iterator, List, Set


POSITIVE_CUTOFF = 1 # XXX

class RBSet:
    """
    Implements the "set" - stores a set of positive and negative elements,
    respectively. These sets are actually lists, and should be composed of
    any basic type that can be compared; strings or integers are the
    recommended types.
    """

    def __init__(self, positive: list = None, negative: list = None) -> None:
        self._positive = positive or []
        self._negative = negative or []


    def add(self, elem: Any, rel: int) -> None:
        """
        Adds an element to the positive or negative list based on the relation value.

        Args:
            elem: The element to add.
            rel: The relation value (>= POSITIVE_CUTOFF for positive).

        Raises:
            TypeError: If the type of the element does not match the existing elements in the list.
        """
        if rel >= POSITIVE_CUTOFF: 
            self.add_positive(elem)
        else:
            self.add_negative(elem)

    def add_positive(self, elem: Any) -> None:
        """
        Adds an element to the positive list.

        Args:
            elem: The element to add.

        Raises:
            TypeError: If the type of the element does not match the existing elements in the positive list.
        """
        self._validate_type(self._positive, elem, "positive")
        self._positive.append(elem)

    def add_negative(self, elem: Any) -> None:
        """
        Adds an element to the negative list.

        Args:
            elem: The element to add.

        Raises:
            TypeError: If the type of the element does not match the existing elements in the negative list.
        """
        self._validate_type(self._negative, elem, "negative")
        self._negative.append(elem)

    def _validate_type(self, elements: List[Any], elem: Any, list_name: str) -> None:
        """
        Validates that the type of the element matches the existing elements in the list.

        Args:
            elements: The list to validate against.
            elem: The element to validate.
            list_name: Name of the list (for error messages).

        Raises:
            TypeError: If the type of the element does not match the existing elements.
        """
        if elements and not isinstance(elem, type(elements[0])):
            raise TypeError(
                f"Cannot add {type(elem)} to {list_name} list containing {type(elements[0])}"
            )
            
    def pos_iter(self) -> Iterator[Any]:
        """
        Iterator for the positive elements
        """
        return iter(self._positive)

    def neg_iter(self) -> Iterator[Any]:
        """
        Iterator for the negative elements
        """
        return iter(self._negative)

    def positive_set(self) -> Set[Any]:
        """
        Returns the positive observations as a set
        """
        return set(self._positive)

    def negative_set(self) -> Set[Any]:
        """
        Returns the negative observations as a set
        """
        return set(self._negative)
    
    def total_elements(self) -> int:
        """
        Returns the total number of elements in the set
        """
        return len(self._positive) + len(self._negative)

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

