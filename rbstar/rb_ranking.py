from __future__ import annotations
from typing import List, Any

class RBRanking:
    """
    Implements the "ranking" - a sequence of groups, where all elements
    within a group share the same rank. This allows ties to be handled
    explicitly. Each group is a list, so the Ranking is essentially a
    list of lists.
    """

    def __init__(self, groups: list[list] = None):
        """Constructor expects a list of lists"""
        self._groups = groups or []
        
    def append(self, group: List[Any]):
        """Add a new group of tied elements to the ranking"""
        if not isinstance(group, list):
            raise TypeError("Group must be a list.")
        self._groups.append(group)
        
    def __iter__(self):
        """Make the ranking iterable, yielding each group"""
        return iter(self._groups)
        
    def __len__(self):
        """Return the number of groups"""
        return len(self._groups)

    def __getitem__(self, index) -> List[Any]:
        """Return a specific group by index"""
        return self._groups[index]
        
    def __add__(self, other: RBRanking) -> RBRanking:
        """
        Combines two rankings by appending the groups from another ranking.

        Args:
            other: Another RBRanking instance.

        Returns:
            A new RBRanking instance with combined groups.
        """
        if not isinstance(other, RBRanking):
            raise TypeError("Can only add another RBRanking instance.")
        return RBRanking(self._groups + other._groups)

    def total_elements(self) -> int:
        """
        Returns the total number of elements in the ranking
        """
        return sum(len(group) for group in self._groups)

    def validate(self) -> None:
        """
        Validate the groups to ensure that:
          - We have no duplicate elements
          - ** Add conditions as necessary
        """
        all_elements = [elem for group in self._groups for elem in group]
        unique_elements = set(all_elements)
        
        if len(unique_elements) != len(all_elements):
            raise ValueError(
                f"RBRanking cannot contain duplicates. "
                f"Unique elements: {len(unique_elements)}, "
                f"Total elements: {len(all_elements)}"
            )

    def __str__(self):
        """Return a pretty string representation of the ranking"""
        return '\n'.join(f"Group {i+1}: {group}" for i, group in enumerate(self._groups))
