from __future__ import annotations

class RBRanking:
    """
    Implements the "ranking" - a sequence of groups, where all elements
    within a group share the same rank. This allows ties to be handled
    explicitly. Each group is a list, so the Ranking is essentially a
    list of lists.
    """

    def __init__(self, groups: list = None):
        self._lists = groups if groups is not None else []
        
    def append(self, group):
        """Add a new group of tied elements to the ranking"""
        self._lists.append(group)
        
    def __iter__(self):
        """Make the ranking iterable, yielding each group"""
        return iter(self._lists)
        
    def __len__(self):
        """Return the number of groups"""
        return len(self._lists)

    def __getitem__(self, index):
        """Return a specific group by index"""
        return self._lists[index]
        
    def __add__(self, other: RBRanking) -> RBRanking:
        return RBRanking(self._lists + other._lists)

    def total_elements(self) -> int:
        """
        Returns the total number of elements in the ranking
        """
        return len([elem for group in self._lists for elem in group])

    def validate(self) -> None:
        """
        Validate the groups to ensure that:
          - We have no duplicate elements
          - ** Add conditions as necessary
        """
        all_elements = [elem for group in self._lists for elem in group]
        unique_elements = set(all_elements)
        
        if len(unique_elements) != len(all_elements):
            raise ValueError(
                f"RBRanking cannot contain duplicates. "
                f"Unique elements: {len(unique_elements)}, "
                f"Total elements: {len(all_elements)}"
            )

    def __str__(self):
        """Return a pretty string representation of the ranking"""
        return '\n'.join(f"Group {i+1}: {group}" for i, group in enumerate(self._lists))
