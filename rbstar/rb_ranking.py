from __future__ import annotations

class RBRanking:
    """
    Implements the "ranking" - a sequence of groups, where all elements
    within a group share the same rank. This allows ties to be handled
    explicitly. Each group is a list, so the Ranking is essentially a
    list of lists.
    """

    def __init__(self, glists: list = []) -> None:
        self._lists = glists

    def __add__(self, other: RBRanking) -> RBRanking:
        for group in other:
            self.add_group(group)
        return self

    def __iter__(self) -> list:
        """
        Iterate the sequence of groups, one at a time
        """
        for group in self._lists:
            yield group
    
    def add_group(self, group: list) -> None:
        """
        Add a new group to our current sequence
        """
        self._lists.append(group)
    

