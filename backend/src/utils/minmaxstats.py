import collections
from typing import Optional


MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
    """
    Holds and updates the minimum and maximum values observed in a tree search.

    Attributes:
        minimum (float): The smallest value observed.
        maximum (float): The largest value observed.
    """

    def __init__(self, known_bounds: Optional[KnownBounds]) -> None:
        """
        Initializes the MinMaxStats with given known bounds if provided.

        Args:
            known_bounds (Optional[KnownBounds]): A tuple with min and max values.
        """
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float) -> None:
        """
        Updates the minimum and maximum with a new value.

        Args:
            value (float): The new value to update the stats.
        """
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """
        Normalizes a value based on the current min and max values.

        Args:
            value (float): The value to normalize.

        Returns:
            float: The normalized value if bounds are set; otherwise, returns the input value.
        """
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value