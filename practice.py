from collections import deque

import numpy as np


class HistoryManager:
    def __init__(self, maxlen=5):
        # Initialize deque with a maximum length of 5
        self.history = deque(maxlen=maxlen)

    def add(self, new_list):
        """Add a new list to the history."""
        self.history.append(new_list)

    def get_mean(self):
        """Compute the mean of the lists in history."""
        # Ensure history is not empty before computing the mean
        if len(self.history) > 0:
            # Stack all lists in history into a numpy array and compute the mean
            stacked = np.stack(self.history, axis=0)
            return np.mean(stacked, axis=0)
        else:
            return None  # Or any other appropriate default value


# Example usage:
history_manager = HistoryManager(maxlen=5)

# Add lists to the history
history_manager.add(np.array([1, 2, 3]))
history_manager.add(np.array([4, 5, 6]))
history_manager.add(np.array([7, 8, 9]))
history_manager.add(np.array([10, 11, 12]))
history_manager.add(np.array([13, 14, 15]))

# After adding 5 lists, compute the mean
mean_values = history_manager.get_mean()
print("Mean of history:", mean_values)

# Add another list, and the deque will remove the oldest element (maxlen = 5)
history_manager.add(np.array([16, 17, 18]))

# Compute the mean after the history has changed
mean_values = history_manager.get_mean()
print("Updated mean of history:", mean_values)
