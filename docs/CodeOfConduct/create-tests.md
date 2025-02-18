# Code Standards for Testing

## Table of Contents

- [Code Standards for Testing](#code-standards-for-testing)
  - [Table of Contents](#table-of-contents)
  - [Testing Example: Replay Buffer](#testing-example-replay-buffer)
    - [Example Implementation: ReplayBuffer](#example-implementation-replaybuffer)
    - [Writing Tests for Replay Buffer](#writing-tests-for-replay-buffer)
    - [Test Cases](#test-cases)
  - [Running the tests](#running-the-tests)

When implementing a module, it is mandatory to write tests that confirm correctness.
The key philosophy: *"If your code runs but behaves incorrectly, your training will waste hours before you even realize there's a bug."*
With unit tests, we can prevent wasted debugging time and quickly verify that changes don’t break existing functionality.

## Testing Example: Replay Buffer

A Replay Buffer stores past experiences so the model can sample and learn from them

### Example Implementation: ReplayBuffer

A minimal replay buffer implementation in `backend/src/replay_buffer.py`

```python
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""
    def __init__(self, capacity: int):
        self.capacity = capacity  # Maximum buffer size
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Stores an experience in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Samples a batch of experiences randomly."""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer!")
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.buffer)
```

### Writing Tests for Replay Buffer

Tests should verify that:

1. Adding elements actually stores experiences.
2. Sampling works correctly, including edge cases (empty buffer, too small buffer).
3. Buffer capacity is respected (old elements are removed when full).
4. Data integrity is maintained when adding/sampling.

### Test Cases

In backend/tests/test_replay_buffer.py:

```python
import pytest
import numpy as np
from backend.src.replay_buffer import ReplayBuffer

def test_add_and_length():
    """Test that adding elements increases the buffer size correctly."""
    buffer = ReplayBuffer(capacity=5)
    assert len(buffer) == 0  # Initially empty

    state = np.array([0.0, 1.0])
    action = 1
    reward = 1.0
    next_state = np.array([1.0, 0.0])
    done = False

    buffer.add(state, action, reward, next_state, done)
    assert len(buffer) == 1  # Buffer should have one element

def test_sample():
    """Test that sampling returns the correct number of elements."""
    buffer = ReplayBuffer(capacity=10)
    
    # Add 10 experiences
    for i in range(10):
        state = np.array([i, i+1])
        action = i % 2
        reward = float(i)
        next_state = np.array([i+1, i+2])
        done = i % 3 == 0
        buffer.add(state, action, reward, next_state, done)

    batch = buffer.sample(batch_size=5)
    assert len(batch) == 5  # Sampling 5 should return exactly 5 elements
    assert all(len(sample) == 5 for sample in batch)  # Each sample should have 5 elements

def test_sample_raises_error_if_not_enough_data():
    """Test that sampling more elements than available raises an error."""
    buffer = ReplayBuffer(capacity=5)
    
    # Add only 3 experiences
    for i in range(3):
        state = np.array([i, i+1])
        buffer.add(state, i, i, state, False)

    with pytest.raises(ValueError):
        buffer.sample(batch_size=5)  # Not enough elements in buffer

def test_capacity_limit():
    """Test that buffer does not exceed its capacity."""
    buffer = ReplayBuffer(capacity=3)
    
    # Add 5 elements, only 3 should remain due to capacity limit
    for i in range(5):
        state = np.array([i, i+1])
        buffer.add(state, i, i, state, False)

    assert len(buffer) == 3  # Old elements should be removed

def test_data_integrity():
    """Test that the sampled data is identical to what was stored."""
    buffer = ReplayBuffer(capacity=10)
    
    state = np.array([1, 2])
    action = 2
    reward = 3.0
    next_state = np.array([2, 3])
    done = True

    buffer.add(state, action, reward, next_state, done)
    sample = buffer.sample(batch_size=1)[0]

    assert np.array_equal(sample[0], state)
    assert sample[1] == action
    assert sample[2] == reward
    assert np.array_equal(sample[3], next_state)
    assert sample[4] == done
```

## Running the tests

To run the test suite, execute the following command from the root directory of the project:

```bash
docker compose run backend pytest
```

To run the tests using local Python installation, execute the following command from the root directory of the project:

```bash
pytest backend/tests
```
