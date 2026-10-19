import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class DQN(nn.Module):
    """
    The Deep Q-Network (DQN) architecture.
    This replaces the Q-Table. Instead of a spreadsheet, we use a neural network
    that takes the state as input and outputs the predicted Q-values for all actions.
    """
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        # Layer 1: Input takes the observation (e.g. 4 values for CartPole)
        self.layer1 = nn.Linear(n_observations, 128)
        # Layer 2: Hidden layer to process complex relationships
        self.layer2 = nn.Linear(128, 128)
        # Layer 3: Output layer gives the Q-value for each possible action (e.g. 2 for CartPole)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines how the data flows through the network.
        """
        # Pass input through layer 1 and apply ReLU activation function
        x = F.relu(self.layer1(x))
        # Pass through layer 2 and apply ReLU
        x = F.relu(self.layer2(x))
        # Pass through final layer (no activation function needed here because Q-values can be any real number)
        return self.layer3(x)


class ReplayBuffer:
    """
    A memory buffer that stores past experiences.
    To train the neural network stably, we randomly sample past experiences
    (state, action, reward, next_state) instead of using consecutive steps.
    """
    def __init__(self, capacity: int):
        # A deque automatically removes the oldest items when it reaches 'maxlen'
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition (an experience tuple) to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Randomly samples a batch of experiences for training."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current number of stored experiences."""
        return len(self.memory)
