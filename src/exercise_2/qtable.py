import numpy as np
import gymnasium as gym

def initialize_q_table(env: gym.Env) -> np.ndarray:
    """
    Initializes a Q-table filled with zeros based on the environment's observation and action space.
    
    Args:
        env (gym.Env): The Gymnasium environment.
        
    Returns:
        np.ndarray: A 2D NumPy array representing the Q-table.
    """
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    
    # Create a 2D array initialized to zeros
    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
