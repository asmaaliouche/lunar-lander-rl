import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)

def evaluate_agent(env: gym.Env, q_table: np.ndarray, num_episodes: int = 100) -> float:
    """
    Evaluates a trained agent by exploiting the provided Q-table.
    No exploration is performed during evaluation.
    
    Args:
        env (gym.Env): The Gymnasium environment.
        q_table (np.ndarray): The trained Q-table.
        num_episodes (int): The number of evaluation episodes.
        
    Returns:
        float: The success rate (percentage of successful episodes).
    """
    logger.info(f"\n--- Evaluating Agent over {num_episodes} episodes ---")
    total_wins = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Pure exploitation: always choose the best known action
            action = np.argmax(q_table[state, :])
            
            # Take action
            state, reward, terminated, truncated, info = env.step(action)
            
            # If the episode ends with a reward of 1.0, it's a win
            if terminated and reward == 1.0:
                total_wins += 1
                
    success_rate = (total_wins / num_episodes) * 100
    logger.info(f"Success rate over {num_episodes} episodes: {success_rate:.1f}%")
    
    return success_rate
