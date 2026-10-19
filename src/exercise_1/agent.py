"""
Module for running agent interactions within a Gymnasium environment.
Contains functions for executing interaction loops.
"""

import logging
import gymnasium as gym

logger = logging.getLogger(__name__)

def run_random_agent(env: gym.Env, num_episodes: int = 10) -> None:
    """
    Executes a random agent in the given environment for a specified number of episodes.
    
    Args:
        env (gym.Env): The Gymnasium environment to run the agent in.
        num_episodes (int): The number of episodes to run.
    """
    logger.info(f"\n--- Running Random Agent for {num_episodes} Episodes ---")
    
    for episode in range(1, num_episodes + 1):
        # 1. Reset the environment at the beginning of each episode
        observation, info = env.reset()
        
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # 2. Run the episode until termination or truncation
        while not (terminated or truncated):
            # Sample a random action from the action space
            action = env.action_space.sample()
            
            # Take a step in the environment using the sampled action
            # The step function returns 5 values in Gymnasium
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Accumulate the reward
            total_reward += float(reward)
            
        logger.info(f"Episode {episode:2d} finished with Total Reward: {total_reward:.2f}")

    # Ensure the environment is properly closed
    env.close()
    logger.info("Environment closed.")
