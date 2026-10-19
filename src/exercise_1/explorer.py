"""
Module for exploring Gymnasium environments.
Provides functionalities to inspect observation and action spaces.
"""

import logging
import gymnasium as gym

logger = logging.getLogger(__name__)

def explore_environment_spaces(env_id: str = "CartPole-v1") -> gym.Env:
    """
    Creates a Gymnasium environment and logs information about its spaces.
    
    Args:
        env_id (str): The ID of the Gymnasium environment to create.
        
    Returns:
        gym.Env: The created Gymnasium environment instance.
    """
    logger.info(f"--- Exploring Environment: {env_id} ---")
    
    # Create the environment
    env = gym.make(env_id)
    
    # Explore observation space
    logger.info("\n[Observation Space]")
    logger.info(f"Type: {type(env.observation_space)}")
    logger.info(f"Shape: {env.observation_space.shape}")
    sample_obs = env.observation_space.sample()
    logger.info(f"Sample Observation: {sample_obs}")
    
    # Explore action space
    logger.info("\n[Action Space]")
    logger.info(f"Type: {type(env.action_space)}")
    if hasattr(env.action_space, 'n'):
        logger.info(f"Number of actions (discrete): {env.action_space.n}")
    sample_action = env.action_space.sample()
    logger.info(f"Sample Action: {sample_action}")
    
    return env
