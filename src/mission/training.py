import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import logging

logger = logging.getLogger(__name__)

def train_baseline(env_id: str = "LunarLander-v3", timesteps: int = 50000):
    """
    Creates the environment and trains a baseline PPO model.
    """
    env = gym.make(env_id)
    # verbose=0 keeps the output clean
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./logs/mission_eagle/")
    
    logger.info(f"Starting baseline training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, tb_log_name="PPO_Baseline")
    logger.info("Baseline training complete.")
    
    return model, env

def train_optimized(env_id: str = "LunarLander-v3", timesteps: int = 500000, hyperparams: dict = None, run_name: str = "PPO_Optimized"):
    """
    Creates the environment and trains an optimized PPO model using provided hyperparameters.
    """
    if hyperparams is None:
        hyperparams = {}
        
    env = gym.make(env_id)
    
    # Initialize PPO with the given hyperparams
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./logs/mission_eagle/", **hyperparams)
    
    logger.info(f"Starting optimized training run '{run_name}' for {timesteps} timesteps with params: {hyperparams}...")
    model.learn(total_timesteps=timesteps, tb_log_name=run_name)
    logger.info(f"Training run '{run_name}' complete.")
    
    return model, env
def evaluate_agent(model, env, n_episodes: int = 50):
    """
    Evaluates the model over a given number of episodes and logs the performance.
    """
    logger.info(f"Evaluating the model over {n_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    logger.info(f"Performance: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return mean_reward, std_reward
