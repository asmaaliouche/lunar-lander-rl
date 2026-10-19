import logging
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

logger = logging.getLogger(__name__)

def train_sb3_dqn(env_id: str = "CartPole-v1", total_timesteps: int = 25000):
    """
    Trains, evaluates, and saves a DQN agent using Stable-Baselines3.
    """
    logger.info(f"--- Training Stable-Baselines3 DQN on {env_id} ---")
    
    # 1. Create the environment
    env = gym.make(env_id)
    
    # 2. Instantiate the DQN model
    # Adjusting hyperparameters for a more stable and faster convergence on CartPole
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        verbose=0,
        tensorboard_log="./logs/"
    )
    
    # 3. Train the model
    # Note: total_timesteps is the number of steps taken in the environment, NOT episodes.
    logger.info(f"Learning for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # 4. Evaluate the model
    logger.info("Evaluating the trained model over 100 episodes...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # 5. Save the model
    save_path = "dqn_cartpole"
    model.save(save_path)
    logger.info(f"Model saved to {save_path}.zip")
    
    env.close()
    return model
