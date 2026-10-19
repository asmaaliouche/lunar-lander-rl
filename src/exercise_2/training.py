import numpy as np
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)

def train_q_learning(env: gym.Env, q_table: np.ndarray, num_episodes: int = 10000, 
                     learning_rate: float = 0.8, discount_factor: float = 0.95, 
                     epsilon: float = 1.0, epsilon_decay: float = 0.995, min_epsilon: float = 0.01) -> np.ndarray:
    """
    Trains an agent using the Q-Learning algorithm.
    
    Args:
        env (gym.Env): The Gymnasium environment.
        q_table (np.ndarray): The initialized Q-table.
        num_episodes (int): The number of training episodes.
        learning_rate (float): The learning rate (alpha).
        discount_factor (float): The discount factor (gamma).
        epsilon (float): The initial exploration rate.
        epsilon_decay (float): The rate at which epsilon decays after each episode.
        min_epsilon (float): The minimum allowable epsilon.
        
    Returns:
        np.ndarray: The trained Q-table.
    """
    logger.info(f"--- Starting Q-Learning Training for {num_episodes} episodes ---")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # 1. Choose action (epsilon-greedy strategy)
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore: random action
            else:
                action = np.argmax(q_table[state, :])  # Exploit: best known action
                
            # 2. Take action and observe the outcome
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # 3. Update the Q-table using the Bellman equation
            # Q(s, a) = Q(s, a) + lr * (reward + gamma * max(Q(s', a')) - Q(s, a))
            max_future_q = np.max(q_table[new_state, :])
            current_q = q_table[state, action]
            
            new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
            q_table[state, action] = new_q
            
            # 4. Transition to the next state
            state = new_state
            
        # Decay epsilon at the end of each episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Log progress periodically
        if (episode + 1) % 2000 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes} completed. Current Epsilon: {epsilon:.3f}")
            
    logger.info("Training complete.")
    return q_table
