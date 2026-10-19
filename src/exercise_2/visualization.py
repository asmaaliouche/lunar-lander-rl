import logging
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import numpy as np

logger = logging.getLogger(__name__)

def display_trained_agent_interaction(env_id: str, q_table: np.ndarray, max_steps: int = 100) -> None:
    """
    Visualizes a trained agent interacting with the environment using a Q-Table.
    Displays dynamically within a Jupyter Notebook using HTML animation.
    
    Args:
        env_id (str): The ID of the Gymnasium environment.
        q_table (np.ndarray): The trained Q-table.
        max_steps (int): The maximum number of steps to run in the visualization.
    """
    logger.info(f"Visualizing trained agent in '{env_id}'...")
    
    # Initialize the environment with render_mode="rgb_array" for plotting frames
    env = gym.make(env_id, render_mode="rgb_array")
    state, info = env.reset()
    
    frames = []
    total_reward = 0.0
    
    for step in range(max_steps):
        # Render and capture frame
        frames.append(env.render())
        
        # Exploit the Q-table to find the best action
        action = np.argmax(q_table[state, :])
        
        # Step the environment
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        
        if terminated or truncated:
            frames.append(env.render()) # Capture the final frame (goal or hole)
            logger.info(f"Episode ended after {step + 1} steps. Total Reward: {total_reward:.2f}")
            break
            
    if not (terminated or truncated):
        logger.info(f"Finished {max_steps} steps without termination. Total Reward: {total_reward:.2f}")
        
    env.close()

    if not frames:
        return

    # Build the animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    img = ax.imshow(frames[0])

    def animate(i):
        img.set_data(frames[i])
        return [img]

    # For FrozenLake, 200ms interval looks better to see discrete steps
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=200, blit=True)
    
    # Close the static figure so it doesn't display beneath the animation
    plt.close(fig)
    
    # Display the animation as an interactive HTML widget
    display(HTML(anim.to_jshtml()))
