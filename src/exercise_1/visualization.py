"""
Module for visualizing agent interactions within a Gymnasium environment
using Matplotlib and IPython display logic. Suitable for Jupyter Notebooks.
"""

import logging
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML

logger = logging.getLogger(__name__)

def display_random_agent_interaction(env_id: str = "CartPole-v1", max_steps: int = 100) -> None:
    """
    Runs a random agent in the specified environment and displays it dynamically 
    within a Jupyter Notebook.
    
    Args:
        env_id (str): The ID of the Gymnasium environment.
        max_steps (int): The maximum number of steps to run in a single episode.
    """
    logger.info(f"Visualizing '{env_id}' for a maximum of {max_steps} steps...")
    
    # Initialize the environment with render_mode="rgb_array" for plotting frames
    env = gym.make(env_id, render_mode="rgb_array")
    observation, info = env.reset()
    
    frames = []
    total_reward = 0.0
    
    for step in range(max_steps):
        # Render and capture frame
        frames.append(env.render())
        
        # Sample a random action
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        
        if terminated or truncated:
            logger.info(f"Episode ended early after {step + 1} steps. Total Reward: {total_reward:.2f}")
            break
            
    if not (terminated or truncated):
        logger.info(f"Finished {max_steps} steps without termination. Total Reward: {total_reward:.2f}")
        
    env.close()

    if not frames:
        return

    # Build the animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    img = ax.imshow(frames[0])

    def animate(i):
        img.set_data(frames[i])
        return [img]

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)
    
    # Close the static figure so it doesn't display beneath the animation
    plt.close(fig)
    
    # Display the animation as an interactive HTML widget
    display(HTML(anim.to_jshtml()))
