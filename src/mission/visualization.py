import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import logging

logger = logging.getLogger(__name__)

def record_video(env_id: str, model, max_steps: int = 500):
    """
    Runs the model in the environment and captures the frames for visualization.
    """
    render_env = gym.make(env_id, render_mode="rgb_array")
    logger.info("Recording the agent's flight...")
    
    state, info = render_env.reset()
    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        frames.append(render_env.render())
        
        # Predict the best action (deterministic=True)
        action, _states = model.predict(state, deterministic=True)
        
        # Take the action
        state, reward, terminated, truncated, info = render_env.step(int(action))
        total_reward += float(reward)
        
        if terminated or truncated:
            frames.append(render_env.render())
            logger.info(f"Flight ended after {step + 1} steps. Total Reward: {total_reward:.2f}")
            break

    render_env.close()
    return frames

def display_frames_as_gif(frames):
    """
    Takes a list of frames and displays them as an HTML5 video/gif in the notebook.
    """
    if not frames:
        logger.warning("No frames to display.")
        return
        
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    display(HTML(anim.to_jshtml()))
    plt.close()
    
def show_agent_flight(env_id: str, model, max_steps: int = 500):
    """
    High-level function to record and display the agent's flight.
    """
    frames = record_video(env_id, model, max_steps)
    logger.info("Rendering video...")
    display_frames_as_gif(frames)
