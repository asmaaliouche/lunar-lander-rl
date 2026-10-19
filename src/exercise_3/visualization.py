import logging
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)

# Same device logic as training
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def _render_animation(frames, interval=50):
    """Helper function to build and display the HTML animation."""
    if not frames:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    img = ax.imshow(frames[0])

    def animate(i):
        img.set_data(frames[i])
        return [img]

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=interval, blit=True)
    plt.close(fig)
    display(HTML(anim.to_jshtml()))

def display_pytorch_agent(env_id: str, policy_net: torch.nn.Module, max_steps: int = 500) -> None:
    """Visualizes the custom PyTorch DQN agent."""
    logger.info(f"Visualizing PyTorch agent in '{env_id}'...")
    env = gym.make(env_id, render_mode="rgb_array")
    state, info = env.reset()
    frames = []
    total_reward = 0.0

    policy_net.eval() # Set network to evaluation mode

    for step in range(max_steps):
        frames.append(env.render())
        
        # Choose action using the neural network (Exploitation only)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy_net(state_tensor).max(1)[1].item()
            
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        
        if terminated or truncated:
            frames.append(env.render())
            logger.info(f"Episode ended after {step + 1} steps. Total Reward: {total_reward:.2f}")
            break

    env.close()
    _render_animation(frames, interval=50)

def display_sb3_agent(env_id: str, model: BaseAlgorithm, max_steps: int = 500) -> None:
    """Visualizes the Stable-Baselines3 DQN agent."""
    logger.info(f"Visualizing Stable-Baselines3 agent in '{env_id}'...")
    env = gym.make(env_id, render_mode="rgb_array")
    state, info = env.reset()
    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        frames.append(env.render())
        
        # Choose action using the SB3 model (deterministic=True means no random exploration)
        action, _states = model.predict(state, deterministic=True)
        
        state, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        
        if terminated or truncated:
            frames.append(env.render())
            logger.info(f"Episode ended after {step + 1} steps. Total Reward: {total_reward:.2f}")
            break

    env.close()
    _render_animation(frames, interval=50)
