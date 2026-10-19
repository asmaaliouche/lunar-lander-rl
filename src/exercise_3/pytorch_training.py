import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def optimize_model(policy_net, target_net, optimizer, memory, batch_size=128, gamma=0.99):
    """
    The core training step. It grabs a batch of memories and updates the neural network.
    """
    # 1. We don't train if the memory isn't big enough yet
    if len(memory) < batch_size:
        return
    
    # 2. Sample a random batch of memories
    transitions = memory.sample(batch_size)
    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
    # This converts batch-array of Transitions to Transition of batch-arrays.
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    # 3. Convert arrays to PyTorch Tensors
    state_batch = torch.tensor(batch_state, dtype=torch.float32, device=device)
    action_batch = torch.tensor(batch_action, dtype=torch.int64, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch_reward, dtype=torch.float32, device=device)
    
    # We need to handle next_states carefully. Some next states might be the end of the game (None)
    non_final_mask = torch.tensor(tuple(not d for d in batch_done), device=device, dtype=torch.bool)
    non_final_next_states = torch.tensor([s for s, d in zip(batch_next_state, batch_done) if not d], dtype=torch.float32, device=device)

    # 4. Compute Current Q Values:
    # Pass states through the policy_net. Then use gather() to pick out the specific Q-value 
    # for the action that was actually taken in that memory.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 5. Compute the Target (Expected) Q Values using the Bellman Equation:
    # First, initialize the next state values to 0
    next_state_values = torch.zeros(batch_size, device=device)
    
    # Then, for all states that didn't end the game, use the target_net to predict the best future Q-value
    with torch.no_grad(): # We don't want to calculate gradients for the target_net
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
    # Apply Bellman equation: Target Q = Reward + (Gamma * Expected Future Q)
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # 6. Compute Loss:
    # Compare what the network predicted (state_action_values) vs what it SHOULD have predicted (expected)
    # Huber loss (Smooth L1 Loss) is commonly used in DQN to prevent massive gradients.
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 7. Optimize the model:
    # Standard PyTorch backpropagation steps
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping (prevents exploding gradients)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train_pytorch_dqn(env, policy_net, target_net, num_episodes=300):
    """
    The main loop that plays the game and calls optimize_model().
    """
    logger.info(f"Training Custom PyTorch DQN on {device}...")
    
    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 2000 # Slowed down decay for more exploration
    TAU = 0.005 # Rate at which to update the target network
    LR = 5e-4 # Increased learning rate

    from src.exercise_3.pytorch_dqn import ReplayBuffer
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayBuffer(10000)

    steps_done = 0
    episode_rewards = []

    for i_episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        
        while True:
            # Epsilon-greedy action selection
            sample = random.random()
            # Math formula to smoothly decay epsilon from 0.9 to 0.05
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if sample > eps_threshold:
                with torch.no_grad():
                    # Exploit: feed state to network, pick action with highest Q-value
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = policy_net(state_tensor).max(1)[1].view(1, 1).item()
            else:
                # Explore: random action
                action = env.action_space.sample()

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Store the transition in memory
            memory.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, optimizer, memory, BATCH_SIZE, GAMMA)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(total_reward)
                if (i_episode + 1) % 50 == 0:
                    logger.info(f"Episode {i_episode + 1}/{num_episodes} - Reward: {total_reward} (Epsilon: {eps_threshold:.2f})")
                break
                
    logger.info("Training complete.")
    return episode_rewards
