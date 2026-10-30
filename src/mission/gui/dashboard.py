import streamlit as st
import requests
import gymnasium as gym
import time
import numpy as np
from PIL import Image

API_URL = "http://127.0.0.1:8000/play"

st.set_page_config(page_title="Mission Eagle Dashboard", layout="wide")

st.title("🚀 Mission Eagle - 1: Autopilot Dashboard")
st.markdown("This dashboard visualizes the performance of the trained PPO agent interacting with the FastAPI backend.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Simulation")
    placeholder = st.empty()
    start_btn = st.button("Start Landing Sequence")

with col2:
    st.subheader("Telemetry")
    step_text = st.empty()
    reward_text = st.empty()
    action_text = st.empty()
    status_text = st.empty()

if start_btn:
    try:
        # Check if API is up
        requests.get("http://127.0.0.1:8000/")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Please ensure the FastAPI server is running (`uvicorn src.mission.api.app:app --reload`).")
        st.stop()
        
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, info = env.reset()
    
    total_reward = 0
    step = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Render the environment
        img_array = env.render()
        img = Image.fromarray(img_array)
        placeholder.image(img, use_container_width=True)
        
        # Get action from API
        state_list = obs.tolist()
        response = requests.post(API_URL, json={"state": state_list})
        
        if response.status_code == 200:
            action = response.json()["action"]
        else:
            st.error(f"API Error: {response.text}")
            break
            
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        # Update telemetry
        step_text.metric("Steps", step)
        reward_text.metric("Total Reward", f"{total_reward:.2f}")
        action_text.metric("Last Action", action)
        
        # Small sleep to make it watchable
        time.sleep(0.02)
        
    env.close()
    
    if total_reward >= 200:
        status_text.success(f"Landing Successful! Final Reward: {total_reward:.2f}")
        st.balloons()
    else:
        status_text.warning(f"Landing Failed. Final Reward: {total_reward:.2f}")
