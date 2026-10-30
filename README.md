# Reinforcement Learning - Eagle-1 Mission

This repository contains a comprehensive exploration of Reinforcement Learning (RL) concepts, progressing from foundational building blocks to training a neural agent capable of piloting a lunar lander. It adheres to standard Data Science and Machine Learning engineering practices, utilizing `Poetry` for dependency management, a clean modular layout, and interactive interfaces.

## Project Structure

```text
├── pyproject.toml         # Poetry dependencies and metadata
├── poetry.lock            # Locked dependencies
├── notebooks/             # Interactive Jupyter Notebooks and media
│   ├── exercise_1.ipynb   # RL building blocks
│   ├── exercise_2.ipynb   # Q-table training
│   ├── exercise_3.ipynb   # DQN training (PyTorch & SB3)
│   ├── mission.ipynb      # Final Lunar Lander Mission (PPO)
│   ├── eagle_landing.mp4  # Video recording of the landing agent
│   └── final_eagle_model.zip # Exported PPO trained model
├── src/                   # Modular source code
│   ├── exercise_1/        # Explorers and random agents
│   ├── exercise_2/        # Q-table implementations and training loops
│   ├── exercise_3/        # PyTorch custom DQN and Stable-Baselines3 wrappers
│   └── mission/           # Productionized Lunar Lander
│       ├── api/app.py       # FastAPI backend serving the trained model
│       └── gui/dashboard.py # Streamlit dashboard for real-time visualization
└── README.md              # Project documentation
```

## Setup & Installation

Ensure you have [Poetry](https://python-poetry.org/) installed.

1. Navigate to the root directory of this project.
2. Install the environment and dependencies:
   ```bash
   poetry install
   ```
3. Activate the environment:
   ```bash
   poetry shell
   ```

## Workflow & Activities

### 1. The Building Blocks (Exercise 1)
Explore the fundamental components of Reinforcement Learning using `Gymnasium`. Run the headless random agent to understand environment spaces, observations, and action spaces.
- **Notebook:** `notebooks/exercise_1.ipynb`

### 2. Q-Table Agent (Exercise 2)
Train a classical RL agent using a Q-table decision matrix. Covers action evaluation, exploration vs. exploitation (Epsilon-Greedy), and state-action value updating.
- **Notebook:** `notebooks/exercise_2.ipynb`

### 3. Neural Brain - DQN (Exercise 3)
Replace the rigid Q-table with a Deep Q-Network (DQN). Explores both a custom implementation using PyTorch and leveraging the `Stable-Baselines3` (SB3) library for a robust implementation.
- **Notebook:** `notebooks/exercise_3.ipynb`

### 4. Mission: Pilot the Lunar Lander (Final Mission)
Train an agent (using PPO from Stable-Baselines3) to land the Eagle-1 module safely. This section is fully productionized with a FastAPI backend and a Streamlit dashboard.
- **Notebook:** `notebooks/mission.ipynb`
- **Trained Model:** `notebooks/final_eagle_model.zip`
- **Video Proof:** `notebooks/eagle_landing.mp4`

## Running the Mission Application

To interact with the final trained Eagle-1 autopilot, you need to launch both the FastAPI backend and the Streamlit frontend.

**1. Start the API (Terminal 1):**
```bash
poetry run uvicorn src.mission.api.app:app --reload
```
The API will be available at http://127.0.0.1:8000.

**2. Start the Dashboard (Terminal 2):**
```bash
poetry run streamlit run src/mission/gui/dashboard.py
```
This will open the Streamlit interface in your browser, allowing you to start the landing sequence and visualize the environment alongside telemetry data in real-time.

## Deliverables Information

For evaluation, the following deliverables have been successfully produced and can be found in this repository:
- **Mission Notebook:** `notebooks/mission.ipynb`
- **Mission Video:** `notebooks/eagle_landing.mp4`
- **API File:** `src/mission/api/app.py`
- **Dashboard File:** `src/mission/gui/dashboard.py`
