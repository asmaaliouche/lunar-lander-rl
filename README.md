# Reinforcement Learning Basics

This project introduces the foundational concepts of Reinforcement Learning using `Gymnasium`. It adheres to standard Data Science and Machine Learning engineering practices, utilizing `Poetry` for dependency management and a clean, modular source code layout.

## Project Structure

```text
├── pyproject.toml      # Poetry dependencies and metadata
├── poetry.lock         # Locked dependencies
├── src/                # Modular source code
│   ├── __init__.py
│   ├── explorer.py     # Functions to inspect environment spaces
│   ├── agent.py        # Random agent interaction loop
│   └── visualization.py# Utility to visualize frames in Jupyter
└── notebooks/
    └── exercise_1.ipynb # Interactive Jupyter Notebook for Exercise 1
```

## Setup & Installation

Ensure you have [Poetry](https://python-poetry.org/) installed.

1. Navigate to the root directory of this project.
2. Install the environment and dependencies (which include Jupyter):
   ```bash
   poetry install
   ```

## Running the Exercises

### Exercise 1: Discover the building blocks of Reinforcement Learning

The exercise is now interactive and runs in a Jupyter Notebook to support visual rendering of the environment.

1. Launch Jupyter from the Poetry environment:
   ```bash
   poetry run jupyter notebook
   ```
2. Open `notebooks/exercise_1.ipynb`.
3. Run the cells to:
   - Explore the `CartPole-v1` spaces.
   - Run the headless random agent for 10 episodes.
   - Watch the agent perform using Matplotlib inline visualization.
