from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import PPO
import os

app = FastAPI(title="Mission Eagle Autopilot API")

# Load the model
MODEL_PATH = "notebooks/final_eagle_model.zip"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction will fail.")

class StateInput(BaseModel):
    state: list[float]

class ActionOutput(BaseModel):
    action: int

@app.get("/")
def read_root():
    return {"message": "Mission Eagle Autopilot API is running. Send POST requests to /play."}

@app.post("/play", response_model=ActionOutput)
def get_action(input_data: StateInput):
    """
    Receives an 8-dimensional state vector from the LunarLander environment and returns the predicted action.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    if len(input_data.state) != 8:
        raise HTTPException(status_code=400, detail="LunarLander state must be a list of 8 floats.")
        
    # The PPO model expects the state in a specific format
    action, _states = model.predict(input_data.state, deterministic=True)
    
    # action is usually a numpy array, extract the integer
    return {"action": int(action)}
