import gymnasium as gym
from stable_baselines3 import DQN
import os

from gas_well_env import GasWellEnv 

MODEL_PATH = "gas_well_dqn_final_model.zip" 
BEST_MODEL_DIR = "./tmp_well_model/"
NUM_TEST_EPISODES = 5

gym.envs.registration.register(
    id='GasWell-v0',
    entry_point='gas_well_env:GasWellEnv', 
    max_episode_steps=100,
)

if not os.path.exists(MODEL_PATH) and not os.path.exists(os.path.join(BEST_MODEL_DIR, "best_model.zip")):
    print("Error: No trained model found!")
    print(f"Please run 'train_agent.py' first to train and save a model.")
    exit()

test_env = gym.make('GasWell-v0', render_mode='human')

model_to_load = MODEL_PATH
best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
if os.path.exists(best_model_path):
    model_to_load = best_model_path
    print(f"Loading BEST model from: {model_to_load}")
else:
    print(f"Loading FINAL model from: {model_to_load}")
    
model = DQN.load(model_to_load, env=test_env)

action_map = {
    0: "Reduce the pressure (-5.0)",
    1: "Keep the pressure on",
    2: "Increase the pressure (+5.0)"
}

print("\n--- Starting Test Simulation ---")
for i in range(NUM_TEST_EPISODES):
    obs, info = test_env.reset() 
    total_reward = 0
    print(f"\n--- Test Episode {i+1} ---")
    
    while True:
        action, _states = model.predict(obs, deterministic=True) 
        
        action_scalar = action.item() 
        action_description = action_map.get(action_scalar, "Unknown Action")
        print(f"-> Agent's Decision: {action_description}")
        
        obs, reward, terminated, truncated, info = test_env.step(action) 
        total_reward += reward
        
        
        if terminated or truncated:
            print(f"--- Episode End ---")
            print(f"Total Reward: {total_reward:.2f}")
            if truncated:
                 print("Reason: Truncated (Max Timesteps Reached)")
            elif terminated:
                 print("Reason: Terminated (Critical Leak or other condition)")
            break

test_env.close() 
print("\n--- Simulation Finished ---")