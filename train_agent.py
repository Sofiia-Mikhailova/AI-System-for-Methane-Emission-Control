import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
import sys

from gas_well_env import GasWellEnv 

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

gym.envs.registration.register(
    id='GasWell-v0',
    entry_point='gas_well_env:GasWellEnv', 
    max_episode_steps=GasWellEnv().max_timesteps,
)

log_dir = "./tmp_well_model/" 
tensorboard_log_dir = "./tensorboard_logs_well/"
training_log_file = "training_log.txt"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

train_env = make_vec_env('GasWell-v0', n_envs=1) 

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    buffer_size=5000,   
    learning_starts=500, 
    train_freq=(1, "step"), 
    target_update_interval=100, 
    verbose=1,          
    tensorboard_log=tensorboard_log_dir, 
)


eval_callback = EvalCallback(
    eval_env=train_env, 
    best_model_save_path=log_dir, 
    log_path=log_dir, 
    eval_freq=5000, 
    deterministic=True, 
    render=False 
)

print("Starting AI agent training... Logs will be saved to training_log.txt")

original_stdout = sys.stdout 
with open(training_log_file, 'w') as log_file:
    sys.stdout = Tee(log_file, original_stdout)
    
    try:
        model.learn(total_timesteps=150_000, callback=eval_callback) 
    finally:
        sys.stdout = original_stdout

print("AI agent training finished.")

model.save("gas_well_dqn_final_model")
print("Final model saved as gas_well_dqn_final_model.zip")

train_env.close()

print("\n--- Testing the trained AI model with rendering ---")

test_env = gym.make('GasWell-v0', render_mode='human')

# try:
#     best_model_path = os.path.join(log_dir, "best_model.zip")
#     if os.path.exists(best_model_path):
#         model = DQN.load(best_model_path)
#         print("Loaded BEST model for testing.")
#     else:
#         print("Best model not found, testing with the LAST trained model.")
# except Exception as e:
#     print(f"Error loading best model: {e}. Testing with the last trained model.")


test_model = model 
print("Testing with the LAST trained model.")

for i in range(5):
    obs, info = test_env.reset()
    total_reward = 0
    print(f"\n--- Test Episode {i+1} ---")
    
    while True:
        action, _states = model.predict(obs, deterministic=True) 
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

print("\nSimulation finished. Check TensorBoard and training_log.txt for details.")
