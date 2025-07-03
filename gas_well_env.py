import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GasWellEnv(gym.Env):
    """
    A simplified gas well simulation environment for DRL.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30} 

    def __init__(self, render_mode=None): 
        super().__init__() 
        self.render_mode = render_mode

        self.min_pressure = 10.0  
        self.max_pressure = 100.0 
        self.pressure_step = 5.0  

        self.production_factor = 1.0 
        self.leak_base_factor = 0.05 
        self.leak_severity_factor = 0.1 
        
        self.max_leak_severity = 0.5 
        self.leak_severity_increase_rate = 0.01 
        self.random_leak_event_prob = 0.05 

        self.max_timesteps = 100    
        self.current_timestep = 0

        low = np.array([self.min_pressure, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([self.max_pressure, 
                         self.max_pressure * self.max_leak_severity * 2, 
                         self.max_pressure * self.production_factor, 
                         self.max_leak_severity], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.action_space = spaces.Discrete(3)

        self.current_pressure = (self.min_pressure + self.max_pressure) / 2 
        self.current_leak_rate = 0.0
        self.current_gas_production = 0.0
        self.leak_severity = 0.0 

    def _get_obs(self):
        return np.array([self.current_pressure, 
                         self.current_leak_rate, 
                         self.current_gas_production,
                         self.leak_severity], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.current_pressure = (self.min_pressure + self.max_pressure) / 2 
        self.current_leak_rate = 0.0
        self.current_gas_production = 0.0
        self.leak_severity = random.uniform(0.01, 0.1) 
        self.current_timestep = 0

        observation = self._get_obs()
        info = {}
        
        if self.render_mode == 'human':
            self.render() 
        
        return observation, info

    def step(self, action):
        self.current_timestep += 1

        if action == 0:  
            self.current_pressure -= self.pressure_step
        elif action == 2: 
            self.current_pressure += self.pressure_step
        self.current_pressure = np.clip(self.current_pressure, self.min_pressure, self.max_pressure)

        self.leak_severity += self.leak_severity_increase_rate * random.uniform(0.8, 1.2) 
        if random.random() < self.random_leak_event_prob: 
            self.leak_severity += random.uniform(0.05, 0.15)
        self.leak_severity = np.clip(self.leak_severity, 0.0, self.max_leak_severity)

        self.current_leak_rate = self.current_pressure * self.leak_base_factor + \
                                 self.current_pressure * self.leak_severity * self.leak_severity_factor
        self.current_leak_rate = max(0.0, self.current_leak_rate)

        self.current_gas_production = (self.current_pressure * self.production_factor) - (self.current_leak_rate * 2.0) 
        self.current_gas_production = max(0.0, self.current_gas_production) 

        reward = self.current_gas_production * 1.0  
        reward -= self.current_leak_rate * 50.0   

        terminated = False 
        truncated = False

        if self.current_timestep >= self.max_timesteps:
            truncated = True 
        
        if self.current_leak_rate > self.max_pressure * self.leak_severity_factor * self.max_leak_severity * 0.8: 
             reward -= 1000 
             terminated = True 

        observation = self._get_obs()
        info = {} 

        if self.render_mode == 'human':
            self.render() 

        return observation, reward, terminated, truncated, info

    def render(self): 
        print(f"Time: {self.current_timestep}/{self.max_timesteps} | "
              f"Pressure: {self.current_pressure:.2f} | "
              f"Gas Production: {self.current_gas_production:.2f} | "
              f"Methane Leak: {self.current_leak_rate:.2f} | "
              f"Leak Severity: {self.leak_severity:.2f}")
        
        return None 

    def close(self):
        pass

if __name__ == "__main__":
    env = GasWellEnv(render_mode='human') 
    obs, info = env.reset() 
    print("Initial State:", obs)

    total_reward = 0
    terminated = False
    truncated = False 

    print("\nStarting simulation with random actions...\n")
    for i in range(env.max_timesteps + 10): 
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action) 
        total_reward += reward
        # env.render() 

        if terminated or truncated:
            print(f"Episode finished after {i+1} timesteps with total reward: {total_reward:.2f}")
            if terminated:
                print("Reason: Terminated (Critical Leak)")
            elif truncated:
                print("Reason: Truncated (Max Timesteps)")
            
            obs, info = env.reset() 
            total_reward = 0
            terminated = False
            truncated = False 
            print("\nResetting environment for a new episode...\n")
            
            if i > env.max_timesteps * 2: 
                break