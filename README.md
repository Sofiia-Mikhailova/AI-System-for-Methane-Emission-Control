# AI-System-for-Methane-Emission-Control

This project demonstrates the development of a Deep Reinforcement Learning (DRL) agent designed to optimize the operation of a simulated gas well. The primary goal of the agent is to learn a control policy that maximizes natural gas production while minimizing harmful methane leaks.

## Features

- **Custom Simulation Environment**: Built using the Gymnasium (formerly OpenAI Gym) standard to model the core dynamics of a gas well.
- **DQN Implementation**: Utilizes the Deep Q-Network (DQN) algorithm from the Stable-Baselines3 library.
- **Separate Training & Testing**: Modular code with dedicated scripts for training (`train_agent.py`) and evaluating (`test_agent.py`) the agent.
- **Performance Monitoring**: Integrated with TensorBoard for real-time visualization of training metrics like rewards and loss.

## Tech Stack

- Python 3.8+
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- TensorBoard

## Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

## How to Run

The project is split into three main actions: training, monitoring, and testing.

### 1. Train the AI Agent

This script will start the training process. The agent will interact with the environment for a large number of timesteps to learn an optimal strategy.

```bash
python train_agent.py
```

- Logs will be saved to the `tensorboard_logs_well/` directory.
- The best-performing model will be saved in `tmp_well_model/best_model.zip`.
- The final model will be saved as `gas_well_dqn_final_model.zip`.

### 2. Monitor Training with TensorBoard

To visualize the agent's learning progress in real-time, open a new terminal and run:

```bash
tensorboard --logdir ./tensorboard_logs_well/
```

Then open your browser and go to [http://localhost:6006/](http://localhost:6006/) to view training metrics like `rollout/ep_rew_mean` (average episode reward).

### 3. Test the Trained Agent

After training is complete, run the following to see the trained agent in action:

```bash
python test_agent.py
```

This will load the best saved model and simulate the environment step-by-step, printing decisions and states in the console.

## Project Structure

```
.
├── gas_well_env.py         # Defines the custom Gymnasium environment for the gas well.
├── train_agent.py          # Script for training the DQN agent and saving the model.
├── test_agent.py           # Script for loading a trained model and evaluating its performance visually.
├── README.md               # This file.
├── tmp_well_model/         # Directory where the best model is saved.
└── tensorboard_logs_well/  # Directory for storing TensorBoard log files.
```
