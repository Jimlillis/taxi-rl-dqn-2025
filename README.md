# Taxi RL DQN (Taxi-v3)
This repo contains a minimal, from-scratch PyTorch DQN implementation for Gymnasium's Taxi-v3 environment.

## What is DQN (simple terms)
- **Goal:** Learn a policy to maximize rewards by choosing actions in states.
- **Q-learning:** Estimates $Q(s,a)$, the value of taking action $a$ in state $s$ and following the best policy afterwards.
- **Neural network:** Approximates $Q(s,a)$ instead of using a big table.
- **Experience replay:** Store past transitions and train on random batches to stabilize learning.
- **Target network:** A frozen copy of the Q-network used to compute stable targets; updated periodically.
- **Exploration (epsilon-greedy):** Sometimes pick a random action so the agent explores and avoids getting stuck.

## Run (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python dqn_taxi.py
```

## Files
- `dqn_taxi.py`: DQN agent, training loop, and evaluation for Taxi-v3.
- `requirements.txt`: Dependencies.

## Notes
- Uses one-hot encoding for the 500 discrete Taxi states, with a small MLP.
- Prints moving averages during training and an evaluation score at the end.
