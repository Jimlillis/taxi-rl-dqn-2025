import random
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym


# ------------------------
# Config
# ------------------------
@dataclass
class Config:
    env_id: str = "Taxi-v3"
    seed: int = 42
    episodes: int = 1200
    max_steps_per_episode: int = 200
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 50_000
    lr: float = 5e-4
    target_update_every: int = 10
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    epsilon_decay_episodes: int = 700
    eval_episodes: int = 10


# ------------------------
# Q-network (small MLP)
# ------------------------
class QNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        # Taxi-v3 has 500 discrete states; we one-hot encode them to a vector
        self.model = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.model(x)


# ------------------------
# Utils
# ------------------------
def set_seed(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)


def one_hot_state(s: int, n_states: int):
    v = np.zeros(n_states, dtype=np.float32)
    v[s] = 1.0
    return torch.from_numpy(v)


# ------------------------
# DQN Agent
# ------------------------
class DQNAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: Config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNetwork(n_states, n_actions).to(self.device)
        self.target_q = QNetwork(n_states, n_actions).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.replay = deque(maxlen=cfg.replay_capacity)
        self.steps_done = 0

    def select_action(self, state_idx: int, episode: int):
        # Linear epsilon decay
        eps = self.epsilon_by_episode(episode)
        if random.random() < eps: 
            return random.randrange(self.n_actions), eps 
        with torch.no_grad():
            s_vec = one_hot_state(state_idx, self.n_states).to(self.device)
            q_values = self.q(s_vec)
            action = int(torch.argmax(q_values).item())
            return action, eps

    def epsilon_by_episode(self, episode: int):
        # Decay epsilon from start to end across epsilon_decay_episodes
        t = min(episode, self.cfg.epsilon_decay_episodes) 
        frac = t / max(1, self.cfg.epsilon_decay_episodes) 
        return self.cfg.start_epsilon + (self.cfg.end_epsilon - self.cfg.start_epsilon) * frac

    def store(self, s, a, r, s_next, done):
        self.replay.append((s, a, r, s_next, done))

    def sample_batch(self):
        batch = random.sample(self.replay, self.cfg.batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            torch.stack([one_hot_state(si, self.n_states) for si in s]).to(self.device), 
            torch.tensor(a, dtype=torch.long, device=self.device),
            torch.tensor(r, dtype=torch.float32, device=self.device),
            torch.stack([one_hot_state(sj, self.n_states) for sj in s_next]).to(self.device),
            torch.tensor(d, dtype=torch.float32, device=self.device),
        )

    def train_step(self):
        if len(self.replay) < self.cfg.batch_size:
            return None
        s, a, r, s_next, d = self.sample_batch()

        # Current Q(s,a)
        q_sa = self.q(s).gather(1, a.view(-1, 1)).squeeze(1) 

        # Target Q: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next = self.target_q(s_next).max(1).values 
            target = r + self.cfg.gamma * max_next * (1.0 - d) 

        loss = self.loss_fn(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()
        return float(loss.item())

    def update_target(self):
        self.target_q.load_state_dict(self.q.state_dict())


# ------------------------
# Training & Evaluation
# ------------------------

def train_and_eval(cfg: Config):
    env = gym.make(cfg.env_id) 
    set_seed(env, cfg.seed) 

    # Discrete observation space (n=500), discrete actions (n=6)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = DQNAgent(n_states, n_actions, cfg)

    rewards = []
    losses = []

    for ep in range(cfg.episodes):
        state, info = env.reset()
        total_r = 0.0
        for t in range(cfg.max_steps_per_episode):
            action, eps = agent.select_action(state, ep)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_r += reward
            if done:
                break

        rewards.append(total_r)

        if (ep + 1) % cfg.target_update_every == 0:
            agent.update_target()

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(rewards[-10:])
            avg_l = np.mean(losses[-10:]) if len(losses) >= 10 else np.nan
            print(f"Episode {ep+1:4d} | eps={eps:.2f} | avgR(10)={avg_r:.2f} | avgL(10)={avg_l:.4f}")

    # Evaluation
    eval_rewards = []
    for i in range(cfg.eval_episodes):
        s, info = env.reset()
        total = 0.0
        for t in range(cfg.max_steps_per_episode):
            with torch.no_grad():
                s_vec = one_hot_state(s, agent.n_states).to(agent.device)
                a = int(torch.argmax(agent.q(s_vec)).item())
            s, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        eval_rewards.append(total)
    print(f"Eval avg reward over {cfg.eval_episodes} episodes: {np.mean(eval_rewards):.2f}")


if __name__ == "__main__":
    cfg = Config()
    train_and_eval(cfg)
