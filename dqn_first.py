import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- ΥΠΕΡ-ΠΑΡΑΜΕΤΡΟΙ (Ρυθμισμένοι για Taxi-v3) ---
env = gym.make("Taxi-v3") # Δημιουργία περιβάλλοντος για να πάρουμε τα μεγέθη

N_STATES = env.observation_space.n  # 500
N_ACTIONS = env.action_space.n      # 6
BATCH_SIZE = 64        # Εκπαίδευση σε παρτίδες των 64
MEMORY_SIZE = 50000    # Μεγαλύτερη μνήμη
GAMMA = 0.99           # Σημαντικό το μέλλον (θέλουμε να φτάσουμε στο στόχο)
LEARNING_RATE = 0.001  # Πιο συντηρητική μάθηση
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MAX_EPISODES = 1000    # Το Taxi θέλει χρόνο για να λυθεί με DQN

# --- 1. ΤΟ ΝΕΥΡΩΝΙΚΟ ΔΙΚΤΥΟ ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # 500 είσοδοι -> 128 κρυφοί -> 64 κρυφοί -> 6 έξοδοι
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Ο ΠΡΑΚΤΟΡΑΣ (AGENT) ---
class Agent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = DQN(N_STATES, N_ACTIONS)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    # --- ΣΗΜΑΝΤΙΚΟ: Μετατροπή Ακεραίου (π.χ. 42) σε One-Hot Vector ---
    def state_to_tensor(self, state):
        # Δημιουργούμε πίνακα με μηδενικά
        state_vec = np.zeros(N_STATES)
        # Βάζουμε 1 στη θέση που αντιστοιχεί στην κατάσταση
        state_vec[int(state)] = 1
        # Το κάνουμε Tensor για το PyTorch
        return torch.FloatTensor(state_vec).unsqueeze(0)

    def act(self, state):
        # Epsilon-Greedy Strategy
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample() # Τυχαία κίνηση από το Gym
        
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item() # Κίνηση με τη μέγιστη τιμή Q

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = self.state_to_tensor(state)
            next_state_tensor = self.state_to_tensor(next_state)

            target = reward
            if not done:
                # Bellman Equation
                target = reward + GAMMA * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor)
            
            # Αντιγραφή για να μην χαλάσουμε το γράφημα (detach/clone)
            target_vec = target_f.clone().detach()
            target_vec[0][action] = target

            # Εκπαίδευση
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, target_vec)
            loss.backward()
            self.optimizer.step()

        # Μείωση Εξερεύνησης
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# --- 3. ΚΥΡΙΟΣ ΒΡΟΧΟΣ ΕΚΠΑΙΔΕΥΣΗΣ ---
if __name__ == "__main__":
    agent = Agent()
    print("Ξεκινάει η εκπαίδευση στο Taxi-v3...")

    for e in range(MAX_EPISODES):
        # Το reset στο νέο gym επιστρέφει (state, info)
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            
            # Το step επιστρέφει 5 τιμές πλέον
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Το επεισόδιο τελειώνει είτε αν νικήσει/χάσει (terminated) είτε αν περάσει ο χρόνος (truncated)
            done = terminated or truncated 

            # Αποθήκευση
            agent.remember(state, action, reward, next_state, done)
            
            # Μάθηση
            agent.replay()

            state = next_state
            total_reward += reward
        
        # Εκτύπωση προόδου κάθε 50 επεισόδια
        if (e + 1) % 50 == 0:
            print(f"Episode: {e+1}/{MAX_EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    print("Η εκπαίδευση ολοκληρώθηκε.")

    # --- 4. ΟΠΤΙΚΟΠΟΙΗΣΗ (TESTING) ---
    # Κλείνουμε το παλιό περιβάλλον και ανοίγουμε ένα νέο με γραφικά (render_mode)
    env.close()
    env = gym.make("Taxi-v3", render_mode="human")
    
    state, info = env.reset()
    env.render()
    done = False
    total_reward = 0
    
    print("\nΠάμε μια βόλτα επίδειξης...")
    while not done:
        # Χωρίς τυχαίες κινήσεις τώρα (Greedy)
        state_tensor = agent.state_to_tensor(state)
        with torch.no_grad():
            action = torch.argmax(agent.model(state_tensor)).item()
            
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    print(f"Τελικό Σκορ Επίδειξης: {total_reward}")
    env.close()