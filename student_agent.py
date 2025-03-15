import numpy as np
import pickle
import random
import gym

# Load the trained Q-table
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

def get_state_key(obs):
    # Convert observation to a tuple to use as a key in the Q-table
    return tuple(obs)

def get_action(obs):
    state_key = get_state_key(obs)
    if state_key in q_table:
        # Exploit: choose the action with the highest Q-value
        return np.argmax(q_table[state_key])
    else:
        # Fallback strategy: choose a random action
        return random.choice([0, 1, 2, 3, 4, 5])  # Adjust action space as needed
