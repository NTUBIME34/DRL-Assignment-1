import random
import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv

# ------------------------------------------------
# Environment and Hyperparameters
# ------------------------------------------------
GRID_SIZE = 5
NUM_EPISODES = 200000        # Number of training episodes
MAX_STEPS = 1000             # Maximum steps per episode

ALPHA = 0.05                 # Learning rate
GAMMA = 0.95                 # Discount factor

EPSILON_START = 1.0          # Initial exploration rate
MIN_EPSILON = 0.01           # Minimum exploration rate
EPSILON_DECAY = 0.9995       # Faster decay for exploitation

# Reward shaping parameters:
SHAPING_SCALE = 10           # Factor for distance improvement bonus
DROP_OFF_BONUS = 250         # Bonus for a correct drop-off
TERMINATION_BONUS = 100      # Extra bonus for finishing successfully
STEP_PENALTY = -2            # Constant penalty per step

# ------------------------------------------------
# Q-table Initialization
# ------------------------------------------------
q_table = {}

# ------------------------------------------------
# Discrete State Encoding Function (Taxi-v3 style)
# ------------------------------------------------
def encode_state(taxi_row, taxi_col, passenger_status, destination_idx, grid_size=5):
    """
    Encode state as a single integer.
    
    The encoding is:
      state = taxi_row + grid_size * taxi_col + (grid_size**2) * passenger_status + (grid_size**2) * 5 * destination_idx
    passenger_status: 0-3 indicates passenger waiting at that station,
                      4 indicates passenger in taxi.
    destination_idx: index of destination station (0 to 3)
    """
    return taxi_row + grid_size * taxi_col + (grid_size**2) * passenger_status + (grid_size**2) * 5 * destination_idx

# ------------------------------------------------
# Manhattan Distance Helper Function
# ------------------------------------------------
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# ------------------------------------------------
# Create Environment and Initialize Variables
# ------------------------------------------------
env = SimpleTaxiEnv(grid_size=GRID_SIZE, fuel_limit=1000)
rewards_per_episode = []
steps_per_episode = []
epsilon = EPSILON_START
success_count = 0  # Count episodes that successfully drop off the passenger

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    taxi_row, taxi_col = obs[0], obs[1]
    # Get indices for passenger and destination from env.stations
    passenger_idx = env.stations.index(env.passenger_loc)
    destination_idx = env.stations.index(env.destination)
    # Initially, passenger_status equals the passenger station index (0-3)
    passenger_status = passenger_idx  
    state = encode_state(taxi_row, taxi_col, passenger_status, destination_idx, GRID_SIZE)
    
    # Set the initial target and compute the Manhattan distance to it.
    target = env.passenger_loc if passenger_status < 4 else env.destination
    prev_distance = manhattan_distance((taxi_row, taxi_col), target)
    
    total_reward = 0
    done = False
    steps = 0
    episode_success = False
    
    for step in range(MAX_STEPS):
        # Ensure current state is in Q-table.
        if state not in q_table:
            q_table[state] = np.zeros(6)
        
        # Epsilon-greedy action selection.
        if random.random() < epsilon:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = int(np.argmax(q_table[state]))
        
        next_obs, reward, done, _ = env.step(action)
        next_taxi_row, next_taxi_col = next_obs[0], next_obs[1]
        
        # Process PICKUP (action 4) and DROPOFF (action 5)
        if action == 4:  # PICKUP
            if (next_taxi_row, next_taxi_col) == env.passenger_loc:
                passenger_status = 4  # Passenger now in taxi.
            else:
                reward -= 10  # Incorrect pickup attempt.
        if action == 5:  # DROPOFF
            if passenger_status == 4 and (next_taxi_row, next_taxi_col) == env.destination:
                reward += DROP_OFF_BONUS
                reward += TERMINATION_BONUS
                done = True
                episode_success = True
            else:
                reward -= 10  # Incorrect drop-off.
        
        # Add a constant step penalty.
        reward += STEP_PENALTY
        
        # Update target and compute dense reward shaping bonus.
        target = env.destination if passenger_status == 4 else env.passenger_loc
        curr_distance = manhattan_distance((next_taxi_row, next_taxi_col), target)
        shaping_bonus = SHAPING_SCALE * (prev_distance - curr_distance)
        reward += shaping_bonus
        prev_distance = curr_distance
        
        next_state = encode_state(next_taxi_row, next_taxi_col, passenger_status, destination_idx, GRID_SIZE)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(6)
        
        # Q-learning update.
        current_q = q_table[state][action]
        max_next_q = np.max(q_table[next_state])
        q_table[state][action] = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    if episode_success:
        success_count += 1

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_steps = np.mean(steps_per_episode[-100:])
        success_rate = success_count / 100.0
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {epsilon:.4f}")
        print(f"Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.2f}, Success Rate = {success_rate:.2f}")
        success_count = 0

with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete. Q-table saved as 'q_table.pkl'.")
