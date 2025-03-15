import random
import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv

# ------------------------------------------------
# Environment Settings and Hyperparameters
# ------------------------------------------------
GRID_SIZE = 5
NUM_EPISODES = 200000        # Number of training episodes
MAX_STEPS = 1000             # Maximum steps per episode
ALPHA = 0.1                  # Learning rate
GAMMA = 0.99                 # Discount factor

EPSILON_START = 1.0          # Initial exploration rate
MIN_EPSILON = 0.01           # Minimum exploration rate
EPSILON_DECAY = 0.99999      # Decay factor

# ------------------------------------------------
# Q-Table Initialization
# ------------------------------------------------
q_table = {}

# ------------------------------------------------
# Discrete State Encoding Function (Taxi-v3 style)
# ------------------------------------------------
def encode_state(taxi_row, taxi_col, passenger_status, destination_index, grid_size=5):
    """
    Encode state as a single integer:
      - taxi_row, taxi_col: taxi’s position (0 to grid_size-1)
      - passenger_status: 0-3 if waiting at a station; 4 if in taxi.
      - destination_index: index (0 to 3) of the destination station.
    This encoding reduces the state space compared to using the full observation.
    """
    return taxi_row + grid_size * taxi_col + (grid_size**2) * passenger_status + (grid_size**2) * 5 * destination_index

# ------------------------------------------------
# Manhattan Distance Helper Function
# ------------------------------------------------
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# ------------------------------------------------
# Create Environment and Initialize Training Variables
# ------------------------------------------------
env = SimpleTaxiEnv(grid_size=GRID_SIZE, fuel_limit=1000)
rewards_per_episode = []
steps_per_episode = []
epsilon = EPSILON_START
success_count = 0  # Count successful episodes (successful drop-off)

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    taxi_row, taxi_col = obs[0], obs[1]
    # Determine passenger and destination indices from the stations list.
    passenger_index = env.stations.index(env.passenger_loc)
    destination_index = env.stations.index(env.destination)
    # Initially, the passenger is waiting at its station; we represent that by its station index (0-3).
    passenger_status = passenger_index  
    # Encode the initial state.
    state = encode_state(taxi_row, taxi_col, passenger_status, destination_index, GRID_SIZE)
    
    # Set the initial target: if the taxi has not picked up the passenger, target is the passenger location;
    # once picked up, target is the destination.
    target = env.passenger_loc if passenger_status < 4 else env.destination
    prev_distance = manhattan_distance((taxi_row, taxi_col), target)
    
    total_reward = 0
    done = False
    steps = 0
    episode_success = False  # Will be True if drop-off is successful
    
    for step in range(MAX_STEPS):
        # Initialize Q-values for unseen states.
        if state not in q_table:
            q_table[state] = np.zeros(6)
        
        # Epsilon-greedy action selection.
        if random.random() < epsilon:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = int(np.argmax(q_table[state]))
        
        next_obs, reward, done, _ = env.step(action)
        next_taxi_row, next_taxi_col = next_obs[0], next_obs[1]
        
        # Process pickup and drop-off actions.
        if action == 4:  # PICKUP
            if (next_taxi_row, next_taxi_col) == env.passenger_loc:
                passenger_status = 4  # Successfully picked up.
            else:
                reward -= 10  # Penalty for incorrect pickup.
        if action == 5:  # DROPOFF
            if passenger_status == 4 and (next_taxi_row, next_taxi_col) == env.destination:
                reward += 50  # Successful drop-off bonus.
                done = True
                episode_success = True
            else:
                reward -= 10  # Penalty for incorrect drop-off.
        
        # Update target and compute dense reward shaping.
        target = env.destination if passenger_status == 4 else env.passenger_loc
        curr_distance = manhattan_distance((next_taxi_row, next_taxi_col), target)
        # Shaping bonus: if taxi moves closer to the target, bonus is positive.
        shaping_bonus = prev_distance - curr_distance
        reward += shaping_bonus
        prev_distance = curr_distance
        
        next_state = encode_state(next_taxi_row, next_taxi_col, passenger_status, destination_index, GRID_SIZE)
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
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}")
        print(f"Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Success Rate: {success_rate:.2f}")
        success_count = 0

# ------------------------------------------------
# Save the Q-Table
# ------------------------------------------------
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete. Q-table saved as 'q_table.pkl'.")
