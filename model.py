import random
import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv
# from new_env import SimpleTaxiEnv
# import env
# env = env.DynamicTaxiEnv()


pass_inx = -1
dest_inx = -1
target = 0

def Q_learning_state(state, action, picked, prev_state): #now state, last action, picked, target, reward_env

    global pass_inx
    global dest_inx
    global target

    taxi_row = state[0] # 1
    taxi_col = state[1] # 2
    station = [[state[2], state[3]], [state[4], state[5]], [state[6], state[7]], [state[8], state[9]]]
    distance_y = [(station[i][0] - taxi_row)  for i in range(4)]
    distance_x = [(station[i][1] - taxi_col)  for i in range(4)]
    distance = [abs(distance_x[i] )+ abs(distance_y[i]) for i in range(4)]
    last_state_distance = [abs(prev_state[i*2+2] - taxi_row) + abs(prev_state[i*2+3] - taxi_col) for i in range(4)]
    obstacle = [state[10], state[11], state[12], state[13]] #north, south, east, west
    passenger_look = state[14]
    destination_look = state[15]


    #which station to go?

    if distance[target] == 0 :
        if destination_look == True:
            dest_inx = target
        if passenger_look == True:
            pass_inx = target
        target = (target + 1) % 4

    if picked == False:
        if action ==4:
            if passenger_look == 1 and prev_state[14] == 1:
                if last_state_distance[pass_inx] ==0:
                    picked = True


    if picked == False:
        if pass_inx != -1:
            target = pass_inx
    if picked == True:
        if dest_inx != -1:
            target = dest_inx

    if distance_x[target] > 0:
        push_distance_x = -1
    elif distance_x[target] < 0:
        push_distance_x = 1
    else:
        push_distance_x = 0
    if distance_y[target] > 0 :
        push_distance_y = -1
    elif distance_y[target] < 0:
        push_distance_y = 1
    else:
        push_distance_y = 0

    return (passenger_look, destination_look, obstacle[0], obstacle[1], obstacle[2], obstacle[3], picked, push_distance_x, push_distance_y)


# ===== Q-Learning 超參數 =====
episodes = 500000        # 訓練回合數
max_steps = 50         # 每回合最大步數
alpha = 0.1              # 學習率
gamma = 0.99             # 折扣因子

epsilon_start = 1.0            # 初始探索率
min_epsilon = 0.01       # 探索率下限
decay_rate = 0.99999    # 衰減速率

reward_shape = True      # 是否使用 reward shaping
q_table = None           # Q-table


# ===== 建立環境 =====
# 這裡使用 grid_size=5 進行訓練，fuel_limit 設為 5000
env = SimpleTaxiEnv(fuel_limit=5000)
# env = env.DynamicTaxiEnv()
fuel_limit = 5000
if q_table is None:
    q_table = {}
rewards_per_episode = []
steps_per_episode = []
epsilon = epsilon_start
# ===== 訓練流程 =====
successes = 0
for episode in range(episodes):
    # 準備訓練環境
    state, _ = env.reset()  # reset() 回傳 (state, info)
    prev_state = state
    q_state = Q_learning_state(state, 0, picked=False, prev_state=state)
    prev_q_state = q_state
    done = False

    passenger_look = q_state[0] # 乘客可視標記
    destination_look = q_state[1] # 目的地可視標記
    obstacle_north = q_state[2]
    obstacle_south = q_state[3]
    obstacle_east = q_state[4]
    obstacle_west = q_state[5]
    picked = q_state[6]
    distance_x = q_state[7]
    distance_y= q_state[8]

    # ====================
    reward = 0
    episode_step = 0
    total_reward = 0
    env_reward_total = 0

    jump = False

    pass_inx = -1
    dest_inx = -1
    target = 0

    for step in range(fuel_limit):

        if q_state not in q_table:
            q_table[q_state] = np.zeros(6)
        # epsilon-greedy 選擇動作
        if np.random.random() < epsilon: 
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(q_table[q_state])
        if jump:
            action = 4

        # 執行動作
        next_state, reward_env, done, _ = env.step(action)
        next_q_state = Q_learning_state(next_state, action, picked, state)
        next_passenger_look = next_q_state[0]
        next_destination_look = next_q_state[1]
        next_obstacle_north = next_q_state[2]
        next_obstacle_south = next_q_state[3]
        next_obstacle_east = next_q_state[4]
        next_obstacle_west = next_q_state[5]
        next_picked = next_q_state[6]
        next_distance_x = next_q_state[7]
        next_distance_y = next_q_state[8]
        episode_step += 1

        shaped_reward = 0
        if reward_shape:
            # #fuel limit -10
            # #wrong pick up -10
            # #wrong drop off -10
            # #move -0.1
            # #currect pick up 0   need to shape 
            # #currect drop off 50  can be shaped
            # #heat wall -5

            if reward_env < -6:
                shaped_reward += 5
            elif reward_env < -1:
                shaped_reward += 2
            elif reward_env == -0.1:
                shaped_reward += 0.09 
            # #coreect pick up
            if picked == False and next_picked == True:
                shaped_reward += 1
            # #incorrect drop off
            if action == 5 and  reward_env<40:
                shaped_reward += -2
                if picked == True:
                    shaped_reward += -50
                    jump = True

            if picked == False:
                if passenger_look == 0 and next_passenger_look == 1:
                   shaped_reward += 1
                if destination_look == 0 and next_destination_look == 1:
                    shaped_reward += 0.5
            if picked == True:
                if destination_look == 0 and next_destination_look == 1:
                   shaped_reward += 2

            if picked == True and dest_inx != -1:
                if distance_x == 0 and distance_y == 0:
                    shaped_reward += 3
                # else:
                #     shaped_reward += -3

            else:
                if distance_x == 0 and distance_y == 0:
                    shaped_reward += 1
                # else:
                #     shaped_reward += -1

            if passenger_look == 1 and destination_look == 1 and picked == True and dest_inx != -1 and distance_x == 0 and distance_y == 0 and action ==5:
                shaped_reward += 20


 
        reward += shaped_reward
        reward += reward_env
        total_reward += reward
        env_reward_total += reward_env

        if jump == True and action == 4:
            jump = False
        else:
            if next_q_state not in q_table:
                q_table[next_q_state] = np.zeros(6)
            # Q-Learning 更新公式
            current_q = q_table[q_state][action]
            max_next_q = np.max(q_table[next_q_state])
            q_table[q_state][action] = current_q + alpha * (reward + gamma * max_next_q - current_q)

        if done and step < fuel_limit-10:
            successes += 1
            break
        state = next_state
        q_state = next_q_state
        passenger_look = next_passenger_look
        destination_look = next_destination_look
        obstacle_north = next_obstacle_north
        obstacle_south = next_obstacle_south
        obstacle_east = next_obstacle_east
        obstacle_west = next_obstacle_west
        picked = next_picked
        distance_x = next_distance_x
        distance_y = next_distance_y
        last_action = action


    rewards_per_episode.append(env_reward_total)
    steps_per_episode.append(episode_step)
    # 衰減 epsilon
    # epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * episode))
    epsilon = max(min_epsilon, epsilon * decay_rate)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_steps = np.mean(steps_per_episode[-100:])
        accuracy = successes/100
        successes = 0
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.4f}")
        print(f"Accuracy: {accuracy:.2f}, Avg. Reward: {avg_reward:.2f}, Avg. Steps: {avg_steps:.2f}")
    

# ===== 儲存 Q 表 =====
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete. Q-table saved as 'q_table.pkl'.")
print(q_table)







