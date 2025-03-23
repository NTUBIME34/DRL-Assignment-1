
import random
import numpy as np
import pickle


# # target = 0

# # def Q_learning_state(state, action, picked):

# #     global target

# #     taxi_row = state[0] # 1
# #     taxi_col = state[1] # 2
# #     station = [[state[2], state[3]], [state[4], state[5]], [state[6], state[7]], [state[8], state[9]]]
# #     distance_x = [abs(taxi_row - station[i][0])  for i in range(4)]
# #     distance_y = [abs(taxi_col - station[i][1])  for i in range(4)]
# #     distance = [distance_x[i] + distance_y[i] for i in range(4)]
# #     obstacle = [state[10], state[11], state[12], state[13]]
# #     passenger_look = state[14]
# #     destination_look = state[15]


# #     if picked == False:
# #         if  state[14] == 1 and action == 4:
# #             picked = True
# #     else:
# #         if action == 5:
# #             picked = False

# #     if distance[0]==0 and target == 0:
# #         target = 1
# #     elif distance[1]==0 and target == 1:
# #         target = 2
# #     elif distance[2]==0 and target == 2:
# #         target = 3
# #     elif distance[3]==0 and target == 3:
# #         target = 0

# #     if picked == False and state[14] == 1:
# #         target = distance.index(min(distance))  # 找到最小距離的索引
# #     elif picked == True and state[15] == 1:
# #         target = distance.index(min(distance))

# #     # Safety check to ensure target is valid before using it
# #     if target < 0 or target >= len(distance):
# #         target = 0  # Default to first station if invalid


# #     push_distance = distance[target]


# #     # return (state)
# #     return (passenger_look, destination_look, obstacle[0], obstacle[1], obstacle[2], obstacle[3], picked, push_distance, action)



# # with open("q_table.pkl", "rb") as f:
# #     Q_table = pickle.load(f)



# # action = 0
# # picked = False
# # reward_env = 0
# # def get_action(obs):
# #     global action
# #     global target
# #     global picked
# #     global reward_env
# #     state = Q_learning_state(obs, action, picked)
# #     if state in Q_table:

# #         action = np.argmax(Q_table[state])
# #         picked = state[6]

# #         print("by q_table")
# #         print(action)
# #         print(state)
# #         return np.argmax(Q_table[state])  # 選擇最高 Q 值的行動
# #     else:
# #         return np.random.choice([0, 1, 2, 3, 4, 5])  #






dest_inx = -1

action = 0
picked = False
target = 0


def Q_learning_state(state, action, picked, prev_state): #now state, last action, picked, target, reward_env


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
        target = (target + 1) % 4

    # if distance[0] == 0 and target == 0:
    #     target = 1
    #     if destination_look == True:
    #         dest_inx = 0
    # elif distance[1] == 0 and target == 1:
    #     target = 2
    #     if destination_look == True:
    #         dest_inx = 1
    # elif distance[2] == 0 and target == 2:
    #     target = 3
    #     if destination_look == True:
    #         dest_inx = 2
    # elif distance[3] == 0 and target == 3:
    #     target = 0
    #     if destination_look == True:
    #         dest_inx = 3
    # else:
    #     target = target

    if picked == False:
        if action ==4:
            if passenger_look == 1 and prev_state[14] == 1:
                if last_state_distance[0] ==0:
                    picked = True
                elif last_state_distance[1] ==0:
                    picked = True
                elif last_state_distance[2] ==0:
                    picked = True
                elif last_state_distance[3] ==0:
                    picked = True

    if picked == True:
        if dest_inx != -1:
            target = dest_inx

    if distance_x[target] > 0:
        push_distance_x = -1 # right
    elif distance_x[target] < 0:
        push_distance_x = 1 # left
    else:
        push_distance_x = 0
    if distance_y[target] > 0 : #up
        push_distance_y = -1
    elif distance_y[target] < 0:#down
        push_distance_y = 1
    else:
        push_distance_y = 0

    return (passenger_look, destination_look, obstacle[0], obstacle[1], obstacle[2], obstacle[3], picked, push_distance_x, push_distance_y)


with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)


conunt_pre = 0
pstate = {}
unstate= 0
epsilon = 0.01

def get_action(obs):
    global action
    global picked
    global conunt_pre
    global pstate
    global unstate
    if conunt_pre == 0:
        state = Q_learning_state(obs, action, picked, obs)
        conunt_pre = 1
        pstate = obs
    else:
        state = Q_learning_state(obs, action, picked, pstate) 
        pstate = obs
    if state in Q_table:

        if np.random.random() < epsilon: 
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(Q_table[state])

        # action = np.argmax(Q_table[state])
        print("by q_table")
        print("action",action)
        print("stat",state)
        print("dest_inx", dest_inx)
        print("target",target)
        return action  # 選擇最高 Q 值的行動
    else:
        action = np.random.choice([0, 1, 2, 3, 4, 5])
        unstate += 1
        return action  #
