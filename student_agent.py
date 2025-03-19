# # import numpy as np
# # import pickle


# # # 載入預先訓練的 Q-table
# # with open("q_table.pkl", "rb") as f:
# #     Q_table = pickle.load(f)

# # picekd = False
# # finded = -1
# # target = 0


# # def Q_learning_state(state, target, picked, finded):

# #     taxi_row = state[0] # 1
# #     taxi_col = state[1] # 2
# #     station = [[state[2], state[3]], [state[4], state[5]], [state[6], state[7]], [state[8], state[9]]]
# #     distance_x = [abs(taxi_row - station[i][0])  for i in range(4)]
# #     distance_y = [abs(taxi_col - station[i][1])  for i in range(4)]
# #     # distance = [distance_x[i] + distance_y[i] for i in range(4)]
# #     now_distance_x = distance_x[0]
# #     now_distance_y = distance_y[0]

# #     obstacle = [state[10], state[11], state[12], state[13]]
# #     passenger_look = state[14]
# #     destination_look = state[15]

# #     return (now_distance_x, now_distance_y, target, passenger_look, destination_look, picked, finded, obstacle[0], obstacle[1], obstacle[2], obstacle[3])


# # def get_action(obs):
# #     """
# #     使用 Q-table 選擇最佳行動
# #     """

# #     state = Q_learning_state(obs, target, picked, finded)
# #     target = state[2]
# #     picked = state[5]
# #     finded = state[6]
# #     if state in Q_table:
# #         return np.argmax(Q_table[state])  # 選擇最高 Q 值的行動
# #     else:
# #         return np.random.choice([0, 1, 2, 3, 4, 5])  # 隨機選擇行動避免崩潰



# import pickle
# import numpy as np
# import random

# with open("q_table.pkl", "rb") as f:
#     Q_table = pickle.load(f)

# def q_state(obs):
#     # Convert the environment's observation into something
#     # that matches how you stored states in Q_table.
#     # Adjust as needed based on how you encoded states.
#     return tuple(obs)

# def get_action(obs):
#     """
#     Called by simple_custom_taxi_env.py to get an action from the loaded Q-table.
#     """
#     state_key = q_state(obs)
#     if state_key in Q_table:
#         return np.argmax(Q_table[state_key])
#     else:
#         return random.choice([0, 1, 2, 3, 4, 5])  # fallback





import random
import numpy as np
import pickle

# 載入預先訓練的 Q-table
with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

def get_action(obs):
    """
    使用 Q-table 選擇最佳行動
    """
    state_key = tuple(obs)
    if state_key in Q_table:
        return np.argmax(Q_table[state_key])  # 選擇最高 Q 值的行動
    else:
        return random.choice([0, 1, 2, 3, 4, 5])  # 隨機選擇行動避免崩潰