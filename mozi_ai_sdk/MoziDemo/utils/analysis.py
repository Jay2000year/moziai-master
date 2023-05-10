import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

def _action_mapping(idx):
    action_mapping = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 0],
        3: [0, 2],
        4: [2, 0],
        5: [1, 1],
        6: [3, 0],
        7: [0, 3],
        8: [1, 2],
        9: [2, 1],
        10: [4, 0],
        11: [0, 4],
        12: [2, 2],
        13: [1, 3],
        14: [3, 1]
    }
    return action_mapping[idx]  
    
def filter(data):
    pass

def which_bar(data, step):
    return int(data/step)

def data_analysis(raw_data):
    distance_idx = 4
    pos, neg = 0, 0
    pos_neg_bar_step = 5
    pos_neg_bar = [[0] * 20, [0]*20]
    intercept_ratio_bar = [0] * 20
    missile_num = [0] * 20
    for i in range(len(raw_data)):
        state, action, reward = raw_data[i].state, raw_data[i].action, raw_data[i].reward
        distance = np.exp(state[distance_idx])
        intercept_plan = _action_mapping(action)
        idx = which_bar(distance, pos_neg_bar_step)
        missile_num[idx] = missile_num[idx] + intercept_plan[0] + intercept_plan[1]
        if reward > 0:
            pos = pos + 1
            pos_neg_bar[0][idx] = pos_neg_bar[0][idx] + 1
        else:
            pos_neg_bar[1][idx] = pos_neg_bar[1][idx] + 1
            neg = neg + 1
    print('[NUM]: pos: %d\tneg: %d'%(pos, neg))
    for i in range(20):
        if pos_neg_bar[0][i] + pos_neg_bar[1][i] == 0:
            continue
        else:
            intercept_ratio_bar[i] =  pos_neg_bar[0][i] * 1.0/ (pos_neg_bar[0][i] + pos_neg_bar[1][i])
    #print('intercept_ratio_bar: ', intercept_ratio_bar)
    for i in range(len(intercept_ratio_bar)):
        #print('%f\t%d'%(intercept_ratio_bar[i], pos_neg_bar[0][i] + pos_neg_bar[1][i]))
        #print('%f'%(intercept_ratio_bar[i]))
        #print('%d'%( pos_neg_bar[0][i] + pos_neg_bar[1][i]))
        print('%f'%(pos_neg_bar[0][i]*1.0/missile_num[i]))
        