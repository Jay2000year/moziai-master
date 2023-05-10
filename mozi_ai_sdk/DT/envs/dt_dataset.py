import csv
import logging
from dtmodel.utils import set_seed
import numpy as np
import pandas as pd
from envs.load_data import ConstructionMatrix
from envs.playeraction_history_multiproc import DataProcessor
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--action_file_name", type=str, default="red_actions.xlsx")
parser.add_argument("--total_rtg_file_name", type=str, default="total_rtg_data.xlsx")
parser.add_argument(
    "--path", type=str, default="D:\\mozi_decision_transformer_dataset\\"
)
parser.add_argument("--action_portrait_dim", type=int, default=15)

args = parser.parse_args()


def stepwise_returns(dim, file_name):
    start_index = 0
    stepwise_returns = np.zeros(dim)
    rtg = np.zeros_like(stepwise_returns)
    rtg_position = pd.read_excel(
        args.path + file_name + "\\" + args.total_rtg_file_name
    )
    pos = rtg_position.duplicated("rtg_pos", keep="last")
    rtg_pos = [step for step, bool in enumerate(pos) if not bool]
    columns = rtg_position.columns.values.tolist()
    final_index = rtg_position.shape[0] - 1
    step_dic = {}
    event = [row["event"] for idx, row in rtg_position.iterrows()]
    final = [
        row["event"] for idx, row in rtg_position.iterrows() if idx == final_index
    ][0]
    rtg1 = {
        row["rtg_pos"]: row["event"]
        for idx, row in rtg_position.iterrows()
        if idx in rtg_pos
    }
    for i, x in rtg1.items():
        step_dic[i] = final - x
    for idx, score in step_dic.items():
        stepwise_returns[idx] = score
        rtg[start_index : idx + 1] = score
        start_index = idx + 1

    return stepwise_returns, rtg


def create_dataset_mozi(num_steps, game, data_dir_prefix, matrix_size):
    # -- load data from memory (make more efficient)
    action_file_lst = os.listdir(data_dir_prefix)
    total_actions = []
    returns = [0]
    returns_lst = []
    done_idxs = []
    # obss
    matrix_data = ConstructionMatrix(matrix_size)
    obss = matrix_data.single_matrix_lst
    # action
    for done_idx, file in enumerate(action_file_lst):
        done_idxs.append(done_idx)
        red_action = pd.read_excel(
            data_dir_prefix + "\\" + file + "\\" + args.action_file_name
        )
        actions = []
        for idx, row in red_action.iterrows():
            action = eval(row["action"])
            action_array_lst = []
            action_dic = {}
            int_action_amount = []
            list_action_amount = []
            for act in action:
                action_array = np.zeros(args.action_portrait_dim, dtype=int)
                if isinstance(act, int):
                    action_array[0] = act
                    int_action_amount.append(idx)
                elif isinstance(act, list):
                    ind = 0
                    list_action_amount.append(idx)
                    for k, v in enumerate(act):
                        if isinstance(v, int):
                            action_array[k] = v
                        else:
                            for k1, v1 in enumerate(v):
                                action_array[ind] = v1
                                ind += 1
                action_array_lst.append(action_array)
            action_array_lst = np.array(action_array_lst)
            actions.append(action_array_lst)
            if done_idx == 0:
                done_idxs[done_idx] = idx + 1
            else:
                done_idxs[done_idx] = idx + 1 + done_idxs[done_idx - 1]
        total_actions = total_actions + actions

    actions = np.array(total_actions)
    returns = np.array(returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    rtg_total = []
    total_stepwise = []
    start_index = 0
    for idx, i in enumerate(done_idxs):
        if idx == 0:
            i = int(i)
            start_index = 0
        else:
            i = int(i) - done_idxs[idx - 1]
            start_index = start_index + done_idxs[idx - 1]
        stepwise_ret, rtg = stepwise_returns(i, action_file_lst[idx])
        rtg_total = rtg_total + list(rtg)
    rtg_total = np.array(rtg_total)
    print("max rtg is %d" % max(rtg))
    returns = returns_lst + returns
    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions) + 1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index : i + 1] = np.arange(i + 1 - start_index)
        start_index = i + 1
    print("max timestep is %d" % max(timesteps))

    return obss, actions, returns, done_idxs, rtg_total, timesteps, action_dic


# if __name__ == '__main__':
#     create_dataset_mozi(400, 'MoZi', 'D:/mozi_luahistory/202281_10/')
