import csv
import logging

# make deterministic
from dtmodel.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset

# GPT  Single output and multi output
from dtmodel.model_mozi import GPT, GPTConfig

# from mingpt.model_mozi_planb import GPT, GPTConfig

from dtmodel.trainer_mozi import Trainer, TrainerConfig
from dtmodel.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from envs.dt_dataset import create_dataset_mozi

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--context_length", type=int, default=30)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--model_type", type=str, default="reward_conditioned")
parser.add_argument("--num_steps", type=int, default=400)
parser.add_argument("--num_buffers", type=int, default=400)
parser.add_argument("--game", type=str, default="MoZi")
parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument(
    "--trajectories_per_buffer",
    type=int,
    default=10,
    help="Number of trajectories to sample from each of the buffers.",
)
parser.add_argument("--action_portrait_dim", type=int, default=15)

parser.add_argument(
    "--data_dir_prefix", type=str, default="D:\\mozi_decision_transformer_dataset\\"
)
parser.add_argument("--multi_output", type=list, default=[4, 6, 8, 10])
parser.add_argument("--matrix_size", type=int, default=256)

args = parser.parse_args()

set_seed(args.seed)


class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        # self.vocab_size = np.max(actions) + 1
        self.vocab_size = 2121
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(
            np.array(self.data[idx:done_idx]), dtype=torch.float32
        ).reshape(
            block_size, -1
        )  # (block_size, 4*84*84)
        # states = states / 255.
        # plan by zhait
        # actions = torch.tensor(np.array(self.actions[idx:done_idx]),
        #                        dtype=torch.float32).reshape(block_size, -1)  # (block_size, 15*15)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(
            1
        )  # (block_size, 1)

        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(
            self.timesteps[idx : idx + 1], dtype=torch.int64
        ).unsqueeze(1)

        return states, actions, rtgs, timesteps


obss, actions, returns, done_idxs, rtgs, timesteps, actions_dic = create_dataset_mozi(
    args.num_steps, args.game, args.data_dir_prefix, args.matrix_size
)

# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

train_dataset = StateActionReturnDataset(
    obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps
)

mconf = GPTConfig(
    train_dataset.vocab_size,
    train_dataset.block_size,
    action_portrait_dim=args.action_portrait_dim,
    n_layer=6,
    n_head=8,
    n_embd=240,
    model_type=args.model_type,
    max_timestep=max(timesteps),
    actions_dic=actions_dic,
    multi_output_lst=args.multi_output,
    batch_size=args.batch_size,
    matrix_size=args.matrix_size,
)
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(
    max_epochs=epochs,
    batch_size=args.batch_size,
    learning_rate=6e-4,
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=2 * len(train_dataset) * args.context_length * 3,
    num_workers=0,
    seed=args.seed,
    model_type=args.model_type,
    game=args.game,
    max_timestep=max(timesteps),
    ckpt_path="C:/Users/3-5/Desktop/moziai-master/mozi_ai_sdk/DT/modelpara.pt",
    block_size=args.context_length,
    matrix_size=args.matrix_size,
)

dtrainer = Trainer(model, train_dataset, None, tconf)

if __name__ == "__main__":
    dtrainer.train()
