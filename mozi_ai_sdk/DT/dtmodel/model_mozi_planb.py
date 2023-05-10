"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import random
import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)

import numpy as np


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.vocab_size_lst = [5, 5, 5, 5, 40, 5, 40, 5, 400, 5, 400, 1200, 5, 5, 5]

        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
        #                              .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.register_buffer(
            "mask", torch.tril(torch.ones(255, 255)).view(1, 1, 255, 255)
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        # B=x.size()[0]
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.vocab_size = 0

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, 255, config.n_embd))
        self.global_pos_emb = nn.Parameter(
            torch.zeros(1, config.max_timestep + 1, config.n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        # head - vocab_size
        self.head = nn.Linear(config.n_embd, 5, bias=False)
        # self.multi_head_1 = nn.Linear(config.n_embd, 40, bias=False)
        self.multi_head_2 = nn.Linear(config.n_embd, 400, bias=False)
        self.head_3 = nn.Linear(config.n_embd, 1200, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(135424, config.n_embd),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, config.n_embd),
            nn.Tanh(),
        )
        self.targets_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 15, bias=False),
            nn.Tanh(),
        )
        self.multi_output = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, 2, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Flatten(),
            nn.Linear(768, 2121),
            nn.Tanh(),
        )
        self.single_decoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 4, stride=4),
            nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=3),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, 2, stride=2),
            nn.ReLU(),
            torch.nn.Flatten(),
            nn.Linear(5632, 2121),
            nn.Tanh(),
        )
        # lstm = nn.LSTM(input_size=2, hidden_size=3, num_layers=1)  # 传入参数

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, 16), nn.Tanh()
        )
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def multi_output(self):
        # head - vocab_size
        self.head = nn.Linear(self.config.n_embd, 5, bias=False)
        # self.multi_head_1 = nn.Linear(config.n_embd, 40, bias=False)
        self.multi_head_2 = nn.Linear(config.n_embd, 400, bias=False)
        self.head_3 = nn.Linear(config.n_embd, 1200, bias=False)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("global_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)   new action:(batch, block_size, 1, 15, 20)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        state_embeddings = self.state_encoder(
            states.reshape(-1, 4, 400, 400).type(torch.float32).contiguous()
        )  # (batch * block_size, n_embd)
        state_embeddings = state_embeddings.reshape(
            states.shape[0], states.shape[1], self.config.n_embd
        )  # (batch, block_size, n_embd)
        if actions is not None and self.model_type == "reward_conditioned":
            a_start = 0
            a_end = 1
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    states.shape[1] * 17 - int(targets is None),
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::17, :] = rtg_embeddings
            token_embeddings[:, 1::17, :] = state_embeddings
            while a_end <= 15:
                actions_single = (
                    actions[:, :, :, a_start:a_end, :].squeeze().contiguous()
                )
                action_embeddings = self.action_embeddings(
                    actions_single.type(torch.long).squeeze(-1)
                )  # (batch, block_size, n_embd)
                action_embeddings = action_embeddings.reshape(
                    states.shape[0], states.shape[1], self.config.n_embd
                )
                action_embeddings = action_embeddings[
                    :, -states.shape[1] + int(targets is None) :, :
                ].squeeze()
                token_embeddings[:, a_end + 1 :: 17, :] = action_embeddings[
                    :, -states.shape[1] + int(targets is None) :, :
                ].squeeze()
                a_start += 1
                a_end += 1
        elif (
            actions is None and self.model_type == "reward_conditioned"
        ):  # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            # by zhait 22/8/19
            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 17, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == "naive":
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1)
            )  # (batch, block_size, n_embd)
            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    states.shape[1] * 2 - int(targets is None),
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[
                :, -states.shape[1] + int(targets is None) :, :
            ]
        elif (
            actions is None and self.model_type == "naive"
        ):  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(
            self.global_pos_emb, batch_size, dim=0
        )  # batch_size, traj_length, n_embd

        position_embeddings = (
            torch.gather(
                all_global_pos_emb,
                1,
                torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1),
            )
            + self.pos_emb[:, : token_embeddings.shape[1], :]
        )

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits_lst = []
        loss_total = 0
        if actions is not None:
            b_start = 0
            b_end = 1
            while b_end <= 15:
                vocab_size = self.config.vocab_size_lst[b_start]
                x_new = x[:, b_end + 1 :: 17, :]
                if b_start in self.config.multi_output_lst:
                    self.vocab_size = vocab_size
                    if vocab_size == 40:
                        # logits = self.multi_head_1(x_new)
                        logits = self.multi_head_2(x_new)
                    else:
                        logits = self.multi_head_2(x_new)
                    logits_lst.append(logits)
                    targets_singe = (
                        targets[:, :, :, b_start:b_end, :].squeeze().contiguous()
                    )
                    targets_singe = self.targets_encoder(
                        targets_singe.reshape(2, 1, 15, 15)
                        .type(torch.float32)
                        .contiguous()
                    )
                    loss = None
                    # if we are given some desired targets also calculate the loss
                    if targets_singe is not None:
                        targets_singe = targets_singe.long()
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            targets_singe.reshape(-1),
                        )
                        loss_total = loss_total + loss
                else:
                    if vocab_size == 1200:
                        logits = self.head_3(x_new)
                    else:
                        logits = self.head(x_new)
                    logits_lst.append(logits)
                    targets_singe = (
                        targets[:, :, :, b_start:b_end, 0:1].squeeze().contiguous()
                    )
                    loss = None
                    # if we are given some desired targets also calculate the loss
                    if targets_singe is not None:
                        targets_singe = targets_singe.long()
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            targets_singe.reshape(-1),
                        )
                        loss_total = loss_total + loss
                b_start += 1
                b_end += 1
        elif actions is None and self.model_type == "reward_conditioned":
            b_start = 0
            b_end = 1
            loss_total = 0
            logits_lst = []
            while b_end <= 15:
                vocab_size = self.config.vocab_size_lst[b_start]
                x_new = x[:, b_end + 1 :: 17, :]
                if b_start in self.config.multi_output_lst:
                    self.vocab_size = vocab_size
                    logits = self.multi_head_2(x_new)
                    logits = logits.repeat(15, 1, 1)
                    logits_lst.append(logits)
                else:
                    if vocab_size == 1200:
                        logits = self.head_3(x_new)
                    else:
                        logits = self.head(x_new)
                    logits_lst.append(logits)
                b_start += 1
                b_end += 1
        loss = loss_total / 15
        # if actions is not None and self.model_type == 'reward_conditioned':
        #     logits = logits[:, 2::3, :]  # only keep predictions from state_embeddings
        # elif actions is None and self.model_type == 'reward_conditioned':
        #     logits = logits[:, 1:, :]
        # elif actions is not None and self.model_type == 'naive':
        #     logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        # elif actions is None and self.model_type == 'naive':
        #     logits = logits # for completeness
        # else:
        #     raise NotImplementedError()

        # self.sample(logits)
        return logits_lst, loss
