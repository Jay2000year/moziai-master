from mingpt import trainer_mozi
import torch

from mingpt.trainer_mozi import Args

args = Args("breakout", 123)
env = trainer_mozi.Env(args)
print(env.ale.getMinimalActionSet())
print(env.actions)
env.eval()
state = env.reset()
state = (
    state.type(torch.float32).to(torch.cuda.current_device()).unsqueeze(0).unsqueeze(0)
)
state, reward, done = env.step(2)
print(state, reward, done)
