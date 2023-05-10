# 时间 : 2021/10/9 16:25
# 作者 : Dixit
# 文件 : main_train.py
# 说明 :
# 项目 : moziai
# 版权 : 北京华戍防务技术有限公司

import argparse
from gym.spaces import Tuple, Dict, Discrete, Box
import os

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_CONFIG
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.test_utils import check_learning_achieved


from mozi_ai_sdk.ray_uav_anti_tank.env.env_uav import UAVEnv

parser = argparse.ArgumentParser()
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--mixed-torch-tf", action="store_true")
parser.add_argument("--stop-iters", type=int, default=20)
parser.add_argument("--stop-reward", type=float, default=150.0)
parser.add_argument("--stop-timesteps", type=int, default=100000)

parser.add_argument(
    "--mozi_server_path", type=str, default="D:\\mozi_4p\\mozi\\Mozi\\MoziServer\\bin"
)
parser.add_argument("--side", type=str, default="红方")
parser.add_argument("--platform_mode", type=str, default="development")

# 创建的docker个数应该是num_workers+1，比如num_workers=3，那么需要创建4个docker
SERVER_DOCKER_DICT = {
    # "127.0.0.1": 1,
    "123.57.142.120": 1,
}  # {'8.140.121.210': 2, '123.57.137.210': 2}


if __name__ == "__main__":

    args = parser.parse_args()
    if args.platform_mode == "train":
        pass
    elif args.platform_mode == "development":
        ray.init(address="auto")
        # ray.init(local_mode=True)
    else:
        os.environ["MOZIPATH"] = args.mozi_server_path
        ray.init(local_mode=True)

    ddpg_extra_config = {
        "framework": "torch",
        "train_batch_size": 512,
        "learning_starts": 1000,
        "exploration_config": {
            # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
            # actions (after a possible pure random phase of n timesteps).
            "type": "OrnsteinUhlenbeckNoise",
            # For how many timesteps should we return completely random actions,
            # before we start adding (scaled) noise?
            "random_timesteps": 1000,
            # The OU-base scaling factor to always apply to action-added noise.
            "ou_base_scale": 0.1,
            # The OU theta param.
            "ou_theta": 0.15,
            # The OU sigma param.
            "ou_sigma": 0.2,
            # The initial noise scaling factor.
            "initial_scale": 1.0,
            # The final noise scaling factor.
            "final_scale": 0.02,
            # Timesteps over which to anneal scale (from initial to final values).
            "scale_timesteps": 10000,
        },
    }
    ddpg_config = Trainer.merge_trainer_configs(
        DDPG_CONFIG, ddpg_extra_config, _allow_unknown_configs=True
    )

    obs_space = Dict(
        {
            "obs": Box(float("-inf"), float("inf"), shape=(14,)),
            # "action_mask": Box(0, 1, shape=(self.action_size,)),
        }
    )

    act_space = Box(-1, 1, shape=(1,))

    policies = {
        "uav_ddpg_policy": (DDPGTorchPolicy, obs_space, act_space, ddpg_config),
    }

    def policy_mapping_fn(agent_id):
        if agent_id == "agent_0":
            return "uav_ddpg_policy"
        else:
            raise NotImplementedError

    policies_to_train = [
        "uav_ddpg_policy",
    ]

    server_ip_port = [
        # "127.0.0.1:6060",
        "123.57.142.120:6060"
    ]
    env_config = {
        "mode": "development",
        "side_name": "红方",
        "enemy_side_name": "蓝方",
        # 'avail_docker_ip_port': server_ip_port,
        "sever_docker_dict": SERVER_DOCKER_DICT,
    }

    config = {
        "env": UAVEnv,
        "num_workers": 10,
        "lr": tune.uniform(5e-5, 5e-4),
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": policies_to_train,
        },
        "framework": "torch",
        "env_config": env_config,
        # "num_sgd_iter": 10,
        # "sgd_minibatch_size": 8,
        # "rollout_fragment_length": 32,
        "train_batch_size": 512,
        "learning_starts": 1000,
        "target_network_update_freq": 1500,
        "timesteps_per_iteration": 32,
    }

    # stop = {
    #     "training_iteration": args.stop_iters,
    #     "timesteps_total": args.stop_timesteps,
    #     # "episode_reward_mean": args.stop_reward,
    # }
    algo = HyperOptSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=1)
    scheduler = AsyncHyperBandScheduler(max_t=3000)
    results = tune.run(
        DDPGTrainer,
        metric="episode_reward_mean",
        mode="max",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=1,
        checkpoint_freq=1,
        keep_checkpoints_num=10,
        config=config,
        # stop=stop
    )

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
