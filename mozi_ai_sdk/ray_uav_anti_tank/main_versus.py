import argparse
import collections
import gym
import json
import os
from pathlib import Path
from gym.spaces import Tuple, Dict, Discrete, Box

import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_CONFIG
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.agents.trainer import Trainer

from mozi_ai_sdk.ray_uav_anti_tank.env.env_uav import UAVEnv


parser = argparse.ArgumentParser()
parser.add_argument("--avail_ip_port", type=str, default="127.0.0.1:6060")
parser.add_argument("--platform_mode", type=str, default="eval")
parser.add_argument(
    "--mozi_server_path", type=str, default="D:\\ZT-mozi\\Mozi\\MoziServer\\bin"
)

print(ray.__version__)


def run(
    checkpoint=None, evaluate_episodes=10, avail_ip_port=None, platform_mode="eval"
):

    ddpg_extra_config = {
        "framework": "torch",
        "train_batch_size": 8,
        "learning_starts": 16,
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
        "127.0.0.1:6060",
    ]
    env_config = {
        "mode": "eval",
        "side_name": "红方",
        "enemy_side_name": "蓝方",
        "avail_docker_ip_port": server_ip_port,
        # 'sever_docker_dict': SERVER_DOCKER_DICT,
    }

    config = {
        "env_config": env_config,
        "num_workers": 0,
        # "bcmodel": {
        #     "custom_model": "autoregressive_model",
        #     "custom_action_dist": "binary_autoreg_dist",
        # },
        # "lr": tune.uniform(5e-6, 5e-4),
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": policies_to_train,
        },
        "framework": "torch",
    }

    agent = DDPGTrainer(env=UAVEnv, config=config)
    agent.restore(checkpoint)
    rollout(agent, UAVEnv, evaluate_episodes)
    agent.stop()


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    return True


def rollout(
    agent,
    env_name,
    num_episodes=0,
):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        env = gym.make(env_name)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent)
            )
        use_lstm = {DEFAULT_POLICY_ID: False}

    # action_init = {
    #     p: flatten_to_single_ndarray(m.action_space.sample())
    #     for p, m in policy_map.items()
    # }
    action_init = {p: flatten_to_single_ndarray(0) for p, m in policy_map.items()}

    episodes = 0
    while keep_going(episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]]
        )
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]]
        )
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        reward_total = 0.0
        while not done and keep_going(episodes, num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            explore=False,
                        )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            explore=False,
                        )
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
            obs = next_obs

        if done:
            episodes += 1


if __name__ == "__main__":
    args = parser.parse_args()

    if args.platform_mode == "versus":
        pass
    elif args.platform_mode == "eval":
        # 设置墨子可执行程序的路径，用于代码启动墨子
        os.environ["MOZIPATH"] = args.mozi_server_path
    # ray.init(address=args.address, _redis_password=args.redis_password)
    ray.init()

    # checkpoint_path = os.path.join(Path(__file__).parent.parent, 'checkpoint/checkpoint_2/checkpoint-2')
    checkpoint_path = os.path.join(Path(__file__).parent, "checkpoint")
    checkpoint_dir = None
    param_dir = None
    for root_dir, dirs, files in os.walk(checkpoint_path):
        if "params.json" in files:
            param_dir = os.path.join(root_dir, "params.json")
        for file in files:
            if "checkpoint-" in file:
                checkpoint_dir = os.path.join(root_dir, file)
                print(checkpoint_dir)
                # by 张志高
                checkpoint_dir = checkpoint_dir.replace(".tune_metadata", "")
                break
        if checkpoint_dir:
            break
    if param_dir:
        with open(param_dir, "r") as fp:
            params = json.load(fp)
    else:
        raise ValueError("参数路径为空")
    run(
        checkpoint=checkpoint_dir,
        evaluate_episodes=10,
        avail_ip_port=args.avail_ip_port,
        platform_mode=args.platform_mode,
    )
