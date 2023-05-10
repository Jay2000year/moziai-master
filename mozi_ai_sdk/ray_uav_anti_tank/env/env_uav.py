# 时间 : 2021/10/9 15:34
# 作者 : Dixit
# 文件 : env_uav.py
# 说明 :
# 项目 : moziai
# 版权 : 北京华戍防务技术有限公司

from collections import namedtuple

from mozi_ai_sdk.ray_uav_anti_tank.env.env import EnvUavAntiTank as Environment
from mozi_ai_sdk.ray_uav_anti_tank.env import etc

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.remote_handle_docker import restart_mozi_container

import sys
import zmq
import time

# zmq init
zmq_context = zmq.Context()
# ray request port
restart_requestor = zmq_context.socket(zmq.REQ)
Function = namedtuple("Function", ["type", "function"])
FEATS_MAX_LEN = 350
MAX_DOCKER_RETRIES = 3


def restart_container(schedule_addr, schedule_port, _training_id, docker_ip_port):
    # 训练5轮后，重启docker
    try:
        message = {}
        message["zmq_command"] = "restart_training_container"
        message["docker_ip_port"] = docker_ip_port
        message["training_id"] = _training_id
        restart_requestor.connect(
            "tcp://%s:%s" % (str(schedule_addr), str(schedule_port))
        )
        restart_requestor.send_pyobj(message)
        recv_msg = restart_requestor.recv_pyobj()
        assert type(recv_msg) == str
        if "OK" in recv_msg:
            pass
        else:
            sys.exit(1)
        return docker_ip_port
    except Exception:
        print("fail restart mozi docker!")
        sys.exit(1)


class UAVEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.steps = None
        self.env_config = env_config
        # windows下训练
        # self.env_config['avail_docker_ip_port'] = ['127.0.0.1:6060', ]
        self.reset_nums = 0
        self._get_env()
        self.side_name = env_config["side_name"]

    def step(self, action):
        reward = 0
        if self.env_config["mode"] in ["train", "development"]:
            obs, reward, done = self.env.execute_action(action["agent_0"])
        elif self.env_config["mode"] in ["versus", "eval"]:
            obs, reward, done = self.env.execute_action(action["agent_0"])

        reward = {"agent_0": reward}
        obs = {"agent_0": {"obs": obs}}
        self.steps += 1
        print(f"steps:{self.steps}  action:{action}  reward:{reward}")

        return obs, reward, {"__all__": done, "agent_0": done}, {}

    def reset(self):
        state_now = self._get_initial_state()
        self.steps = 0

        obs = {"agent_0": {"obs": state_now}}

        return obs

    def _get_env(self):
        if self.env_config["mode"] == "train":
            self.schedule_addr = self.env_config["schedule_addr"]
            self.schedule_port = self.env_config["schedule_port"]
            scenario_name = etc.SCENARIO_NAME
            platform = "linux"
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config["mode"] == "development":
            scenario_name = etc.SCENARIO_NAME
            platform = "linux"
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config["mode"] == "versus":
            scenario_name = etc.SCENARIO_NAME
            platform = "linux"
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config["mode"] == "eval":
            scenario_name = etc.EVAL_SCENARIO_NAME
            platform = "windows"
            self._create_env(platform, scenario_name=scenario_name)

            # platform = 'linux'
            # self._create_env(platform)
        else:
            raise NotImplementedError

    def _create_env(self, platform, scenario_name=None):
        for _ in range(MAX_DOCKER_RETRIES):
            # noinspection PyBroadException
            try:
                self.env = Environment(
                    etc.SERVER_IP,
                    etc.SERVER_PORT,
                    platform,
                    scenario_name,
                    etc.SIMULATE_COMPRESSION,
                    etc.DURATION_INTERVAL,
                    etc.SYNCHRONOUS,
                    etc.app_mode,
                )
                # by dixit
                if self.env_config["avail_docker_ip_port"]:
                    self.avail_ip_port_list = self.env_config["avail_docker_ip_port"]
                else:
                    raise Exception("no avail port!")
                # self.self.reset_nums = 0
                self.ip_port = self.avail_ip_port_list[0]
                print(self.ip_port)
                self.ip = self.avail_ip_port_list[0].split(":")[0]
                self.port = self.avail_ip_port_list[0].split(":")[1]
                self.ip_port = f"{self.ip}:{self.port}"
                self.env.start(self.ip, self.port)
                break
            except Exception:
                continue

    def _get_initial_state(self):
        """
        dixit 2021/3/22
        每5局重启墨子，获取初始态势
        """

        self.reset_nums += 1
        if self.env_config["mode"] in ["train", "development"]:
            if self.reset_nums % 5 == 0:
                docker_ip_port = self.avail_ip_port_list[0]
                for _ in range(MAX_DOCKER_RETRIES):
                    # noinspection PyBroadException
                    try:
                        if self.env_config["mode"] == "train":
                            restart_container(
                                self.schedule_addr,
                                self.schedule_port,
                                self.env_config["training_id"],
                                docker_ip_port,
                            )
                        else:
                            restart_mozi_container(docker_ip_port)
                        self.env = Environment(
                            etc.SERVER_IP,
                            etc.SERVER_PORT,
                            "linux",
                            etc.SCENARIO_NAME,
                            etc.SIMULATE_COMPRESSION,
                            etc.DURATION_INTERVAL,
                            etc.SYNCHRONOUS,
                            etc.app_mode,
                        )
                        self.env.start(self.ip, self.port)
                        break
                    except Exception:
                        print(
                            f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，第{_}次重启docker失败！！！"
                        )
                        continue
                state_now, _ = self.env.reset(self.side_name)
                return state_now
            else:
                state_now, _ = self.env.reset(self.side_name)
                return state_now
        else:
            state_now, _ = self.env.reset(self.side_name)
            return state_now
