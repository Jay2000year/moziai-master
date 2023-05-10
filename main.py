# -*- coding:utf-8 -*-
# File name : main.py
# Create date : 2022/9/20
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.0

import os
import argparse
import pyproj
from mozi_ai_sdk.AA_command_agent.env import Environment
from mozi_ai_sdk.AA_command_agent import etc
from mozi_ai_sdk.AA_command_agent.theater_command.theater_agent import TheaterCommand
from mozi_ai_sdk.AA_command_agent.brigade_command.brigade_agent import BrigadeCommand
from mozi_ai_sdk.AA_command_agent.battalion_command.battalion_agent import (
    BattalionCommand,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mozi_path", type=str, default="D:\\ZT-mozi\\Mozi\\MoziServer\\bin"
)
parser.add_argument("--side_name", type=str, default="联邦")

cmd_structure = {
    "西部战区": {
        "第72防空导弹旅": {  # 'S-400一营': '18e3ae42-e5d0-4d44-af6d-b65636ee9cb4',
            # 'S-400二营': 'bc0af20c-a2be-49ed-9b4b-d4e6baeefd72',
            "S-400三营": "96e5710a-61f0-4468-b7c5-8b1afb030e1e"
        }
    }
}


def init_theater_cmd_structure(side, structure):
    """
    初始化分层规则体架构，完成战区级规则体及下属各层级的规则体的实例化与编制目前仅支持单个战区级规则体，但战区级规则体麾下支持多个旅级规则体同理旅级规则体麾下支持多个营级规则体
    参数：
        structure：dict，包含了各层级规则体的名称、附属嵌套关系以及各营对应活动单元的GUID
    返回：
        theater_agent：战区级规则体，TheaterCommand类的实例化对象
    """
    for k1, v1 in structure.items():
        theater_agent = TheaterCommand(k1)
        for k2, v2 in v1.items():
            bde = BrigadeCommand(k2)
            for k3, v3 in v2.items():
                bn = BattalionCommand(k3, side.get_unit_by_guid(v3))
                bn.superior_command = bde
                bde.subordinate_command.append(bn)
            bde.superior_command = theater_agent
            theater_agent.subordinate_command.append(bde)
    return theater_agent


def run(env, side_name):
    """
    规则体运行的起始函数
    参数：
        env: 墨子环境类的实例化对象
        side_name: str，推演方名称
    返回：
        None
    """
    # 连接服务器，产生mozi_server
    env.start()

    # 重置函数，加载想定,拿到想定发送的数据
    env.scenario = env.reset()
    side = env.scenario.get_side_by_name(side_name)

    # 初始化分层规则体
    agent = init_theater_cmd_structure(side, cmd_structure)

    step_count = 0
    while True:
        # 更新动作
        agent.step(side)
        env.step()
        print(f"'推演步数：{step_count}")
        step_count += 1
        if env.is_done():
            print("推演已结束！")
            os.system("pause")
        else:
            pass


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["MOZIPATH"] = args.mozi_path
    env = Environment(
        etc.SERVER_IP,
        etc.SERVER_PORT,
        etc.PLATFORM,
        etc.SCENARIO_NAME,
        etc.SIMULATE_COMPRESSION,
        etc.DURATION_INTERVAL,
        etc.SYNCHRONOUS,
        etc.app_mode,
    )

    run(env, args.side_name)
