# -*- coding:utf-8 -*-
# File name : ballistic_test.py
# Create date : 2022/10/8
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.0

import os
import argparse
import sys

from mozi_ai_sdk.AA_command_agent.env import Environment
from mozi_ai_sdk.AA_command_agent import etc
import pandas as pd
from pandas import DataFrame
from mozi_utils import geo

parser = argparse.ArgumentParser()
parser.add_argument("--mozi_path", type=str, default="D:\\Mozi\\MoziServer\\bin")
parser.add_argument("--side_name", type=str, default="联邦")


def get_distance(activeunit_a, activeunit_b):
    return geo.get_two_point_distance(
        activeunit_a.dLongitude,
        activeunit_a.dLatitude,
        activeunit_b.dLongitude,
        activeunit_b.dLatitude,
    )


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

    data = {}
    sam_activeunit = side.get_unit_by_guid("96e5710a-61f0-4468-b7c5-8b1afb030e1e")

    step_count = 0
    flag = True
    while True:
        # 更新动作
        env.step()

        if step_count >= 5:
            if flag:
                for guid, wpn in side.weapons.items():
                    data.update(
                        {
                            guid: DataFrame(
                                columns=["distance", "altitude", "speed", "time"]
                            )
                        }
                    )
                flag = False
            for guid, wpn in side.weapons.items():
                df = data[guid]
                data[guid] = pd.concat(
                    [
                        df,
                        DataFrame(
                            data=[
                                [
                                    get_distance(sam_activeunit, wpn),
                                    wpn.fCurrentAltitude_ASL,
                                    wpn.fCurrentSpeed,
                                    step_count * 2,
                                ]
                            ],
                            columns=["distance", "altitude", "speed", "time"],
                            index=[step_count],
                        ),
                    ]
                )
            if step_count == 125:
                for guid, df in data.items():
                    df.to_excel(
                        os.path.dirname(sys.argv[0]) + "\\data" + guid + ".xlsx"
                    )

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
        "弹道测试.scen",
        etc.SIMULATE_COMPRESSION,
        etc.DURATION_INTERVAL,
        etc.SYNCHRONOUS,
        etc.app_mode,
    )

    run(env, args.side_name)
