# -*- coding:utf-8 -*-
# File name : battalion_config.py
# Create date : 2022/8/30
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.0
import pandas as pd
from enum import Enum

# 火控通道数量字典
FireControlChannels = {"S-400_40N6": 20, "S-400_48N6DM": 20, "HQ-9A": 12}

# 弹药飞行地面速度字典(km/h)
MunitionVelocity = {"40N6": {"lowalt": 4122, "highalt": 5706}}

# 弹药名称与DBGUID对应关系表
munition = pd.DataFrame(
    data=[
        ["40N6", "$hsfw-dataweapon-00000000002104"],
        ["9M313", "$hsfw-dataweapon-00000000001153"],
    ],
    columns=["name", "DBGUID"],
)


def get_munition_name(str_dbguid):
    return munition.loc[munition["DBGUID"] == str_dbguid, "name"][0]


def get_munition_dbguid(str_name):
    return munition.loc[munition["name"] == str_name, "DBGUID"][0]


# 火力分配权重值与决策阈值字典，通过配置键对应的值来实现不同的策略偏好
EPD = {
    "S-400_40N6": {
        "Aircraft": 10,
        "Missile": 5,
        "HighAlt": 1,
        "MidAlt": 2,
        "LowAlt": 3,
        "HighSpd": 5,
        "MidSpd": 3,
        "LowSpd": 1,
        "FarRange": 1,
        "MidRange": 10,
        "CloseRange": 20,
        "UnknowRange": 3,
        "HeadingSelf": 5,
        "HeadingProtect": 10,
        "OtherHeading": 0,
        "HighAltThreshold": 10973,
        "LowAltThreshold": 3657,
        "HighSpdThreshold": 1200,
        "LowSpdThreshold": 648,
        "CloseRangeThreshold": 60000,
        "FarRangeThreshold": 200000,
    }
}


class CombatMode(Enum):
    AMBUSH = 1
    GENERAL = 2
    TGTPROTECT = 3


class EMCONMode(Enum):
    RADARONLY = 1
    OECMONLY = 2
    BOTH = 3
    SILENCE = 4
