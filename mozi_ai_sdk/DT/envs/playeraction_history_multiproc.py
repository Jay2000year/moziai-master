# -*- coding:utf-8 -*-
# File name : playeraction_history_multiproc.py
# Create date : 2022/8/1
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.2

import re
import os
import time
import datetime
import pandas as pd
from multiprocessing import Process
from pandas import DataFrame
from envs.load_data_1 import ConstructionMatrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_LuaHistory",
    type=str,
    default=r"D:\BaiduNetdiskDownload\JSPT0718\JSPT0718\DeductionServer\DeductionServer\ServerInt\bin64\Analysis\海峡风暴_202208231427\1\LuaHistory_2022-08-23_14_27_21.txt",
)
parser.add_argument(
    "--file_red_UnitMission",
    type=str,
    default=r"D:\BaiduNetdiskDownload\JSPT0718\JSPT0718\DeductionServer\DeductionServer\ServerInt\bin64\Analysis\海峡风暴_202208231427\1\红方_UnitMission.csv",
)
parser.add_argument(
    "--file_blue_UnitMission",
    type=str,
    default=r"D:\BaiduNetdiskDownload\JSPT0718\JSPT0718\DeductionServer\DeductionServer\ServerInt\bin64\Analysis\海峡风暴_202208231227\1\蓝方_UnitMission.csv",
)
parser.add_argument("--save_dir", type=str, default="D:\\mozi_luahistory\\202281_13\\")

# lua指令筛选白名单（剔除不需要的lua指令）
whitelist = [
    "ScenEdit_AddMission",
    "ScenEdit_SetMission",
    "ScenEdit_DeleteMission",
    "ScenEdit_SetDoctrine",
    "ScenEdit_UnAssignUnitFromMission",
    "ScenEdit_SetEMCON",
    "ScenEdit_AssignUnitAsTarget",
    "ScenEdit_RemoveUnitAsTarget",
    "Hs_LuaSetDoctrineWRA",
    "Hs_LUA_AssignUnitToMission",
    "Hs_WeaponUsageRules",
    "Hs_ResetDoctrine",
]


# 单元编码字典
units_encoder_dict = {
    "d030f19b-424f-40b5-9c3d-297170c7deeb": 1,
    "7e6b815e-a392-4919-9080-e3d6de487a59": 2,
    "8b9b9d15-35eb-4213-b883-1dc608d0d1c0": 3,
    "d0e68300-702b-4f1a-82b6-e63a886eac77": 4,
    "df633307-5a10-490c-899d-d1cc1b156942": 5,
    "b377474e-8251-4b7a-94c1-6919b289c762": 6,
    "00b16e1a-53b8-4267-803b-64bc6a7bc1da": 7,
    "ed4aa324-52ca-4eca-a9b1-e7ff70acb07b": 8,
    "ecd52aec-5d45-4326-937e-fef7d5eb0079": 9,
    "123df277-4bee-4adc-b79d-2c7403a996ed": 10,
    "9ffc1510-5189-4c4b-8f46-a6fe208948f2": 11,
    "b0a2818a-51f5-4c6d-aaeb-c6f76fbe3268": 12,
    "eb74a25a-e644-4d26-b711-0ff5da8ae4ac": 13,
    "20ddc9cc-3000-4d18-a127-7ceffecd4d89": 14,
    "bfd543f7-ed60-4940-9c9e-06cb4ca22c7d": 15,
    "c8dca793-a20f-4056-af88-e354289fcbcd": 16,
    "ccbb06d0-0557-4a26-ab88-440c122340c7": 18,
    "1f193bb2-6f4b-4e04-9813-172c00e81006": 19,
    "f18b8f8d-746c-4791-a796-0ddedb6feda2": 21,
    "d3fd73d8-6863-4ba9-9f74-9c7cb4130d54": 22,
    "53edb3af-f376-4d98-96ad-14b26fca2683": 23,
    "703f52fa-92f7-4213-887a-a6a98aadf80a": 24,
    "597c0e77-a407-497c-b8c9-1a3663c5e9db": 25,
    "fdb4b2a6-6404-431c-8ee6-dd3b4b5ee8b2": 26,
    "a02457ff-a878-44bf-b6ea-cd9ff94a4e8b": 27,
    "699e669b-ecd2-4f85-bd1c-dca79b9c53cd": 28,
    "909e75d3-bd8f-42c2-b81f-7957fc63c864": 29,
    "d16e7f09-64aa-4f80-965b-954fdb44bc8c": 30,
    "db2e89d5-db43-4a57-8e9b-330a19add8da": 31,
    "eadfb12a-9f46-43b5-9067-d1f44574b82d": 32,
    "08a672a4-f58f-4455-9c04-2a171381063e": 33,
    "c513dbd1-26a3-497a-9e63-3ac8dc6243d1": 34,
    "cf477819-139d-486a-8501-f1e7e814cd3a": 35,
    "38547fa7-1c85-4ce9-bea1-8068cd5af84d": 36,
    "6d829dba-2092-4b9d-a824-05ae1cb74c9b": 38,
    "e2ebe1aa-baf0-4ca8-8be2-8372a4a3707a": 39,
}


class CommandBuffer:
    def __init__(self, guid):
        """
        参数：guid：指令所属任务的guid
        """
        self.command_type = "Not_Operate"
        self.mission_guid = guid
        self.mission_type = "Not_Operate"
        self.unit_control = "Not_Operate"
        self.mission_units = []
        self.target_control = "Not_Operate"
        self.mission_targets = []
        self.patrol_control = "Not_Operate"
        self.patrol_zone = []
        self.precaution_control = "Not_Operate"
        self.precaution_area = []
        self.mission_starttime = "Not_Operate"
        self.emcon_radar = "Not_Operate"
        # self.emcon_sonar = 'Not_Operate'
        self.emcon_oecm = "Not_Operate"
        self.wra = "Not_Operate"

    def update_command(self, cmd_list):
        """
        功能：使用指令列表中的指令更新指令缓冲状态
        参数：cmd_list：由多条lua指令组成的列表
        """
        if len(cmd_list) != 0:
            for cmd in cmd_list:
                if re.search(r"ScenEdit\_SetMission.+starttime", cmd) is not None:
                    self.mission_starttime = re.search(
                        r"starttime=\'(\d{2}\/\d{2}\/\d{4} \d{2}\:\d{2}\:\d{2})\'", cmd
                    ).group(1)
                    if self.command_type != "Create":
                        self.command_type = "Modify"
                if re.search(r"Hs\_LUA\_AssignUnitToMission", cmd) is not None:
                    self.mission_units.append(
                        re.search(r"\(\'([^\'\,]+)\'", cmd).group(1)
                    )
                    if self.command_type != "Create":
                        self.command_type = "Modify"
                    self.unit_control = "Update"
                if re.search(r"ScenEdit\_UnAssignUnitFromMission", cmd) is not None:
                    self.mission_units.remove(
                        re.search(r"\(\'([^\'\,]+)\'\,.+\)", cmd).group(1)
                    )
                    if self.command_type != "Create":
                        self.command_type = "Modify"
                    self.unit_control = "Update"
                # if re.search(r'ScenEdit\_AssignUnitAsTarget', cmd) is not None:
                # self.mission_targets.append(re.search(r'\(\{\'([^\'\,]+)\' \}', cmd).group(1))
                # self.target_control = 'Update'
                # if re.search(r'ScenEdit\_RemoveUnitAsTarget', cmd) is not None:
                # self.mission_targets.remove(re.search(r'\(\{\'([^\'\,]+)\' \}', cmd).group(1))
                # self.target_control = 'update'
                if re.search(r"ScenEdit\_SetEMCON\(.+Radar", cmd) is not None:
                    self.emcon_radar = re.search(r"\'Radar\=([^\'\,]+)\'", cmd).group(1)
                    if self.command_type != "Create":
                        self.command_type = "Modify"
                if re.search(r"ScenEdit\_SetEMCON\(.+OECM", cmd) is not None:
                    self.emcon_oecm = re.search(r"\'OECM\=([^\'\,]+)\'", cmd).group(1)
                    if self.command_type != "Create":
                        self.command_type = "Modify"
                # if re.search(r'ScenEdit\_SetEMCON\(.+Sonar', cmd) is not None:
                # self.emcon_sonar = re.search(r'\'Sonar\=([^\'\,]+)\'').group(1)
                if (
                    re.search(r"Hs\_LuaSetDoctrineWRA.+MISSION\=.+qty\_salvo", cmd)
                    is not None
                ):
                    self.wra = re.search(r"qty\_salvo\=\'([^\'\,]+)\'", cmd).group(1)
                    if self.command_type != "Create":
                        self.command_type = "Modify"
                if re.search(r"ScenEdit\_AddMission", cmd) is not None:
                    self.command_type = "Create"
                    if re.search(r"Patrol", cmd) is not None:
                        if re.search(r"type=\'AAW\'", cmd) is not None:
                            self.mission_type = "Combat_Air_Patrol"
                        # elif re.search(r'type=\'SUR\_SEA\'', cmd) is not None:
                        # self.mission_type = 'Anti Ship Patrol'
                    elif re.search(r"Strike", cmd) is not None:
                        if re.search(r"SEA", cmd) is not None:
                            self.mission_type = "Anti_Ship_Strike"

    def set_patrol_zone(self, area_list):
        """
        功能：更新任务缓冲的巡逻区记录
        参数：area_list：巡逻区参考点坐标元组组成的列表
        """
        if self.command_type == "Not_Operate":
            self.command_type = "Modify"
        self.patrol_control = "Update"
        self.patrol_zone = area_list

    def set_precaution_area(self, area_list):
        """
        功能：更新任务缓冲的警戒区记录
        参数：area_list：警戒区参考点坐标元组组成的列表
        """
        if self.command_type == "Not_Operate":
            self.command_type = "Modify"
        self.precaution_control = "Update"
        self.precaution_area = area_list

    def update_target_list(self, target_list):
        """
        功能：更新任务缓冲的目标记录
        参数：target_list：任务目标组成的列表
        """
        if self.command_type == "Not_Operate":
            self.command_type = "Modify"
        self.mission_targets = target_list
        self.target_control = "Update"

    def reset_cmd_buffer(self):
        """
        功能：重置任务缓冲
        """
        self.command_type = "Not_Operate"
        self.mission_type = "Not_Operate"
        self.unit_control = "Not_Operate"
        self.target_control = "Not_Operate"
        self.mission_targets = []
        self.patrol_control = "Not_Operate"
        self.patrol_zone = []
        self.precaution_control = "Not_Operate"
        self.precaution_area = []
        self.mission_starttime = "Not_Operate"
        self.emcon_radar = "Not_Operate"
        # self.emcon_sonar = 'Not_Operate'
        self.emcon_oecm = "Not_Operate"
        self.wra = "Not_Operate"

    def command_output(self):
        """
        功能：将任务缓冲中的记录进行输出
        """
        return {
            "cmd_type": self.command_type,
            "guid": self.mission_guid,
            "type": self.mission_type,
            "unit_control": self.unit_control,
            "units": self.mission_units if self.unit_control == "Update" else [],
            "target_control": self.target_control,
            "targets": self.mission_targets,
            "patrol_control": self.patrol_control,
            "patrol": self.patrol_zone,
            "precaution_control": self.precaution_control,
            "precaution": self.precaution_area,
            "starttime": self.mission_starttime,
            "emcon_radar": self.emcon_radar,
            # 'emcon_sonar': self.emcon_sonar,
            "emcon_oecm": self.emcon_oecm,
            "wra": self.wra,
        }


class DataProcessor:
    def __init__(self, side, lua_file, mssn_history_file, save_dir):
        """
        参数：side：处理数据所属推演方，'red'或'blue'
             lua_file：LuaHistory.txt文件位置，
             mssn_history_file：UnitMission.csv文件位置
             save_dir：处理过程记录文件，False为不保存
        """
        self.side = side
        self.save_dir = save_dir
        self.fileobj = open(lua_file, "r", encoding="utf-8")
        self.lua_content = self.fileobj.read()
        self.fileobj.close()

        # 读取UnitMission.csv获得的DataFrame
        self.mssn_df = pd.read_csv(mssn_history_file, encoding="utf-8")

        # 逐条lua指令组成的列表
        self.lua_list = []

        # 与step进行对应的lua指令构成的DataFrame
        self.lua_df = None

        # 某一方所有任务记录的DataFrame组成的字典
        self.mssn_dfdict = None

        # 某一方所有与step对应的任务记录DataFrame组成的字典
        self.mssn_inStep_dfdict = {}

        # 将lua记录拆分成逐条lua指令组成的列表
        self.lua_list = re.findall(r".+", self.lua_content)

        # 从lua记录中获取红蓝方guid
        if self.side == "red":
            self.side_guid = re.search(
                r"红方 : ScenEdit\_AddMission \(\'(.*?)\',", self.lua_content
            ).group(1)
        elif self.side == "blue":
            self.side_guid = re.search(
                r"蓝方 : ScenEdit\_AddMission \(\'(.*?)\',", self.lua_content
            ).group(1)

        # 从lua记录中获取红蓝方任务guid列表
        # 无法应对选手“过河拆桥”在推演后期删除无用任务的情况，故弃用该方法
        # self.red_mssnguid_list = list(
        # set(re.findall(r'ScenEdit\_SetMission\(\'' + self.red_side_guid + r'\', \'(.*?)\'', self.lua_content)))
        # for lua in self.lua_content:
        # if re.search(r'ScenEdit\_DeleteMission', lua) is not None and\
        # re.search(self.red_side_guid, lua) is not None:
        # self.red_mssnguid_list.remove(re.search(r'\(\'([^\'\,]+)\'', lua).group(1))
        # self.blue_mssnguid_list = list(
        # set(re.findall(r'ScenEdit\_SetMission\(\'' + self.blue_side_guid + r'\', \'(.*?)\'', self.lua_content)))
        # for lua in self.lua_content:
        # if re.search(r'ScenEdit\_DeleteMission', lua) is not None and\
        # re.search(self.blue_side_guid, lua) is not None:
        # self.blue_mssnguid_list.remove(re.search(r'\(\'([^\'\,]+)\'', lua).group(1))

        # 从任务记录中获取红蓝方任务guid列表
        self.mssnguid_list = list(set(self.mssn_df["MissionID"].tolist()))

    def fix_commands_without_timestamp(self):
        """
        功能：修复不带时间戳的lua指令，为其补充上一条指令的时间戳
        """
        timestamp = ""
        lua_list_fixed = []
        for v in self.lua_list:
            if re.match(r"\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}", v) is not None:
                timestamp = datetime.datetime.strptime(
                    re.match(r"\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}", v).group(),
                    "%Y-%m-%d %H:%M:%S",
                )
                lua_list_fixed.append(v)
            else:
                lua_list_fixed.append(
                    timestamp.strftime("%Y-%m-%d %H:%M:%S") + ".000" + v
                )
        self.lua_list = lua_list_fixed

    def filter_commands_by_whitelist(self, wl):
        """
        功能：根据lua指令白名单筛选lua指令记录
        参数：wl：{list：正则表达式pattern组成的列表}
        """
        lua_list = []
        for luastr in self.lua_list:
            for pattern in wl:
                if re.search(pattern, luastr) is not None:
                    lua_list.append(luastr)
        self.lua_list = lua_list

    def sort_commands_by_side(self):
        """
        功能：根据lua指令所属方名称/方guid/任务guid来分类指令所属方
        """
        lua_list = []
        # 遍历lua指令列表
        for lst in self.lua_list:
            flag = False
            # 判断指令是否作用于某一方的任务
            for guid in self.mssnguid_list:
                if re.search(guid, lst) is not None:
                    flag = True
            if self.side == "red":
                if (
                    re.search("红方", lst) is not None
                    or re.search(self.side_guid, lst) is not None
                    or flag
                ):
                    lua_list.append(lst)
            elif self.side == "blue":
                if (
                    re.search("蓝方", lst) is not None
                    or re.search(self.side_guid, lst) is not None
                    or flag
                ):
                    lua_list.append(lst)

        if self.save_dir:
            LuaHistory_fileobj = open(
                self.save_dir + self.side + "_LuaHistory.txt", "w"
            )
            # 将列表中的指令逐条写入文件
            for lua in lua_list:
                LuaHistory_fileobj.write(lua + "\n")

            LuaHistory_fileobj.close()

        self.lua_list = lua_list

    def sort_commands_by_timesteps(self):
        """
        功能：将lua指令与steps进行对应
        """
        # 设置初始步数
        steps = 1
        # 设置每步幅时长跨度
        step_stride = 15
        # 获取起始/步进时间戳
        timestamp_begin = datetime.datetime.strptime(
            re.match(
                r"\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}", self.lua_list[0]
            ).group(),
            "%Y-%m-%d %H:%M:%S",
        )
        timestamp_step = timestamp_begin + datetime.timedelta(seconds=step_stride)
        # 根据推演总时长计算结束时间戳
        timestamp_end = timestamp_begin + datetime.timedelta(minutes=100)
        # 计算steps总数
        total_steps = (timestamp_end - timestamp_begin).seconds // step_stride

        # 判断指令时间戳是否在step时间跨度内（如10:00:01-10:00:16）
        def is_commands_in_stride(lua, ts_begin, ts_step):
            lua_timestamp = datetime.datetime.strptime(
                re.match(r"\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}", lua).group(),
                "%Y-%m-%d %H:%M:%S",
            )
            return lua_timestamp.__ge__(ts_begin) and lua_timestamp.__lt__(ts_step)

        step_list = []
        temp_lua_list = []
        ts_begin_list = []
        ts_step_list = []
        luastr = ""
        while total_steps:
            # 遍历lua指令列表
            for v in self.lua_list:
                # 匹配条件：指令时间戳在step时间跨度内
                if is_commands_in_stride(v, timestamp_begin, timestamp_step):
                    luastr = luastr + v + "|"
            # 将匹配数据写入列表对应位置
            step_list.append(steps)
            temp_lua_list.append(luastr)
            ts_begin_list.append(timestamp_begin.strftime("%H:%M:%S"))
            ts_step_list.append(timestamp_step.strftime("%H:%M:%S"))
            # step、时间戳步进，lua记录缓存清空，循环步数-1
            steps += 1
            luastr = ""
            timestamp_begin = timestamp_begin + datetime.timedelta(seconds=step_stride)
            timestamp_step = timestamp_step + datetime.timedelta(seconds=step_stride)
            total_steps -= 1

        # 构造DataFrame
        data = {
            "timestamp_begin": ts_begin_list,
            "timestamp_step": ts_step_list,
            "lua_history": temp_lua_list,
        }
        self.lua_df = DataFrame(
            data,
            index=step_list,
            columns=["timestamp_begin", "timestamp_step", "lua_history"],
        )
        if self.save_dir:
            self.lua_df.to_excel(self.save_dir + self.side + "_luahistory_in_ts.xlsx")

    def filter_mssn_history(self):
        """
        功能：根据任务guid对UnitMission.csv进行筛选，并保留任务目标/任务区域改动时产生的记录
        """
        filter_dict = {}
        self.mssn_dfdict = filter_dict

        # 根据任务guid筛选出子DataFrame
        for s in self.mssnguid_list:
            filter_dict[s] = self.mssn_df.groupby("MissionID").get_group(s)
            # 删除不需要的列”TimelineID“、”MissionSide“，缩减数据大小
            filter_dict[s] = filter_dict[s].drop(columns="TimelineID")
            filter_dict[s] = filter_dict[s].drop(columns="MissionSide")

        # 去除子DataFrame中任务目标&巡逻区&警戒区完全一致的记录，只保留最后一条
        for k, v in filter_dict.items():
            filter_dict[k] = v.drop_duplicates(
                ["Attacktarget", "PatrolZone", "PrecautionArea"], keep="first"
            )

        if self.save_dir:
            for k, v in filter_dict.items():
                v.to_excel(self.save_dir + self.side + "_mssn_history_" + k + ".xlsx")

    def sort_mssn_history_by_timesteps(self):
        """
        功能：将任务记录与steps进行对应
        """
        mssn_dfdict = {}
        timestamp_begin = datetime.datetime.strptime(
            re.match(
                r"\d{4}\-\d{2}\-\d{2} (\d{2}\:\d{2}\:\d{2})", self.lua_list[0]
            ).group(1),
            "%H:%M:%S",
        )

        # 判断时间戳是否在step时间跨度内
        def is_timestamp_in_stride(ts, ts_begin, ts_step):
            return ts.__ge__(ts_begin) and ts.__lt__(ts_step)

        # 迭代所有任务记录DataFrame
        for k, v in self.mssn_dfdict.items():
            # 初始化与step对应的任务记录DataFrame
            mssnHis_inStep_df = DataFrame(
                columns=[
                    "timestamp_begin",
                    "timestamp_step",
                    "Time",
                    "MissionID",
                    "MissionClass",
                    "MissionName",
                    "Attacktarget",
                    "PatrolZone",
                    "PrecautionArea",
                ]
            )
            # 迭代steps
            for idx, row in self.lua_df.iterrows():
                # 获取step时间戳
                ts_b = datetime.datetime.strptime(row["timestamp_begin"], "%H:%M:%S")
                ts_s = datetime.datetime.strptime(row["timestamp_step"], "%H:%M:%S")
                # 初始化任务记录缓存（仅含时间戳的空行）
                mssn_history = DataFrame(
                    [
                        {
                            "timestamp_begin": row["timestamp_begin"],
                            "timestamp_step": row["timestamp_step"],
                            "Time": "",
                            "MissionID": "",
                            "MissionClass": "",
                            "MissionName": "",
                            "Attacktarget": "",
                            "PatrolZone": "",
                            "PrecautionArea": "",
                        }
                    ]
                )
                # 迭代每个任务记录
                for idx2, row2 in v.iterrows():
                    # 计算时间戳
                    td = re.match(r"\d\.(\d+)\:(\d+)\:(\d+)", row2["Time"])
                    timestamp = (
                        timestamp_begin
                        - datetime.timedelta(seconds=1)
                        + datetime.timedelta(
                            hours=int(td.group(1)),
                            minutes=int(td.group(2)),
                            seconds=int(td.group(3)),
                        )
                    )
                    # 判断任务记录时间戳是否处于step时间跨度内
                    if is_timestamp_in_stride(timestamp, ts_b, ts_s):
                        # 若True，则刷新mssn_history（记录最后的满足条件的任务记录）
                        mssn_history = DataFrame(
                            [
                                {
                                    "timestamp_begin": row["timestamp_begin"],
                                    "timestamp_step": row["timestamp_step"],
                                    "Time": timestamp.strftime("%H:%M:%S"),
                                    "MissionID": row2["MissionID"],
                                    "MissionClass": row2["MissionClass"],
                                    "MissionName": row2["MissionName"],
                                    "Attacktarget": row2["Attacktarget"],
                                    "PatrolZone": row2["PatrolZone"],
                                    "PrecautionArea": row2["PrecautionArea"],
                                }
                            ]
                        )
                # 将处于时间跨度内的任务记录作为新行插入mssnHis_inStep_df，若无符合条件的任务记录则插入仅含时间戳的空行
                mssnHis_inStep_df = pd.concat(
                    [mssnHis_inStep_df, mssn_history], ignore_index=True
                )
            # 复制最早一条区域更改记录至step=0，用于开始推演前初始化任务区域
            for idx3, row3 in mssnHis_inStep_df.iterrows():
                if row3["MissionID"] != "":
                    mssnHis_inStep_df.at[0, "Time"] = timestamp_begin.strftime(
                        "%H:%M:%S"
                    )
                    mssnHis_inStep_df.at[0, "MissionID"] = row3["MissionID"]
                    mssnHis_inStep_df.at[0, "MissionClass"] = row3["MissionClass"]
                    mssnHis_inStep_df.at[0, "MissionName"] = row3["MissionName"]
                    mssnHis_inStep_df.at[0, "Attacktarget"] = row3["Attacktarget"]
                    mssnHis_inStep_df.at[0, "PatrolZone"] = row3["PatrolZone"]
                    mssnHis_inStep_df.at[0, "PrecautionArea"] = row3["PrecautionArea"]
                    break

            # 调整DataFrame的index值，使其从1开始计数
            mssnHis_inStep_df.index = mssnHis_inStep_df.index + 1
            self.mssn_inStep_dfdict[k] = mssnHis_inStep_df
            if self.save_dir:
                mssnHis_inStep_df.to_excel(
                    self.save_dir + self.side + "_mssn_history_" + k + "_in_steps.xlsx"
                )

    def data_parser(self):
        """
        功能：传递数据
        返回：元组
        """
        return self.lua_df, self.mssn_inStep_dfdict, self.side_guid, self.mssnguid_list


class CommandSimplifier:
    def __init__(self, side, data, save_dir):
        """
        参数：side：处理数据所属推演方，'red'或'blue'
             data：传递过来的数据
             save_dir：处理过程记录文件，False为不保存
        """
        self.side = side
        self.save_dir = save_dir
        self.lua_df = data[0]
        self.mssn_dfdict = data[1]
        self.side_guid = data[2]
        self.mssnguid_list = data[3]
        self.simp_cmd_df = None

    def simplify_lua_commands(self):
        """
        功能：简化每step中的所有lua指令
        """
        # 初始化DataFrame
        simp_cmd_df = DataFrame(
            columns=["timestamp_begin", "timestamp_step", "commands"]
        )
        # 对每个任务初始化CommandBuffer
        cmd_buffer_dict = {}
        for guid in self.mssnguid_list:
            cmd_buffer_dict[guid] = CommandBuffer(guid)
        # 指令字符串拆解表达式
        pattern = re.compile(r"([^\|]+)\|")
        # 迭代DataFrame的每一行（迭代每step）
        for idx, row in self.lua_df.iterrows():
            # 获取时间戳
            simp_cmd_df.at[idx, "timestamp_begin"] = row["timestamp_begin"]
            simp_cmd_df.at[idx, "timestamp_step"] = row["timestamp_step"]
            # 初始化该位置数据类型为列表
            simp_cmd_df.at[idx, "commands"] = []
            # 判断该step是否有lua指令记录
            if len(row["lua_history"]) != 0:
                # 将lua指令字符串拆解并生成lua指令列表
                lua_list = pattern.findall(row["lua_history"])
                # 迭代由CommandBuffer组成的字典
                for k, v in cmd_buffer_dict.items():
                    cmd_list = []
                    for cmd in lua_list:
                        # 判断指令是否与任务guid对应
                        if re.search(k, cmd) is not None:
                            # 收集与任务guid对应的所有指令
                            cmd_list.append(cmd)
                    # 用收集的指令更新CommandBuffer
                    v.update_command(cmd_list)
            for k1, v1 in cmd_buffer_dict.items():
                # 将CommandBuffer中的指令记录输出到DataFrame
                if v1.command_output()["cmd_type"] != "Not_Operate":
                    simp_cmd_df.at[idx, "commands"].append(v1.command_output())
                # 重置CommandBuffer，用于下一step
                v1.reset_cmd_buffer()
        self.simp_cmd_df = simp_cmd_df
        if self.save_dir:
            self.simp_cmd_df.to_excel(
                self.save_dir + self.side + "_simp_lua_his_inTS.xlsx"
            )

    def merge_cmds_with_mssn_history(self):
        """
        功能：将任务记录转化为区域操作指令并入简化后的lua指令记录
        """
        # 迭代每个任务记录
        for guid, df in self.mssn_dfdict.items():
            temp_cb = CommandBuffer(guid)
            # 迭代任务记录DataFrame中每一行（每step）
            for idx, row in df.iterrows():
                # 条件：任务类型为巡逻任务
                if row["MissionClass"] == 2:
                    patrol_zone = []
                    precaution_area = []
                    target_list = []
                    # 条件：巡逻区/警戒区参考点记录不为空 或 不为nan（float）
                    if row["PatrolZone"] != "" and type(row["PatrolZone"]) != float:
                        for rp_pz in re.findall(r"[^\$]+", row["PatrolZone"]):
                            patrol_zone.append(rp_pz)
                    if (
                        row["PrecautionArea"] != ""
                        and type(row["PrecautionArea"]) != float
                    ):
                        for rp_pa in re.findall(r"[^\$]+", row["PrecautionArea"]):
                            precaution_area.append(rp_pa)
                    # 条件：指令记录DataFrame中对应行的指令记录是否为空
                    if len(self.simp_cmd_df.at[idx, "commands"]) == 0:
                        temp_cb.set_patrol_zone(patrol_zone)
                        temp_cb.set_precaution_area(precaution_area)
                        # 写入任务区域变动指令
                        self.simp_cmd_df.at[idx, "commands"].append(
                            temp_cb.command_output()
                        )
                        temp_cb.reset_cmd_buffer()
                    else:
                        for cmd in self.simp_cmd_df.at[idx, "commands"]:
                            if cmd["guid"] == guid:
                                # 修改对应任务指令
                                cmd["patrol_control"] = "Update"
                                cmd["patrol"] = patrol_zone
                                cmd["precaution_control"] = "Update"
                                cmd["precaution"] = precaution_area
                # 条件：任务类型为打击任务
                if row["MissionClass"] == 1:
                    target_list = []
                    # 条件：任务目标记录不为空
                    if row["Attacktarget"] != "" and type(row["Attacktarget"]) != float:
                        for tgt in re.findall(r"[^\,]+", row["Attacktarget"]):
                            target_list.append(tgt)
                    if len(self.simp_cmd_df.at[idx, "commands"]) == 0:
                        temp_cb.update_target_list(target_list)
                        self.simp_cmd_df.at[idx, "commands"].append(
                            temp_cb.command_output()
                        )
                        temp_cb.reset_cmd_buffer()
                    else:
                        for cmd in self.simp_cmd_df.at[idx, "commands"]:
                            if cmd["guid"] == guid:
                                # 修改对应任务指令
                                cmd["target_control"] = "Update"
                                cmd["targets"] = target_list

    def disperse_cmds(self):
        """
        功能：将堆积在某step的指令分散到后续无指令step中，若最后step出现堆积则抛弃指令
        """
        max_idx = self.simp_cmd_df.index.tolist()[-1]
        for idx, row in self.simp_cmd_df.iterrows():
            while len(row["commands"]) > 1:
                if idx != max_idx:
                    self.simp_cmd_df.at[idx + 1, "commands"].append(
                        row["commands"].pop(1)
                    )
                else:
                    row["commands"].pop()
        for idx2, row2 in self.simp_cmd_df.iterrows():
            if len(row2["commands"]) != 0:
                cmd_temp = row2["commands"][0]
            if len(row2["commands"]) == 0:
                row2["commands"].append(cmd_temp)
        if self.save_dir:
            self.simp_cmd_df.to_excel(self.save_dir + self.side + "_" + "cmds.xlsx")

    def data_parser(self):
        """
        功能：传递数据
        返回：元组
        """
        return self.simp_cmd_df, self.mssnguid_list


class CommandEncoder:
    def __init__(self, side, data, save_dir):
        """
        参数：side：处理数据所属推演方，'red'或'blue'
             data：传递过来的数据
             save_dir：处理过程记录文件，False为不保存
        """
        self.side = side
        self.cmd_df = data[0]
        self.mssnguid_list = data[1]
        self.save_dir = save_dir
        # 编码字典
        self.actiontype_dict = {"Not_Operate": 0, "Create": 1, "Modify": 2}
        self.mssntype_dict = {
            "Not_Operate": 0,
            "Combat_Air_Patrol": 1,
            "Anti_Ship_Strike": 2,
        }
        self.unitctrl_dict = {"Not_Operate": 0, "Update": 1}
        self.targetctrl_dict = {"Not_Operate": 0, "Update": 1}
        self.patrolctrl_dict = {"Not_Operate": 0, "Update": 1}
        self.precautionctrl_dict = {"Not_Operate": 0, "Update": 1}
        self.radarctrl_dict = {"Not_Operate": 0, "Active": 1, "Passive": 2}
        self.oecmctrl_dict = {"Not_Operate": 0, "Active": 1, "Passive": 2}
        self.wractrl_dict = {"Not_Operate": 0, "27": 1}

    def encode_cmds(self):
        """
        功能：将指令记录编码成算法动作空间格式
        """
        action_df = DataFrame(columns=["step", "action"])
        mssnguid_dict = {}
        i = 1
        for guid in self.mssnguid_list:
            mssnguid_dict[guid] = i
            i += 1
        # 调用load_data模块中类ConstructionMatrix的方法reference_point()，用于离散化处理指令记录中的参考点坐标
        init_matrix = ConstructionMatrix()
        discretized_rp = init_matrix.reference_point
        # 迭代指令记录DataFrame中的每行
        for idx, row in self.cmd_df.iterrows():
            # 条件：该行无指令记录
            if len(row["commands"]) == 0:
                action = [0, 0, 0, 0, [], 0, [], 0, [], 0, [], 0, 0, 0, 0]
            else:
                cmd = row["commands"][0]
                action = [0, 0, 0, 0, [], 0, [], 0, [], 0, [], 0, 0, 0, 0]
                # 对指令进行编码
                action[0] = self.actiontype_dict[cmd["cmd_type"]]
                action[1] = self.mssntype_dict[cmd["type"]]
                action[2] = mssnguid_dict[cmd["guid"]]
                action[3] = self.unitctrl_dict[cmd["unit_control"]]
                if action[3] != 0:
                    for unit in cmd["units"]:
                        action[4].append(units_encoder_dict[unit])
                # action[4] = cmd['units']
                action[5] = self.targetctrl_dict[cmd["target_control"]]
                if action[5] != 0:
                    for unit in cmd["targets"]:
                        action[6].append(units_encoder_dict[unit])
                # action[6] = cmd['targets']
                action[7] = self.patrolctrl_dict[cmd["patrol_control"]]
                if action[7] != 0:
                    action[8] = []
                    # 离散化处理参考点
                    for rp in cmd["patrol"]:
                        rp_tuple = re.findall(r"([\d\.]+)\,([\d\.]+)", rp)[0]
                        loc = discretized_rp(float(rp_tuple[1]), float(rp_tuple[0]))
                        action[8].append(loc)
                action[9] = self.precautionctrl_dict[cmd["precaution_control"]]
                if action[9] != 0:
                    action[10] = []
                    # 离散化处理参考点
                    for rp in cmd["precaution"]:
                        rp_tuple = re.findall(r"([\d\.]+)\,([\d\.]+)", rp)[0]
                        loc = discretized_rp(float(rp_tuple[1]), float(rp_tuple[0]))
                        action[10].append(loc)
                if cmd["starttime"] == "Not_Operate":
                    action[11] = 0
                else:
                    # 将任务开始时间离散至timestamp_begin
                    td = (
                        datetime.datetime.strptime(
                            re.search(r"\d{2}\:\d{2}\:\d{2}", cmd["starttime"]).group(),
                            "%H:%M:%S",
                        )
                        - datetime.datetime.strptime(
                            self.cmd_df.at[1, "timestamp_begin"], "%H:%M:%S"
                        )
                    ).seconds
                    action[11] = td // 15 + 1
                action[12] = self.radarctrl_dict[cmd["emcon_radar"]]
                action[13] = self.oecmctrl_dict[cmd["emcon_oecm"]]
                action[14] = self.wractrl_dict[cmd["wra"]]
            # 将转换完成的动作输出到DataFrame
            action_df.at[idx, "step"] = idx
            action_df.at[idx, "action"] = action
        if self.save_dir:
            action_df.to_excel(self.save_dir + self.side + "_" + "actions.xlsx")


def data_process(side, argsparser):
    # 实例化数据处理对象
    if side == "red":
        data_processor = DataProcessor(
            side, argsparser.file_LuaHistory, argsparser.file_red_UnitMission, False
        )
    elif side == "blue":
        data_processor = DataProcessor(
            side, argsparser.file_LuaHistory, argsparser.file_blue_UnitMission, False
        )

    # 修补无时间戳的lua指令
    data_processor.fix_commands_without_timestamp()

    # 根据白名单筛选lua指令
    data_processor.filter_commands_by_whitelist(whitelist)

    # 根据指令所属推演方筛选
    data_processor.sort_commands_by_side()

    # 根据指令时间戳所属step筛选
    data_processor.sort_commands_by_timesteps()

    # 筛选处理任务记录
    data_processor.filter_mssn_history()

    # 根据任务记录时间戳所属step筛选
    data_processor.sort_mssn_history_by_timesteps()

    # 返回数据
    return data_processor.data_parser()


def cmd_simplify(side, data, argsparser):
    # 创建数据简化对象，传入参数
    cmd_simplifier = CommandSimplifier(side, data, False)

    # 简化lua指令
    cmd_simplifier.simplify_lua_commands()

    # 合并任务记录到指令记录中
    cmd_simplifier.merge_cmds_with_mssn_history()

    # 分散堆积在某step的指令
    cmd_simplifier.disperse_cmds()

    return cmd_simplifier.data_parser()


def cmd_encode(side, data, argsparser):
    # 创建指令编码对象，传入参数
    encoder = CommandEncoder(side, data, argsparser.save_dir)

    # 将指令编码为动作
    encoder.encode_cmds()


def process(side, argsparser):
    # 定义子进程处理函数执行顺序与数据传递
    dp_data = data_process(side, argsparser)
    cs_data = cmd_simplify(side, dp_data, argsparser)
    cmd_encode(side, cs_data, argsparser)


if __name__ == "__main__":
    process_starttime = time.clock()
    args = parser.parse_args()

    sub_proc1 = Process(target=process, args=("red", args))
    # sub_proc2 = Process(target=process, args=('blue', args))
    sub_proc1.start()
    # sub_proc2.start()
    sub_proc1.join()
    # sub_proc2.join()

    print("主进程耗时：", time.clock() - process_starttime, "秒")
    os.system("pause")
