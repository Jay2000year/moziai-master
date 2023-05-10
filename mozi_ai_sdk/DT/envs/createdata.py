#!/usr/bin/env python
# coding: utf-8
"""
获取海峡风暴单元信息
2022/07/19
zhait
华戍防务
"""
from openpyxl import load_workbook, Workbook
import numpy as np
from sklearn import preprocessing
import pandas as pd
import re
import os
import argparse
from threading import Thread
import time
from time import sleep, ctime
import datetime

# 读取CSV转化为EXCEl

"""
dic = ['飞机', '舰船', '导弹']
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    default="D:\\BaiduNetdiskDownload\\JSPT0718\\JSPT0718\\DeductionServer\\DeductionServer\\ServerInt\\bin64\\Analysis\\",
)
parser.add_argument(
    "--save_file_path",
    type=str,
    default="D:\\mozi_decision_transformer_dataset\\20220823_",
)
parser.add_argument("--total_unit_file_name", type=str, default="total_data.xlsx")
parser.add_argument(
    "--total_enemy_file_name", type=str, default="total_enemy_data.xlsx"
)
parser.add_argument("--total_rtg_file_name", type=str, default="total_rtg_data.xlsx")
# parser.add_argument('--rtg_file_name', type=str, default='2022-8-8_12.19.23.txt')
parser.add_argument("--file_LuaHistory", type=str, default="LuaHistory")
parser.add_argument("--step_size", type=int, default=15)


class Data_proprecessing:
    def __init__(self):
        # self.process_start = time.clock()
        self.args = parser.parse_args()
        self.enemy_time_list = []
        self.timestamp_step_lst = []
        self.time_list = []
        self.UnitName = None
        self.unit_information = {
            "UnitLatitude": None,
            "UnitLongitude": None,
            "UnitCourse": None,
            "UnitSpeed_kms": None,
            "UnitAltitude_m": None,
            "UnitType": None,
        }
        self.unit_type = {"Ship": 2, "Aircraft": 0, "Weapon": 1}
        # 加载要保存的xlsx文件
        self.load_data()

    def load_data(self):
        """
        功能：加载csv数据
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        self.total_file = os.listdir(self.args.path)
        for idx, file_name in enumerate(self.total_file):
            self.file_name = file_name
            single_file_list = os.listdir(self.args.path + file_name + "\\1")
            self.blue_side_list = [
                v for v in single_file_list if "蓝方_UnitPositions" in v
            ]
            self.red_side_list = [
                v for v in single_file_list if "红方_UnitPositions" in v
            ]
            self.red_side_rtg = [v for v in single_file_list if "_SidesPoints.csv" in v]
            self.lua_history = [
                v for v in single_file_list if self.args.file_LuaHistory in v
            ]
            rtg_data = self.rtg_data_get(self.red_side_rtg[0])
            total_unit = self.get_unit_information(self.red_side_list)
            total_enemy = self.get_contact_unit(single_file_list)
            # 获取探测器探测目标CSV
            # self.get_contacts(single_file_list)
            self.file_path = self.args.save_file_path + str(idx + 1) + "\\"
            self.rtg_export(rtg_data)
            self.export_excel(total_unit)
            self.export_enemy_excel(total_enemy)
            self.timestamp_step_lst = []

    def rtg_data_get(self, rtg_file):
        """
        功能：加载txt数据获取reward情况
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        dict_lst = []
        reward = {}
        time_lst = []
        rtg_file = self.read_csv(rtg_file)
        rtg_file_col = rtg_file.columns
        log_csv_file = {
            row["Time"]: row[rtg_file_col[2]]
            for index, row in rtg_file.iterrows()
            if "红方" in row["SideName"]
        }
        with open(self.path(self.lua_history[0]), encoding="utf-8") as lua_history:
            self.lua_content = lua_history.readlines()
            self.lua_list = re.findall(r".+", self.lua_content[0])
            self.sort_commands_by_timesteps()
        dic = {}
        # 2020-04-16 14:00:01    0.0:20:15
        for time, score in log_csv_file.items():
            a = re.findall(r"0.(.*):", time)[0]
            b = re.findall(r":(.*)", time)[0]
            second_num = (
                int(re.findall(r"(.*):", a)[0]) * 3600
                + int(re.findall(r":(.*):", time)[0]) * 60
                + int(re.findall(r":(.*)", b)[0])
            )
            second_num = second_num - 1
            offset = datetime.timedelta(seconds=second_num)
            # 获取修改后的时间并格式化
            re_time = (self.timestamp_begin + offset).strftime("%Y-%m-%d %H:%M:%S")
            if re_time in time_lst:
                dic["event"] = score
            else:
                rtg_position = self.is_commands_in_stride(
                    re_time, self.timestamp_begin, self.timestamp_step
                )
                dict_lst.append(dic)
                dic = {}
                dic["rtg_pos"] = rtg_position
                dic["Time"] = re_time
                time_lst.append(re_time)
                dic["event"] = score
        # with open('C:\\Users\\3-5\\Desktop\\aircraft_missile\\' + self.args.rtg_file_name, encoding='utf-8') as log_file:
        #     log_file = [i.strip() for i in log_file.readlines() if '蓝方' in i and '被摧毁' in i]
        #     dic = {}
        #     for text in log_file:
        #         if text:
        #             text = text.replace('/', '-')
        #             time = re.findall("(\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2})", text)[0]
        #             if time in time_lst:
        #                 dic['event'] = dic['event'] + re.findall(r' - (.*)', text)
        #             else:
        #                 rtg_position = self.is_commands_in_stride(time, self.timestamp_begin, self.timestamp_step)
        #                 dict_lst.append(dic)
        #                 dic = {}
        #                 dic['rtg_pos'] = rtg_position
        #                 dic['Time'] = time
        #                 time_lst.append(time)
        #                 dic['event'] = re.findall(r' - (.*)', text)
        dict_lst.remove({})
        return dict_lst

    def sort_commands_by_timesteps(self):
        """
        功能：将lua指令与steps进行对应
        """
        # 设置初始步数
        steps = 1
        # 设置每步幅时长跨度
        self.step_stride = 15
        # 获取起始/步进时间戳  2020-04-16 14:00:01
        self.timestamp_begin = datetime.datetime.strptime(
            re.match(
                r"\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}", self.lua_list[0]
            ).group(),
            "%Y-%m-%d %H:%M:%S",
        )
        self.timestamp_step = self.timestamp_begin + datetime.timedelta(
            seconds=self.step_stride
        )
        for step in range(400):
            time = step * 15
            timestamp_step = self.timestamp_begin + datetime.timedelta(seconds=time)
            self.timestamp_step_lst.append(str(timestamp_step))

        # 根据推演总时长计算结束时间戳
        timestamp_end = self.timestamp_begin + datetime.timedelta(minutes=100)
        # 计算steps总数
        self.total_steps = (
            timestamp_end - self.timestamp_begin
        ).seconds // self.step_stride

    # 判断指令时间戳是否在step时间跨度内（如10:00:01-10:00:16）
    def is_commands_in_stride(self, lua, ts_begin, ts_step):
        step = 0
        lua_timestamp = datetime.datetime.strptime(
            re.match(r"\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}", lua).group(),
            "%Y-%m-%d %H:%M:%S",
        )
        while step <= self.total_steps:
            if lua_timestamp.__ge__(ts_step):
                step += 1
                ts_begin = ts_step
                ts_step = ts_begin + datetime.timedelta(seconds=self.step_stride)
            else:
                b = step
                break
            # 大于等于
            ge = lua_timestamp.__ge__(ts_begin)
            # 小于
            lt = lua_timestamp.__lt__(ts_step)
        return step

    def get_contacts(self, file_list):
        """
        功能：筛选探测器Csv文件
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        total = [v for v in file_list if "SensorDetectionAttempt" in v]
        lst = []
        for file_name in total:
            file = self.read_csv(file_name)
            if (
                "闪电" in file["SensorParentName"].unique()[0]
                and file["SensorParentSide"].unique()[0] == "红方"
            ):
                lst.append(file_name)
            elif (
                "航空母舰" in file["SensorParentName"].unique()[0]
                and file["SensorParentSide"].unique()[0] == "红方"
            ):
                lst.append(file_name)
        self.get_enemy_unit(lst)

    def get_contact_unit(self, file_list):
        """
        功能：筛选探测器Csv文件
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        contact_position = [v for v in file_list if "红方_ContactPositions" in v]
        enemy_data = self.get_enemy_unit(contact_position)
        return enemy_data

    def get_enemy_unit(self, file_name_list):
        """
        功能：对Analysis目录下文件进行读取
        参数：{list: 文件名称}
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        total_data_enemy = []
        for file_name in file_name_list:
            file = self.read_csv(file_name)
            columns_map = {}
            for index, row in file.iterrows():
                time = row["Time"]
                if isinstance(row["Time"], str):
                    a = re.findall(r"0.(.*):", row["Time"])[0]
                    b = re.findall(r":(.*)", row["Time"])[0]
                    second_num = (
                        int(re.findall(r"(.*):", a)[0]) * 3600
                        + int(re.findall(r":(.*):", row["Time"])[0]) * 60
                        + int(re.findall(r":(.*)", b)[0])
                    )
                    second_num = second_num - 1
                    offset = datetime.timedelta(seconds=second_num)
                    # 获取修改后的时间并格式化
                    re_time = (self.timestamp_begin + offset).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if re_time in self.timestamp_step_lst:
                        dic = {}
                        if row["Time"] in self.enemy_time_list:
                            dic[row["ContactName"]] = self.laod_contact_unit(row)
                            columns_map.update(dic)
                        else:
                            total_data_enemy = total_data_enemy + [columns_map]
                            columns_map = {}
                            self.enemy_time_list.append(row["Time"])
                            columns_map["Time"] = row["Time"]
                            columns_map[row["ContactName"]] = self.laod_contact_unit(
                                row
                            )
            # 添加表中最后一个Time
            total_data_enemy = total_data_enemy + [columns_map]
            self.enemy_time_list = []
        return total_data_enemy

    def step_size(self, row_length=6000):
        """
        功能: 确定step大小
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        step_lst = []
        for i in range(0, row_length, self.args.step_size):
            if i != 1:
                step_lst.append(i)
            else:
                pass
        step_lst.append(row_length - 1)
        return step_lst

    def get_unit_information(self, file_name_list):
        """
        功能：对Analysis目录下文件进行读取
        参数：{list: 文件名称}
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        file_name_list_index = list(
            map(lambda x: int(re.findall(r"[0-9]+", x)[0]), file_name_list)
        )
        file_dic = {k: v for k, v in zip(file_name_list_index, file_name_list)}
        file_name_list = [v[1] for v in sorted(file_dic.items())]
        total_data = []
        for file_name in file_name_list:
            # print(file_name)
            file = self.read_csv(file_name)
            columns_map = {}
            for index, row in file.iterrows():
                if isinstance(row["Time"], str):
                    a = re.findall(r"0.(.*):", row["Time"])[0]
                    b = re.findall(r":(.*)", row["Time"])[0]
                    second_num = (
                        int(re.findall(r"(.*):", a)[0]) * 3600
                        + int(re.findall(r":(.*):", row["Time"])[0]) * 60
                        + int(re.findall(r":(.*)", b)[0])
                    )
                    second_num = second_num - 1
                    offset = datetime.timedelta(seconds=second_num)
                    # 获取修改后的时间并格式化
                    re_time = (self.timestamp_begin + offset).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if re_time in self.timestamp_step_lst:
                        dic = {}
                        if row["Time"] in self.time_list:
                            dic[row["UnitName"]] = self.load_unit_data(row)
                            columns_map.update(dic)
                        else:
                            total_data = total_data + [columns_map]
                            columns_map = {}
                            self.time_list.append(row["Time"])
                            columns_map["Time"] = row["Time"]
                            columns_map[row["UnitName"]] = self.load_unit_data(row)
                # 添加表中最后一个Time
            total_data = total_data + [columns_map]
            self.time_list = []
        return total_data

    def read_csv(self, specific_file_name):
        """
        功能：读取csv
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        file = pd.read_csv(
            self.args.path + self.file_name + "\\1\\" + specific_file_name,
            encoding="utf-8",
            index_col="TimelineID",
        )
        return file

    def path(self, file_path):
        path = self.args.path + self.file_name + "\\1\\" + file_path
        return path

    def laod_contact_unit(self, row):
        """
        功能：敌方单元信息
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        self.unit_information = {
            "UnitLatitude": row["ContactLatitude"],
            "UnitLongitude": row["ContactLongitude"],
            "UnitCourse": row["ContactCurrentHeading"],
            "UnitSpeed_kms": row["ContactCurrentSpeed"],
            "UnitAltitude_m": row["ContactCurrentAltitude_ASL"],
            "UnitType": row["ContactType"],
        }
        return self.unit_information

    def load_enemy_unit(self, row):
        """
        功能：敌方单元信息
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        self.unit_information = {
            "UnitLatitude": row["TargetLatitude"],
            "UnitLongitude": row["TargetLongitude"],
            "UnitCourse": row["TargetHeading"],
            "UnitSpeed_kms": row["TargetSpeed_kms"],
            "UnitAltitude_m": row["TargetAltitude_AGL_m"],
        }
        return self.unit_information

    def load_unit_data(self, row):
        """
        功能：单元信息
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        self.unit_information = {
            "UnitLatitude": row["UnitLatitude"],
            "UnitLongitude": row["UnitLongitude"],
            "UnitCourse": row["UnitCourse"],
            "UnitSpeed_kms": row["UnitSpeed_kms"],
            "UnitAltitude_m": row["UnitAltitude_m"],
            "UnitType": self.unit_type[row["UnitType"]],
        }
        return self.unit_information

    def create_file(self, file_name):
        os.makedirs(self.file_path, exist_ok=True)
        if os.path.exists(self.file_path + file_name):
            os.remove(self.file_path + file_name)
            print(f"文件{file_name}存在，删除原文件创建同名新文件")
            pf = pd.DataFrame()
            pf.to_excel(self.file_path + file_name, sheet_name="Sheet1")
        else:
            print(f"文件{file_name}不存在，创建新文件")
            pf = pd.DataFrame()
            pf.to_excel(self.file_path + file_name, sheet_name="Sheet1")

    def rtg_export(self, rtg_data):
        """
        功能：将字典列表导出为Excel文件
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        self.create_file(self.args.total_rtg_file_name)
        # self.create_file('total_enemy_data.xlsx')
        pf = pd.DataFrame(rtg_data)
        # 指定生成的Excel表格名称(xlsx文件要事先创建)
        file_writer = pd.ExcelWriter(self.file_path + self.args.total_rtg_file_name)
        # 输出
        pf.to_excel(file_writer, sheet_name="Sheet1")
        file_writer.save()
        print(f"{ctime()}程序1完成")

    def export_excel(self, total_unit):
        """
        功能：将字典列表导出为Excel文件
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        self.create_file(self.args.total_unit_file_name)
        # self.create_file('total_enemy_data.xlsx')
        pf = pd.DataFrame(total_unit)
        # 指定生成的Excel表格名称(xlsx文件要事先创建)
        file_writer = pd.ExcelWriter(self.file_path + self.args.total_unit_file_name)
        # 输出
        pf.to_excel(file_writer, sheet_name="Sheet1")
        file_writer.save()
        print(f"{ctime()}程序2完成")

    def export_enemy_excel(self, total_enemy):
        self.create_file(self.args.total_enemy_file_name)
        pf = pd.DataFrame(total_enemy)
        # 指定生成的Excel表格名称(xlsx文件要事先创建)
        file_writer = pd.ExcelWriter(self.file_path + self.args.total_enemy_file_name)
        # 输出
        pf.to_excel(file_writer, sheet_name="Sheet1")
        file_writer.save()
        print(f"{ctime()}程序3完成")
        # print(f'主进程耗时:{time.clock() - self.process_start}')


def multithreading():
    """
    功能：多线程
    参数：无
    作者：zhait
    单位：北京华戍防务技术有限公司
    时间：7/19/22
    """
    # 创建 Thread 实例
    Data = Data_proprecessing()
    process_start = time.clock()
    t0 = Thread(target=Data.rtg_export(), args=())
    t1 = Thread(target=Data.export_excel(), args=())
    t2 = Thread(target=Data.export_enemy_excel(), args=())
    # 启动线程运行
    t0.start()
    t1.start()
    t2.start()
    # 等待所有线程执行完毕
    t0.join()
    t1.join()  # join() 等待线程终止，要不然一直挂起
    t2.join()
    print(f"主进程耗时:{time.clock()-process_start}")


if __name__ == "__main__":
    Data_proprecessing()
    # multithreading()
