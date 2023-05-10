#!/usr/bin/env python
# coding: utf-8
"""
获取海峡风暴单元信息
2022/07/19
zhait
华戍防务
"""
import tensorflow as tf
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
import multiprocessing

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
    "--save_file_path", type=str, default="C:\\Users\\3-5\\Desktop\\aircraft_missile\\"
)
parser.add_argument("--total_unit_file_name", type=str, default="total_data.xlsx")
parser.add_argument(
    "--total_enemy_file_name", type=str, default="total_enemy_data.xlsx"
)


class Data_proprecessing:
    def __init__(self, side):
        if side == "red":
            self.side = side
            self.process_start = time.clock()
            self.args = parser.parse_args()
            # self.enemy_time_list = []
            self.time_list = []
            # self.total_data_enemy = []
            self.total_data = []
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
        else:
            self.side = side
            self.process_start = time.clock()
            self.args = parser.parse_args()
            self.enemy_time_list = []
            # self.time_list = []
            self.total_data_enemy = []
            # self.total_data = []
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
        if self.side == "red":
            self.total_file = os.listdir(self.args.path)
            for file_name in self.total_file:
                if file_name == "海峡风暴_202207211042":
                    self.file_name = file_name
                    single_file_list = os.listdir(self.args.path + file_name + "\\1")
                    # self.blue_side_list = [v for v in single_file_list if '蓝方_UnitPositions' in v]
                    self.red_side_list = [
                        v for v in single_file_list if "红方_UnitPositions" in v
                    ]
                    self.get_unit_information(self.red_side_list)
                    # self.get_contact_unit(single_file_list)
                    # 获取探测器探测目标CSV
                    # self.get_contacts(single_file_list)
                    self.export_excel()
                    # self.export_enemy_excel()
        else:
            self.total_file = os.listdir(self.args.path)
            for file_name in self.total_file:
                if file_name == "海峡风暴_202207211042":
                    self.file_name = file_name
                    single_file_list = os.listdir(self.args.path + file_name + "\\1")
                    self.blue_side_list = [
                        v for v in single_file_list if "蓝方_UnitPositions" in v
                    ]
                    # self.red_side_list = [v for v in single_file_list if '红方_UnitPositions' in v]
                    # self.get_unit_information(self.red_side_list)
                    self.get_contact_unit(single_file_list)
                    # 获取探测器探测目标CSV
                    # self.get_contacts(single_file_list)
                    # self.export_excel()
                    self.export_enemy_excel()

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
            file = self.red_csv(file_name)
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
        self.get_enemy_unit(contact_position)

    def get_enemy_unit(self, file_name_list):
        """
        功能：对Analysis目录下文件进行读取
        参数：{list: 文件名称}
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        for file_name in file_name_list:
            print(file_name)
            file = self.red_csv(file_name)
            columns_map = {}
            for index, row in file.iterrows():
                dic = {}
                if row["Time"] in self.enemy_time_list:
                    dic[row["ContactName"]] = self.laod_contact_unit(row)
                    columns_map.update(dic)
                else:
                    self.total_data_enemy = self.total_data_enemy + [columns_map]
                    columns_map = {}
                    self.enemy_time_list.append(row["Time"])
                    columns_map["Time"] = row["Time"]
                    columns_map[row["ContactName"]] = self.laod_contact_unit(row)
            # 添加表中最后一个Time
            self.total_data_enemy = self.total_data_enemy + [columns_map]

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
        for file_name in file_name_list:
            print(file_name)
            file = self.red_csv(file_name)
            columns_map = {}
            for index, row in file.iterrows():
                dic = {}
                if row["Time"] in self.time_list:
                    dic[row["UnitName"]] = self.load_unit_data(row)
                    columns_map.update(dic)
                else:
                    self.total_data = self.total_data + [columns_map]
                    columns_map = {}
                    self.time_list.append(row["Time"])
                    columns_map["Time"] = row["Time"]
                    columns_map[row["UnitName"]] = self.load_unit_data(row)
            # 添加表中最后一个Time
            self.total_data = self.total_data + [columns_map]

    def red_csv(self, specific_file_name):
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
        if os.path.exists(self.args.save_file_path + file_name):
            os.remove(self.args.save_file_path + file_name)
            print(f"文件{file_name}存在，删除原文件创建同名新文件")
            pf = pd.DataFrame()
            pf.to_excel(self.args.save_file_path + file_name, sheet_name="Sheet1")
        else:
            print(f"文件{file_name}不存在，创建新文件")
            pf = pd.DataFrame()
            pf.to_excel(self.args.save_file_path + file_name, sheet_name="Sheet1")

    def export_excel(self):
        """
        功能：将字典列表导出为Excel文件
        参数：无
        作者：zhait
        单位：北京华戍防务技术有限公司
        时间：7/19/22
        """
        self.create_file("total_data.xlsx")
        # self.create_file('total_enemy_data.xlsx')
        pf = pd.DataFrame(self.total_data)
        # 指定生成的Excel表格名称(xlsx文件要事先创建)
        file_writer = pd.ExcelWriter(
            self.args.save_file_path + self.args.total_unit_file_name
        )
        # 输出
        pf.to_excel(file_writer, sheet_name="Sheet1")
        file_writer.save()
        print(f"{ctime()}程序1完成")

    def export_enemy_excel(self):
        self.create_file("total_enemy_data.xlsx")
        pf = pd.DataFrame(self.total_data_enemy)
        # 指定生成的Excel表格名称(xlsx文件要事先创建)
        file_writer = pd.ExcelWriter(
            self.args.save_file_path + self.args.total_enemy_file_name
        )
        # 输出
        pf.to_excel(file_writer, sheet_name="Sheet1")
        file_writer.save()
        print(f"{ctime()}程序2完成")
        # print(f'主进程耗时:{time.clock() - self.process_start}')


# def multithreading():
#     """
#     功能：多线程
#     参数：无
#     作者：zhait
#     单位：北京华戍防务技术有限公司
#     时间：7/19/22
#     """
# 创建 Thread 实例
# Data = Data_proprecessing()
# process_start = time.clock()
# t1 = Thread(target=Data.export_excel(), args=())
# t2 = Thread(target=Data.export_enemy_excel(), args=())

# # 启动线程运行
# t1.start()
# t2.start()
#
# # 等待所有线程执行完毕
# t1.join()  # join() 等待线程终止，要不然一直挂起
# t2.join()
# print(f'主进程耗时:{time.clock()-process_start}')


def process(side):
    Data_proprecessing(side)


def multiprocess():
    process_start = time.clock()
    p1 = multiprocessing.Process(target=process, args=("red",))
    p2 = multiprocessing.Process(target=process, args=("blue",))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(f"主进程耗时:{time.clock()-process_start}")


if __name__ == "__main__":
    multiprocess()
