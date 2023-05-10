#!/usr/bin/env python
# coding: utf-8
"""
2022/03/30
华戍防务

"""
from openpyxl import load_workbook, Workbook
import numpy as np
from sklearn import preprocessing
import pandas as pd
import os, time
import xlsxwriter
import re
import os

# 读取CSV转化为EXCEl


def data_proprecessing():
    count_csv = 0
    # work_load = pd.ExcelFile('C:/Users/3-5/Desktop/aircraft_missile','missile_new')
    missile_work_load = pd.ExcelWriter(
        "C:/Users/3-5/Desktop/aircraft_missile/missile_zuixin.xlsx"
    )
    air_work_load = pd.ExcelWriter(
        "C:/Users/3-5/Desktop/aircraft_missile/aircraft_zuixin.xlsx"
    )
    # 读取CSV文件
    list_file_names = os.listdir("D:/ZT-mozi/Mozi/MoziServer/bin/Analysis")
    # list_file_names = os.listdir('D:/ZT-mozi/Mozi/MoziServer/bin/AAAA')
    for one_file_name in list_file_names:
        judge = os.listdir("D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/" + one_file_name)
        # judge = os.listdir('D:/ZT-mozi/Mozi/MoziServer/bin/AAAA/' + one_file_name)
        if len(judge) == 1:
            list_file_names = list_file_names
            for file_name in list_file_names:
                count_csv += 1

                if count_csv <= 1000:
                    (
                        missile_mozi_data_return_csv_screen,
                        sign_start_time,
                        sign_end_time,
                    ) = missile_load_proprecessing(file_name)
                    air_mozi_data_return_csv_screen = air_load_proprecessing(
                        file_name, sign_start_time, sign_end_time
                    )
                    count_csv_str = count_csv
                    count_csv_str = str(count_csv_str)
                    missile_mozi_data_return_csv_screen.to_excel(
                        missile_work_load, sheet_name="Sheet" + count_csv_str
                    )
                    air_mozi_data_return_csv_screen.to_excel(
                        air_work_load, sheet_name="Sheet" + count_csv_str
                    )

                else:
                    break
        else:
            # list_file_names = []
            # for index in judge:
            #     total = one_file_name+"/"+index
            #     list_file_names.append(total)
            for index in judge:
                if index.isdigit():
                    count_csv += 1
                    num_position = int(index)
                    index_down = judge.index(index)
                    file_name = one_file_name + "/" + index
                    documents = os.listdir(
                        "D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/" + file_name
                    )
                    # documents = os.listdir('D:/ZT-mozi/Mozi/MoziServer/bin/AAAA/' + file_name)
                    counts_num = 0
                    for sublist_name in documents:
                        if "中国_UnitPositions" in sublist_name:
                            counts_num += 1
                    if count_csv <= 5000:
                        (
                            missile_mozi_data_return_csv_screen,
                            sign_start_time,
                            sign_end_time,
                        ) = missile_load_proprecessing_one(file_name)
                        air_mozi_data_return_csv_screen = air_load_proprecessing_one(
                            counts_num,
                            index_down,
                            num_position,
                            file_name,
                            sign_start_time,
                            sign_end_time,
                        )

                        count_csv_str = count_csv
                        count_csv_str = str(count_csv_str)
                        missile_mozi_data_return_csv_screen.to_excel(
                            missile_work_load, sheet_name="Sheet" + count_csv_str
                        )
                        air_mozi_data_return_csv_screen.to_excel(
                            air_work_load, sheet_name="Sheet" + count_csv_str
                        )

                    else:
                        break
                else:
                    pass
    missile_work_load.save()
    missile_work_load.close()
    air_work_load.save()
    air_work_load.close()


# 获取飞机数据，蒙特卡洛仿真返回数据
def air_load_proprecessing_one(
    counts_num, index_down, num_position, file_name, sign_start_time, sign_end_time
):
    pattern = re.compile(r"0.0:(.*):")
    num = re.findall(pattern, sign_start_time)[0]
    num = str(int(num) + counts_num * (num_position - 1))
    num_2 = str(int(num) + 1)

    print(str(index_down) + file_name + "/中国_UnitPositions_" + num + ".csv")
    if num == num_2:

        air_mozi_data_return_csv = pd.read_csv(
            "D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"
            + file_name
            + "/中国_UnitPositions_"
            + num
            + ".csv",
            encoding="utf-8",
            index_col="TimelineID",
        )
        # air_mozi_data_return_csv = pd.read_csv(
        #     "D:/ZT-mozi/Mozi/MoziServer/bin/AAAA/" + file_name + "/中国_UnitPositions_" + num + ".csv",
        #     encoding='utf-8', index_col='TimelineID')
    else:
        air_mozi_data_return_csv = pd.read_csv(
            "D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"
            + file_name
            + "/中国_UnitPositions_"
            + num
            + ".csv",
            encoding="utf-8",
            index_col="TimelineID",
        )
        air_mozi_data_return_csv_2 = pd.read_csv(
            "D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"
            + file_name
            + "/中国_UnitPositions_"
            + num_2
            + ".csv",
            encoding="utf-8",
            index_col="TimelineID",
        )
        # air_mozi_data_return_csv = pd.read_csv(
        #     "D:/ZT-mozi/Mozi/MoziServer/bin/AAAA/" + file_name + "/中国_UnitPositions_" + num + ".csv",
        #     encoding='utf-8', index_col='TimelineID')
        # air_mozi_data_return_csv_2 = pd.read_csv(
        #     "D:/ZT-mozi/Mozi/MoziServer/bin/AAAA/" + file_name + "/中国_UnitPositions_" + num_2 + ".csv",
        #     encoding='utf-8', index_col='TimelineID')

    list_2 = ["飞机1"]
    air_mozi_data_return_csv = air_mozi_data_return_csv[
        air_mozi_data_return_csv["UnitName"].isin(list_2)
    ]
    air_mozi_data_return_csv_2 = air_mozi_data_return_csv_2[
        air_mozi_data_return_csv_2["UnitName"].isin(list_2)
    ]
    # air_mozi_data_return_csv = air_mozi_data_return_csv.append(air_mozi_data_return_csv_2,ignore_index=True)
    air_mozi_data_return_csv = pd.concat(
        [air_mozi_data_return_csv, air_mozi_data_return_csv_2], ignore_index=False
    )

    air = air_mozi_data_return_csv["Time"]
    air = list(air)
    air_list = []
    start = len(air)
    end = len(air)
    for i in range(len(air)):
        if air[i] == sign_start_time:
            start = i
        elif air[i] == sign_end_time:
            end = i
        if i >= start and i <= end:
            air_list.append(air[i])
        else:
            pass
    air_mozi_data_return_screen = air_mozi_data_return_csv[
        air_mozi_data_return_csv["Time"].isin(air_list)
    ]
    return air_mozi_data_return_screen


# 获取导弹数据，蒙特卡洛仿真返回数据
def missile_load_proprecessing_one(file_name):

    # mozi_data_return = load_workbook("D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"+file_name+"??_SensorDetectionAttempt.csv")
    missile_mozi_data_return_csv = pd.read_csv(
        "D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"
        + file_name
        + "/中国_SensorDetectionAttempt.csv",
        index_col="TargetRangeSlant_km",
    )
    # missile_mozi_data_return_csv = pd.read_csv(
    #     "D:/ZT-mozi/Mozi/MoziServer/bin/AAAA/" + file_name + "/中国_SensorDetectionAttempt.csv",
    #     index_col="TargetRangeSlant_km")
    columns = missile_mozi_data_return_csv.columns
    shape = missile_mozi_data_return_csv.shape

    list_1 = [
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #1",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #2",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #3",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #4",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #5",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #6",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #7",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #8",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #9",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #10",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #11",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #12",
    ]
    missile_mozi_data_return_csv = missile_mozi_data_return_csv[
        missile_mozi_data_return_csv["TargetName"].isin(list_1)
    ]

    missile_mozi_data_return_csv_screen = missile_mozi_data_return_csv.query(
        "TargetRangeHoriz_km<26 & TargetRangeHoriz_km>=0"
    ).copy()
    max_targetrange = missile_mozi_data_return_csv_screen["TargetRangeHoriz_km"].max()
    min_targetrange = missile_mozi_data_return_csv_screen["TargetRangeHoriz_km"].min()
    index_max = missile_mozi_data_return_csv_screen[
        (missile_mozi_data_return_csv_screen.TargetRangeHoriz_km == max_targetrange)
    ].index[0]
    index_min = missile_mozi_data_return_csv_screen[
        (missile_mozi_data_return_csv_screen.TargetRangeHoriz_km == min_targetrange)
    ].index[0]
    missile_mozi_data_return_csv_screen_1 = missile_mozi_data_return_csv_screen[
        index_max:index_min
    ].copy()
    if list(missile_mozi_data_return_csv_screen_1["Time"]) == []:
        index_list = list(
            missile_mozi_data_return_csv_screen[index_min:index_max].index
        )
        missile_mozi_data_return_csv_screen_1 = (
            missile_mozi_data_return_csv_screen.drop(index=index_list).copy()
        )
    missile_mozi_data_return_csv_screen_1.drop_duplicates("Time", inplace=True)
    sign_start_time = list(missile_mozi_data_return_csv_screen_1["Time"])[0]
    sign_end_time = list(missile_mozi_data_return_csv_screen_1["Time"])[-1]
    missile_mozi_data_return_csv_screen_2 = (
        missile_mozi_data_return_csv_screen_1.set_index("TimelineID").copy()
    )
    # missile_mozi_data_return_csv_screen_2.drop("TargetRangeSlant_km", axis=1, inplace=True)
    return missile_mozi_data_return_csv_screen_2, sign_start_time, sign_end_time


# 获取飞机数据，手动操作数据
def air_load_proprecessing(file_name, sign_start_time, sign_end_time):
    pattern = re.compile(r"0.0:(.*):")
    num = re.findall(pattern, sign_start_time)[0]

    air_mozi_data_return_csv = pd.read_csv(
        "D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"
        + file_name
        + "/1/??_UnitPositions_"
        + num
        + ".csv",
        index_col="TimelineID",
    )

    list_2 = ["飞机1"]
    air_mozi_data_return_csv = air_mozi_data_return_csv[
        air_mozi_data_return_csv["UnitName"].isin(list_2)
    ]

    air = air_mozi_data_return_csv["Time"]
    air = list(air)
    air_list = []
    start = len(air)
    end = len(air)
    for i in range(len(air)):
        if air[i] == sign_start_time:
            start = i
        elif air[i] == sign_end_time:
            end = i
        if i >= start and i <= end:
            air_list.append(air[i])
        else:
            pass
    air_mozi_data_return_screen = air_mozi_data_return_csv[
        air_mozi_data_return_csv["Time"].isin(air_list)
    ]
    return air_mozi_data_return_screen


# 获取导弹数据，手动操作数据
def missile_load_proprecessing(file_name):
    # mozi_data_return = load_workbook("D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"+file_name+"??_SensorDetectionAttempt.csv")
    missile_mozi_data_return_csv = pd.read_csv(
        "D:/ZT-mozi/Mozi/MoziServer/bin/Analysis/"
        + file_name
        + "/1/??_SensorDetectionAttempt.csv",
        index_col="TimelineID",
    )

    columns = missile_mozi_data_return_csv.columns
    shape = missile_mozi_data_return_csv.shape

    list_1 = [
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #1",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #2",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #3",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #4",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #5",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #6",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #7",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #8",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #9",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #10",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #11",
        "AIM-152B型先进远程空空导弹[休斯/雷锡恩] #12",
    ]
    missile_mozi_data_return_csv = missile_mozi_data_return_csv[
        missile_mozi_data_return_csv["TargetName"].isin(list_1)
    ]

    missile_mozi_data_return_csv_screen = missile_mozi_data_return_csv.query(
        "TargetRangeHoriz_km<26 & TargetRangeHoriz_km>=0"
    ).copy()

    missile_mozi_data_return_csv_screen.drop_duplicates("Time", inplace=True)
    sign_start_time = missile_mozi_data_return_csv_screen["Time"][0]
    sign_end_time = missile_mozi_data_return_csv_screen["Time"][-1]
    return missile_mozi_data_return_csv_screen, sign_start_time, sign_end_time


if __name__ == "__main__":
    data_proprecessing()
