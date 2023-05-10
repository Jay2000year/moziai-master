from openpyxl import load_workbook
import numpy as np
from sklearn import preprocessing
import pandas as pd
import cv2
from math import radians, cos, sin, asin, sqrt, degrees, atan2, degrees
import matplotlib.pyplot as plt
import scipy.sparse as sps
import argparse
import datetime
from tqdm import tqdm

# proprecess_data = Data_proprecessing()
from threading import Thread
import time
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path", type=str, default="D:\\mozi_decision_transformer_dataset\\"
)
parser.add_argument("--total_unit_file_name", type=str, default="total_data.xlsx")
parser.add_argument(
    "--total_enemy_file_name", type=str, default="total_enemy_data.xlsx"
)
parser.add_argument(
    "--Lua_history_file_name", type=str, default="LuaHistory_2022-08-01_10_27_07.txt"
)
# parser.add_argument('--path', type=str, default=proprecess_data.args.save_file_path)
# # 将两个文件放入以上目录中
# parser.add_argument('--total_unit_file_name', type=str, default=proprecess_data.args.total_unit_file_name)
# parser.add_argument('--total_enemy_file_name', type=str, default=proprecess_data.args.total_enemy_file_name)
parser.add_argument("--step_size", type=int, default=15)
# 矩阵大小设置
# parser.add_argument('--matrix_size', type=int, default=128)
# 中点坐标
parser.add_argument(
    "--initial_position", type=list, default=[25.7956364490066, 156.435337062311]
)


class ConstructionMatrix:
    def __init__(self, matrix_size):
        self.matrix_size = matrix_size
        self.args = parser.parse_args()
        self.single_matrix_lst = []
        self.normalization_data_dic = {}
        self.max_min = {
            2: [[64.82, 0], [0, 0], [359, 0]],
            1: [[1713.1, 648.2], [16000, 0], [359, 0]],
            0: [[4907.8, 981.56], [30480, 0], [359, 0]],
        }
        self.contents = os.listdir(self.args.path)
        self.initialization()

    def initialization(self):
        for sub_file in self.contents:
            # 导入本方数据
            self.red_unit = pd.read_excel(
                self.args.path + sub_file + "\\" + self.args.total_unit_file_name
            )
            # 数据处理
            self.null_row = self.red_unit["Time"].isnull()
            # 导入目标数据
            self.enemy_unit = pd.read_excel(
                self.args.path + sub_file + "\\" + self.args.total_enemy_file_name
            )
            # 导入rtg数据
            # self.rtg = pd.read_excel(self.args.path + sub_file + '\\' + self.args.total_rtg_file_name)
            # 删除红方单元数据读取中的空行
            self.remove_null()
            # 生成矩阵范围
            self.generate_zone()
            a = self.matrix_high_sum[self.matrix_size - 1]
            b = self.matrix_lenth_sum[self.matrix_size - 1]
            self.merge_file()

    def merge_file(self):
        """
        功能：合并本方和目标单元表格
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        self.scenario_all_unit = self.red_unit.merge(self.enemy_unit, on="Time")
        # self.scenario_unit_rtg = self.scenario_all_unit.merge(self.rtg, on='Time')
        self.get_columns_data()

    def remove_null(self):
        """
        功能：删除空行
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        for k, v in enumerate(self.null_row):
            if k != 0 and v:
                self.red_unit = self.red_unit.drop(k, axis=0)

    # def step_size(self):
    #     """
    #     功能: 确定step大小
    #     返回：列表
    #     作者: zhait
    #     单位：北京华戍防务技术有限公司
    #     时间：7/22/22
    #     """
    #     step_lst = []
    #     row_length = self.scenario_all_unit.shape[0]
    #     for i in range(0, row_length, self.args.step_size):
    #         if i != 1:
    #             step_lst.append(i)
    #         else:
    #             pass
    #     step_lst.append(row_length-1)
    #     return step_lst

    def lookup_table(self, name_lst):
        """
        功能: 对应表
        参数：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        self.table = {v: k * 100 for k, v in enumerate(name_lst)}

    def get_columns_data(self):
        """
        功能：获取booksheet对象中的指定列数据，并且将其处理成矩阵数据
        参数：
            booksheet: booksheet对象
            columns： {int - 行数}
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        total_length = self.scenario_all_unit.shape[0]
        # step split
        # step_lst = self.step_size()
        columns = self.scenario_all_unit.columns.values.tolist()
        self.lookup_table(columns)
        print("loading obss data......")
        pbar = tqdm(enumerate(self.scenario_all_unit.iterrows()), total=total_length)
        for idx, (ind, row) in pbar:
            # if idx == 343:
            #     print('wa')
            population_input = {}
            for column in columns:
                if "Unnamed" in column:
                    pass
                elif column == "Time":
                    pass
                elif row[column] != str(np.nan) and isinstance(row[column], str):
                    population_input[column] = row[column]
            if population_input:
                # 归一化后的数据
                self.nor_data_dic = {
                    k: self.normalization_data(eval(v))
                    for k, v in population_input.items()
                    if isinstance(eval(v), dict)
                }
                three_dimensional_matrix = self.matrix_data(self.nor_data_dic)
                # 将返回的三维数据添加到列表中
                three_dimensional_matrix = three_dimensional_matrix.transpose(2, 0, 1)
                self.single_matrix_lst.append(three_dimensional_matrix)
        # return self.single_matrix_lst
        return three_dimensional_matrix

    def get_bounds_point(self, lat, lon):
        """
        功能：获取区域界限坐标
        参数：无
        返回：无
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        dic = self.get_point_with_point_bearing_distance(lat, lon, 0, 230)
        dic_1 = self.get_point_with_point_bearing_distance(lat, lon, 90, 230)
        high = dic["latitude"] - lat
        lenth = dic_1["longitude"] - lon
        return lenth, high

    def generate_zone(self):
        """
        功能：以战场中心位置确定范围
        参数：无
        返回：无
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        initial_latitude = self.args.initial_position[0]
        initial_longitude = self.args.initial_position[1]
        # 获取上下限经纬度
        lenth, high = self.get_bounds_point(initial_latitude, initial_longitude)
        northwest = [0] * 2
        northwest[0], northwest[1] = initial_latitude + high, initial_longitude - lenth
        northeast = [0] * 2
        northeast[0], northeast[1] = initial_latitude + high, initial_longitude + lenth
        southeast = [0] * 2
        southeast[0], southeast[1] = initial_latitude - high, initial_longitude + lenth
        southwest = [0] * 2
        southwest[0], southwest[1] = initial_latitude - high, initial_longitude - lenth
        # construction_matrix(100, 130)
        len_1, len_2 = (
            abs(northeast[1] - northwest[1]) / self.matrix_size,
            abs(northeast[0] - northwest[0]) / self.matrix_size,
        )
        self.matrix_lenth_sum = []
        Longitude_construction = northwest[1]
        for index_1 in range(self.matrix_size):
            Longitude_construction = Longitude_construction + len_1
            self.matrix_lenth_sum.append(Longitude_construction)
        len_3, len_4 = (
            abs(northwest[1] - southwest[1]) / self.matrix_size,
            abs(northwest[0] - southwest[0]) / self.matrix_size,
        )
        self.matrix_high_sum = []
        Latitude_construction = northwest[0]
        for index_2 in range(self.matrix_size):
            Latitude_construction = Latitude_construction - len_4
            self.matrix_high_sum.append(Latitude_construction)

    def normalization(self, data):
        """
        功能：归一化函数
        参数：
            data:{字典 - 归一化数据}
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        if isinstance(data, list):
            minmax_scale = preprocessing.MinMaxScaler()
            dataframe = pd.DataFrame(data)
            # 输入数据DataFrame类型数据   返回的数据是numpy.ndarray
            minmax_scale_data = minmax_scale.fit_transform(dataframe)
            # numpy.ndarray   切片取第0列
            nor = list(minmax_scale_data[:, 0])[0]
        else:
            nor = data
        return nor

    def normalization_data(self, dic):
        """
        功能：对所有数据进行归一化
        参数：无
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        max_min = self.max_min[dic["UnitType"]]
        data = {
            "latitude": dic["UnitLatitude"],
            "longitude": dic["UnitLongitude"],
            "speed": [dic["UnitSpeed_kms"]] + max_min[0],
            "high": [dic["UnitAltitude_m"]] + max_min[1],
            "course": [dic["UnitCourse"]] + max_min[2],
            "type": dic["UnitType"],
        }
        self.normalization_data_dic = {
            k: self.normalization(v) for k, v in data.items()
        }
        return self.normalization_data_dic

    def matrix_create(self):
        """
        功能：在矩阵中过去单元位置      构造三维矩阵并把归一化数据传入
        参数：
            字典{dict: 一帧归一化后的数据}
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        # 创建二维数据
        # 速度二维数据
        speed_pos_matrix = np.zeros(
            shape=(self.matrix_size, self.matrix_size), dtype=np.float64
        )
        # 高度二维数据
        high_pos_matrix = np.zeros(
            shape=(self.matrix_size, self.matrix_size), dtype=np.float64
        )
        # 航向二维数据
        course_pos_matrix = np.zeros(
            shape=(self.matrix_size, self.matrix_size), dtype=np.float64
        )
        # 单元类型二维数据
        unit_type_pos_matrix = np.zeros(
            shape=(self.matrix_size, self.matrix_size), dtype=np.float64
        )
        # 可视化二维数据
        visualization_matrix = np.zeros(
            shape=(self.matrix_size, self.matrix_size), dtype=np.float64
        )
        return (
            speed_pos_matrix,
            high_pos_matrix,
            course_pos_matrix,
            unit_type_pos_matrix,
            visualization_matrix,
        )

    def binary_search_normal(self, lat, lon, lst):
        """
        功能：二分法
        参数：
            data:dict-输入的单元信息
            lst: 列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        row, col = 0, 0
        lst.sort()
        start, end = 0, len(self.matrix_lenth_sum) - 1
        if lst == self.matrix_lenth_sum:
            data = lon
            num = col
        else:
            data = lat
            num = row
        while end >= start:
            mid_index = (end + start) // 2
            if data == lst[mid_index]:
                num = mid_index
                return col
            elif data > lst[mid_index]:
                start = mid_index + 1
            elif data < lst[mid_index]:
                end = mid_index - 1
        if lst == self.matrix_high_sum:
            num = len(self.matrix_high_sum) - start
        else:
            num = start
        return num, lst[num]

    def ergodic_method(self, data):
        """
        功能：遍历法
        参数：
            data:dict-输入的单元信息
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        row, col = 0, 0
        while data["longitude"] >= self.matrix_lenth_sum[col]:
            col += 1
        while data["latitude"] <= self.matrix_high_sum[row]:
            row += 1
        return row, col

    def matrix_data(self, nor_dic):
        """
        功能：在矩阵中过去单元位置      构造三维矩阵并把归一化数据传入
        参数：
            字典{dict: 一帧归一化后的数据}
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        (
            speed_pos_matrix,
            high_pos_matrix,
            course_pos_matrix,
            unit_type_pos_matrix,
            visualization_matrix,
        ) = self.matrix_create()
        for k, data in nor_dic.items():
            col, in_matrix_lon = self.binary_search_normal(
                data["latitude"], data["longitude"], self.matrix_lenth_sum
            )
            row, in_matrix_lat = self.binary_search_normal(
                data["latitude"], data["longitude"], self.matrix_high_sum
            )
            # 确定单元位置
            speed_pos_matrix[row, col] = data["speed"]
            high_pos_matrix[row, col] = data["high"]
            course_pos_matrix[row, col] = data["course"]
            unit_type_pos_matrix[row, col] = self.table[k]
        # self.visualization_cv2(unit_type_pos_matrix)
        # self.visualization(unit_type_pos_matrix)
        # 合并二维数据成三维数据
        unit_matrix = np.array(
            [speed_pos_matrix, high_pos_matrix, course_pos_matrix, unit_type_pos_matrix]
        )
        unit_matrix = unit_matrix.transpose(1, 2, 0)
        return unit_matrix

    def reference_point(self, lat, lon):
        """
        功能：输入参考点坐标生成矩阵离散化坐标
        参数：
            str{str-经纬度}
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/22/22
        """
        col, in_matrix_lon = self.binary_search_normal(lat, lon, self.matrix_lenth_sum)
        row, in_matrix_lat = self.binary_search_normal(lat, lon, self.matrix_high_sum)
        return col, row

    def visualization_cv2(self, matrix):
        """
        功能 ：cv2矩阵可视化

        """
        cv2.imwrite("2.jpg", matrix)
        s = cv2.imread("2.jpg")
        cv2.imshow("img2", s)
        cv2.waitKey(600)
        cv2.destroyWindow("img2")

    def visualization(self, matrix):
        """
        功能：对矩阵数据进行可视化

        """
        # 矩阵切分
        matrix = np.split(matrix, 2, axis=1)[0]
        matrix_pos = sps.csr_matrix(matrix)
        print(matrix_pos)
        # 密集矩阵
        # plt.matshow(matrix)
        # plt.show()
        # scipy可视化
        # sps.csr_matrix(matrix)
        # print()

    @staticmethod
    def data_cutting(air_missile_3D):
        """
        功能：对矩阵数据进行裁剪

        """
        img = air_missile_3D
        _, h, w, _ = img.shape
        new_h1, new_h2 = np.random.randint(0, h - 48, 2)
        new_w1, new_w2 = np.random.randint(0, w - 48, 2)
        img_crop1 = img[:, new_h1 : new_h1 + 512, new_w1 : new_w1 + 512, :]
        img_crop2 = img[:, new_h2 : new_h2 + 512, new_w2 : new_w2 + 512, :]

    @staticmethod
    def data_rotate(air_missile_3D):
        """
        功能：对矩阵数据进行翻转

        """
        img = air_missile_3D
        # 水平镜像
        level_img = cv2.flip(img, 1)
        # 垂直镜像
        v_flip = cv2.flip(img, 0)
        # 水平垂直镜像
        hv_flip = cv2.flip(img, -1)
        # 90度旋转
        rows, cols, _ = img.shape
        m = cv2.getRotationmatrix2D((cols / 2, rows / 2), 45, 1)
        rotation_45 = cv2.warpAffine(img, m, (cols, rows))
        # 45度旋转
        m = cv2.getRotationmatrix2D((cols / 2, rows / 2), 135, 2)
        rotation_135 = cv2.warpAffine(img, m, (cols, rows))
        return level_img, v_flip, hv_flip, rotation_45, rotation_135

    @staticmethod
    def get_point_with_point_bearing_distance(lat, lon, bearing, distance):
        """
        功能：已知一点求沿某一方向一段距离的点
        :param lat:纬度
        :param lon:经度
        :param bearing:朝向角，正北为0， 顺时针依次增大
        :param distance:距离, 海里
        :return:
        """
        radius_earth_kilometres = 3440
        initial_bearing_radians = radians(bearing)
        dis_ratio = distance / radius_earth_kilometres
        dist_ratio_sine = sin(dis_ratio)
        dist_ratio_cosine = cos(dis_ratio)
        start_lat_rad = radians(lat)
        start_lon_rad = radians(lon)
        start_lat_cos = cos(start_lat_rad)
        start_lat_sin = sin(start_lat_rad)
        end_lat_rads = asin(
            (start_lat_sin * dist_ratio_cosine)
            + (start_lat_cos * dist_ratio_sine * cos(initial_bearing_radians))
        )
        end_lon_rads = start_lon_rad + atan2(
            sin(initial_bearing_radians) * dist_ratio_sine * start_lat_cos,
            dist_ratio_cosine - start_lat_sin * sin(end_lat_rads),
        )
        my_lat = degrees(end_lat_rads)
        my_lon = degrees(end_lon_rads)
        dic = {"latitude": my_lat, "longitude": my_lon}
        return dic


def multithreading():
    """
    功能：多线程
    参数：无
    作者：zhait
    单位：北京华戍防务技术有限公司
    时间：7/19/22
    """
    # 创建 Thread 实例
    Data = ConstructionMatrix()
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
    ConstructionMatrix()
