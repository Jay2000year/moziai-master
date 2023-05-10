# 时间 : 2022/4/12 17:17
# 作者 : zhait
# 文件 : model_train.py
# 项目 : moziai
# 版权 : 北京华戍防务技术有限公司
# 加载数据构造数据矩阵


from openpyxl import load_workbook
import numpy as np
from sklearn import preprocessing
import pandas as pd
import cv2
import tensorflow as tf

# from moziai.mozi_ai_sdk.imitation_learning.tf1.interactive_test import interactive_test


class ConstructionMatrix:
    def __init__(self):
        # 导入飞机数据
        self.air_workbook = load_workbook(
            "C:/users/3-5/Desktop/aircraft_missile/aircraft_1.xlsx"
        )  # 相对路径，找到需要打开的文件位置
        # 导入导弹数据
        self.missile_workbook = load_workbook(
            "C:/users/3-5/Desktop/aircraft_missile/missile_1.xlsx"
        )
        # 依次去表格
        # self.data_num = self.air_workbook._sheet
        self.air_data_columns = {
            "unitLongitude": 7,
            "unitLatitude": 8,
            "unitSpeedKms": 10,
            "unit_high": 11,
            "unit_course": 9,
        }
        self.missile_data_columns = {
            "missileLatitude": 20,
            "missileLongitude": 19,
            "missileSpeedKms": 23,
            "missile_high": 21,
            "missile_course": 24,
        }
        self.normalization_data_dic = {}
        self.air_missile_3D_array_list = []
        self.air_hsc_data_array_list = []
        self.batch_list = []
        self._init_load_data()

    def _init_load_data(self):
        """
        功能：加载CSV数据
        参数：无
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/12/22
        """
        for num in range(1, 10):
            num = str(num)
            self.air_booksheet = self.air_workbook.get_sheet_by_name("Sheet" + num)
            self.missile_booksheet = self.missile_workbook.get_sheet_by_name(
                "Sheet" + num
            )
            print("**************", num, "***************")
            self.air_data = {
                k: self.get_columns_data(self.air_booksheet, v)
                for k, v in self.air_data_columns.items()
            }
            self.missile_data = {
                k: self.get_columns_data(self.missile_booksheet, v)
                for k, v in self.missile_data_columns.items()
            }
            self.normalization_data()
            self.generate_zone()
            self.matrix_data()
            self.export()

    def get_columns_data(self, booksheet, columns):
        """
        功能：获取booksheet对象中的指定列数据
        参数：
            booksheet: booksheet对象
            columns： {int - 行数}
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/12/22
        """
        data_lst = []
        flag = True
        num = 2
        while flag:
            # 获取sheet页的行数据
            data = booksheet.cell(row=num, column=columns).value
            if data:
                num += 1
                data_lst.append(data)
            else:
                flag = False
        return data_lst

    def generate_zone(self):
        """
        功能：以飞机初始位置为中心确定范围
        参数：无
        返回：无
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/12/22
        """
        air_initial_latitude = self.air_data["unitLatitude"][0]
        air_initial_longitude = self.air_data["unitLongitude"][0]
        northwest = [0] * 2
        northwest[0], northwest[1] = (
            air_initial_latitude + 0.20144026,
            air_initial_longitude - 0.222952932,
        )
        northeast = [0] * 2
        northeast[0], northeast[1] = (
            air_initial_latitude + 0.202979658,
            air_initial_longitude + 0.219372318,
        )
        southeast = [0] * 2
        southeast[0], southeast[1] = (
            air_initial_latitude - 0.185782859,
            air_initial_longitude + 0.235042626,
        )
        southwest = [0] * 2
        southwest[0], southwest[1] = (
            air_initial_latitude - 0.138750844,
            air_initial_longitude - 0.186322428,
        )
        # construction_matrix(100, 130)
        len_1, len_2 = (
            abs(northeast[1] - northwest[1]) / 115,
            abs(northeast[0] - northwest[0]) / 115,
        )
        self.matrix_lenth_sum = []
        Longitude_construction = northwest[1]
        for index_1 in range(128):
            Longitude_construction = Longitude_construction + len_1
            self.matrix_lenth_sum.append(Longitude_construction)
        len_3, len_4 = (
            abs(northwest[1] - southwest[1]) / 115,
            abs(northwest[0] - southwest[0]) / 115,
        )
        self.matrix_high_sum = []
        Latitude_construction = northwest[0]
        for index_2 in range(128):
            Latitude_construction = Latitude_construction - len_4
            self.matrix_high_sum.append(Latitude_construction)

    def normalization(self, data):
        """
        功能：归一化函数
        参数：
            data:{list - 归一化数据}
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/12/22
        """
        minmax_scale = preprocessing.MinMaxScaler()
        dataframe = pd.DataFrame(data)
        # 输入数据DataFrame类型数据   返回的数据是numpy.ndarray
        minmax_scale_data = minmax_scale.fit_transform(dataframe)
        # numpy.ndarray   切片取第0列
        normalization_data = list(minmax_scale_data[:, 0])
        return normalization_data

    def normalization_data(self):
        """
        功能：对飞机、导弹数据进行归一化
        参数：无
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/12/22
        """
        data = {
            "air_speed": self.air_data["unitSpeedKms"] + [1685.32, 648.2],
            "air_high": self.air_data["unit_high"] + [13716, 6.1],
            "air_course": self.air_data["unit_course"] + [359, 0],
            "missile_speed": self.missile_data["missileSpeedKms"] + [3611.4, 0],
            "missile_high": self.missile_data["missile_high"] + [30480, 0],
            "missile_course": self.missile_data["missile_course"] + [359, 0],
        }
        self.normalization_data_dic = {
            k: self.normalization(v) for k, v in data.items()
        }

    def matrix_data(self):
        """
        功能：构造三维矩阵并把归一化数据传入
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/12/22
        """
        # 构造矩阵放入飞机导弹数据
        air_index = -1
        missile_index = -1
        air_hsc_total = []
        self.total_data = []
        self.air_missile_3D_list = []
        self.air_hsc_data_list = []
        # 归一化（0，1）
        for flag in range(1, 500):
            air_hsc_data = []
            row, row_1 = 0, 0
            col, col_1 = 0, 0
            if flag <= len(self.air_data["unitLongitude"]) - 1:
                batch = flag
                while (
                    self.air_data["unitLongitude"][flag] >= self.matrix_lenth_sum[col]
                ):
                    col += 1
                while self.air_data["unitLatitude"][flag] <= self.matrix_high_sum[row]:
                    row += 1
                air_index += 1
                if flag <= len(self.missile_data["missileLongitude"]) - 1:
                    while (
                        self.missile_data["missileLongitude"][flag]
                        >= self.matrix_lenth_sum[col_1]
                    ):
                        col_1 += 1
                    while (
                        self.missile_data["missileLatitude"][flag]
                        <= self.matrix_high_sum[row_1]
                    ):
                        row_1 += 1
                    missile_index += 1
                else:
                    self.normalization_data_dic["missile_speed"][missile_index] = 0
                    self.normalization_data_dic["missile_high"][missile_index] = 0
                    self.normalization_data_dic["missile_course"][missile_index] = 0

                rows = [row, row_1]
                cols = [col, col_1]
                # 创建二维数据
                # 飞机导弹速度二维数据
                speed_pos_matrix = np.zeros(shape=(128, 128), dtype=np.float64)
                # 飞机导弹高度二维数据
                high_pos_matrix = np.zeros(shape=(128, 128), dtype=np.float64)
                # 飞机导弹航向二维数据
                course_pos_matrix = np.zeros(shape=(128, 128), dtype=np.float64)
                # 确定飞机导弹的位置以及飞机导弹速度
                speed_pos_matrix[rows, cols] = (
                    self.normalization_data_dic["air_speed"][air_index],
                    self.normalization_data_dic["missile_speed"][missile_index],
                )
                # print(speed_pos_matrix)
                # 确定飞机导弹的位置以及飞机导弹高度
                high_pos_matrix[rows, cols] = (
                    self.normalization_data_dic["air_high"][air_index],
                    self.normalization_data_dic["missile_high"][missile_index],
                )
                # 确定飞机导弹的位置以及飞机导弹航向
                course_pos_matrix[rows, cols] = (
                    self.normalization_data_dic["air_course"][air_index],
                    self.normalization_data_dic["missile_course"][missile_index],
                )
                # 飞机高度 速度 航向动作数据
                air_hsc_data.append(self.normalization_data_dic["air_speed"][air_index])
                air_hsc_data.append(self.normalization_data_dic["air_high"][air_index])
                air_hsc_data.append(
                    self.normalization_data_dic["air_course"][air_index]
                )

                # 合并二维数据成三维数据
                air_missile_3D = np.array(
                    [speed_pos_matrix, high_pos_matrix, course_pos_matrix]
                )
                air_missile_3D = air_missile_3D.transpose(1, 2, 0)
                self.air_missile_3D_list.append(air_missile_3D)
                self.air_hsc_data_list.append(air_hsc_data)
                self.batch_1 = batch

    def export(self):
        """
        功能：对所有数据进行汇总
        返回：列表
        作者: zhait
        单位：北京华戍防务技术有限公司
        时间：7/12/22
        """
        # self.toatal_data = self.air_missile_3D_list + self.total_data
        air_missile_3D_array = np.array(self.air_missile_3D_list)
        self.air_missile_3D_array_list.append(air_missile_3D_array)
        air_hsc_data_array = np.array(self.air_hsc_data_list)
        air_hsc_data_array = air_hsc_data_array.reshape(self.batch_1, 3, 1)
        self.air_hsc_data_array_list.append(air_hsc_data_array)
        self.batch_list.append(self.batch_1)
        return (
            self.air_missile_3D_array_list,
            self.air_hsc_data_array_list,
            self.batch_list,
        )

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
