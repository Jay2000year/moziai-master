# 时间 : 2022/4/12 17:17
# 作者 : zhait
# 文件 : model_train.py
# 项目 : moziai
# 版权 : 北京华戍防务技术有限公司
# 模型与墨子交互


import os
import pandas as pd
from sklearn import preprocessing
from mozi_ai_sdk.quick_start.env import etc
from mozi_ai_sdk.quick_start.env.env import Environment
from mozi_ai_sdk.quick_start.utils import common
from mozi_utils.geo import *
from mozi_ai_sdk.imitation_learning.tf1.model_test import multi_layer_cnn
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("--matrix_size", type=int, default=128)


class interactive_test:
    def __init__(self):
        # 设置墨子安装目录下bin目录为MOZIPATH，程序会跟进路径自动启动墨子
        os.environ["MOZIPATH"] = etc.MOZI_PATH
        self.args = argparse.parse_args()

        # ①创建环境类对象
        # 环境类对象将仿真服务类中定义的方法串联起来
        self.env = Environment(
            etc.SERVER_IP,
            etc.SERVER_PORT,
            etc.PLATFORM,
            etc.SCENARIO_NAME_8,
            etc.SIMULATE_COMPRESSION,
            etc.DURATION_INTERVAL,
            etc.SYNCHRONOUS,
            etc.APP_MODE,
        )
        # 启动墨子服务端
        # 通过gRPC连接墨子服务端，产生env.mozi_server对象
        # 连接上墨子后，可以通过send_and_recv方法，发送指令
        # 设置推进模式 SYNCHRONOUS
        # 设置决策步长 DURATION_INTERVAL
        self.env.start()
        # 加载想定，产生env.scenario
        # 设置推进速度 SIMULATE_COMPRESSION
        # 初始化全局态势
        # 将所有推演方对象静态化
        self.env.scenario = self.env.reset()
        self.mozi_test()

    def mozi_test(self):
        red_side = self.env.scenario.get_side_by_name("中国")
        bule_side = self.env.scenario.get_side_by_name("美国")
        flag = False
        while not self.env.is_done():

            """
            决策
            """
            if self.env.step_count == 0 and not flag:
                flag = True
                # 获取红方飞机
                aircrafts = red_side.get_aircrafts()
                aircraft = [v for v in aircrafts.values() if v.strName == "飞机1"][0]
                contacts = bule_side.get_weapons()
                # env.step()
                pass
            else:
                self.env.step()
                # 获取飞机与导弹距离
                # 获取飞机探测目标
                red_contacts = red_side.get_contacts()
                missile = common.get_obj_list_by_name(red_contacts, "导弹#2")
                # missile = common.get_obj_list_by_guid(contacts, 'af4b70b8-f823-4834-9e29-c41d52c12052')
                if missile is not None:
                    missile = missile[0]
                    air_missile_distance_sea = aircraft.get_range_to_contact(
                        missile.strGuid
                    )
                    air_missile_distance_sea = eval(air_missile_distance_sea)
                    num_transform = 1.852
                    air_missile_distance = air_missile_distance_sea * num_transform
                    self.env.step()
                    aircrafts = red_side.get_aircrafts()
                    count_num = 0
                    while air_missile_distance <= 22 and air_missile_distance >= 1:
                        air_missile_distance_sea = aircraft.get_range_to_contact(
                            missile.strGuid
                        )
                        if air_missile_distance_sea != "脚本执行出错":
                            air_missile_distance_sea = eval(air_missile_distance_sea)
                            num_transform = 1.852
                            air_missile_distance = (
                                air_missile_distance_sea * num_transform
                            )
                            self.air_Latitude = aircraft.dLatitude
                            self.air_Longitude = aircraft.dLongitude
                            self.air_course = aircraft.fCurrentHeading
                            num = 0.53995680345572
                            self.air_speed = aircraft.fCurrentSpeed / num
                            self.air_high = aircraft.fCurrentAltitude_ASL
                            self.missile_Latitude = missile.dLatitude
                            self.missile_Longitude = missile.dLongitude
                            self.mi_course = missile.fCurrentHeading
                            self.mi_speed = missile.fCurrentSpeed / num
                            self.mi_high = missile.fCurrentAltitude_ASL
                            air_missile_3D_array = self.data_load()
                            self.model_channel(
                                aircraft, air_missile_3D_array, count_num
                            )
                            count_num += 1
                            self.env.step()
                            aircrafts = red_side.get_aircrafts()
                            missile = common.get_obj_list_by_name(red_contacts, "导弹#2")
                            missile = missile[0]
                        else:
                            break
                    missile = common.get_obj_list_by_name(red_contacts, "导弹#2")

                    if len(aircrafts) == 0 or missile is None:
                        return red_side.get_aircrafts()
                else:
                    self.env.step()

    def data_load(self):
        aircraft_speed = []
        aircraft_high = []
        aircraft_course = []
        missile_speed = []
        missile_high = []
        missile_course = []
        # matrix 范围
        northwest = [0] * 2
        northwest[0], northwest[1] = (
            self.air_Latitude + 0.20144026,
            self.air_Longitude - 0.222952932,
        )
        northeast = [0] * 2
        northeast[0], northeast[1] = (
            self.air_Latitude + 0.202979658,
            self.air_Longitude + 0.219372318,
        )
        southeast = [0] * 2
        southeast[0], southeast[1] = (
            self.air_Latitude - 0.185782859,
            self.air_Longitude + 0.235042626,
        )
        southwest = [0] * 2
        southwest[0], southwest[1] = (
            self.air_Latitude - 0.138750844,
            self.air_Longitude - 0.186322428,
        )
        len_1, len_2 = (
            abs(northeast[1] - northwest[1]) / 115,
            abs(northeast[0] - northwest[0]) / 115,
        )
        matrix_lenth_sum = []
        Longitude_construction = northwest[1]
        for index_1 in range(self.args.matrix_size):
            Longitude_construction = Longitude_construction + len_1
            matrix_lenth_sum.append(Longitude_construction)
        # print(matrix_lenth_sum[119])
        len_3, len_4 = (
            abs(northwest[1] - southwest[1]) / 115,
            abs(northwest[0] - southwest[0]) / 115,
        )
        matrix_high_sum = []
        Latitude_construction = northwest[0]
        for index_2 in range(self.args.matrix_size):
            Latitude_construction = Latitude_construction - len_4
            matrix_high_sum.append(Latitude_construction)

        # 对飞机速度，高度，航向进行归一化
        max_speed = 1685.32
        min_speed = 648.2
        max_high = 13716
        min_high = 6.1
        max_course = 359
        min_course = 0
        aircraft_speed.append(max_speed)
        aircraft_speed.append(min_speed)
        aircraft_speed.append(self.air_speed)
        aircraft_high.append(max_high)
        aircraft_high.append(min_high)
        aircraft_high.append(self.air_high)
        aircraft_course.append(max_course)
        aircraft_course.append(min_course)
        aircraft_course.append(self.air_course)
        aircraft_speed_dt = pd.DataFrame(aircraft_speed)
        aircraft_high_dt = pd.DataFrame(aircraft_high)
        aircraft_course_dt = pd.DataFrame(aircraft_course)
        # 归一化
        minmax_scale = preprocessing.MinMaxScaler()
        # 输入数据DataFrame类型数据   返回的数据是numpy.ndarray
        aircraft_speed_dt = minmax_scale.fit_transform(aircraft_speed_dt)
        aircraft_high_dt = minmax_scale.fit_transform(aircraft_high_dt)
        aircraft_course_dt = minmax_scale.fit_transform(aircraft_course_dt)
        # numpy.ndarray   切片取第0列
        air_speed_normalization = list(aircraft_speed_dt[:, 0])
        air_high_normalization = list(aircraft_high_dt[:, 0])
        air_course_normalization = list(aircraft_course_dt[:, 0])

        m_max_speed = 4907.8
        m_min_speed = 0
        m_max_high = 30480
        m_min_high = 0
        m_max_course = 359
        m_min_course = 0
        missile_speed.append(m_max_speed)
        missile_speed.append(m_min_speed)
        missile_speed.append(self.mi_speed)
        missile_high.append(m_max_high)
        missile_high.append(m_min_high)
        missile_high.append(self.mi_high)
        missile_course.append(m_max_course)
        missile_course.append(m_min_course)
        missile_course.append(self.mi_course)
        # 对导弹速度，高度，航向进行归一化
        missile_speed_df = pd.DataFrame(missile_speed)
        missile_high_dt = pd.DataFrame(missile_high)
        missile_course_dt = pd.DataFrame(missile_course)
        # 归一化
        missile_speed_df = minmax_scale.fit_transform(missile_speed_df)
        missile_high_dt = minmax_scale.fit_transform(missile_high_dt)
        missile_course_dt = minmax_scale.fit_transform(missile_course_dt)
        missile_speed_normalization = list(missile_speed_df[:, 0])
        missile_high_normalization = list(missile_high_dt[:, 0])
        missile_course_normalization = list(missile_course_dt[:, 0])

        # 构造矩阵放入飞机导弹数据
        air_nor_index = -1
        missile_nor_index = -1
        air_missile_3D_list = []
        air_hsc_data_list = []
        air_hsc_data = []
        row, row_1 = 0, 0
        col, col_1 = 0, 0

        while self.air_Longitude >= matrix_lenth_sum[col]:
            col += 1
        while self.air_Latitude <= matrix_high_sum[row]:
            row += 1

        while self.missile_Longitude >= matrix_lenth_sum[col_1]:
            col_1 += 1
        while self.missile_Latitude <= matrix_high_sum[row_1]:
            row_1 += 1

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
            air_speed_normalization[2],
            missile_speed_normalization[2],
        )
        # print(speed_pos_matrix)
        # 确定飞机导弹的位置以及飞机导弹高度
        high_pos_matrix[rows, cols] = (
            air_high_normalization[2],
            missile_high_normalization[2],
        )
        # print(high_pos_matrix)
        # 确定飞机导弹的位置以及飞机导弹航向
        course_pos_matrix[rows, cols] = (
            air_course_normalization[2],
            missile_course_normalization[2],
        )
        # 飞机高度 速度 航向动作数据
        air_hsc_data.append(air_speed_normalization[2])
        air_hsc_data.append(air_high_normalization[2])
        air_hsc_data.append(air_course_normalization[2])

        # 合并二维数据成三维数据
        air_missile_3D = np.array(
            [speed_pos_matrix, high_pos_matrix, course_pos_matrix]
        )
        air_missile_3D = air_missile_3D.transpose(1, 2, 0)
        air_missile_3D_list.append(air_missile_3D)
        air_hsc_data_list.append(air_hsc_data)

        air_missile_3D_array = np.array(air_missile_3D_list)
        air_missile_3D_array = np.tile(air_missile_3D_array, (24, 1, 1, 1))
        # air_missile_3D_array = air_missile_3D_array.transpose(0, 2, 3, 1)
        print(air_hsc_data)
        # air_missile_3D_list = []
        return air_missile_3D_array

    def model_channel(self, aircraft, air_missile_3D_array, count_num):
        air_sp, air_hi, air_cou = multi_layer_cnn("lstm", air_missile_3D_array, 24)
        air_original_sp = 1037.12 * air_sp + 648.2
        air_original_hi = 13709.9 * air_hi + 6.1
        air_original_cou = 359 * air_cou + 0
        print(air_original_sp, air_original_hi, air_original_cou)
        # return air_missile_3D_array,air_hsc_data_array
        # data_load()
        # 设置飞机高度，航向，航速
        # 设置飞机高度
        aircraft.set_desired_height(air_original_hi, "true")
        # 设置航路点
        # 返回经纬度字典
        way_point_dict = get_point_with_point_bearing_distance(
            self.air_Latitude, self.air_Longitude, air_original_cou, 3
        )
        aircraft.set_waypoint(way_point_dict["longitude"], way_point_dict["latitude"])

        # 设置飞机速度
        aircraft.set_desired_speed(air_original_sp)
        # aircraft.control_operation_unit('F-14E型“超级雄猫”战斗机', 3000, 480, 180)
        # 设置飞机航向
        aircraft.set_unit_heading(air_original_cou)
        # # env.step()
        # 修改单元条令
        aircraft_doctrine = aircraft.get_doctrine()

        if count_num == 0:
            # 修改飞机的电磁管控
            aircraft_doctrine.set_emcon_according_to_superiors("no", "false")
            aircraft_doctrine.unit_obeys_emcon("true")
            aircraft_doctrine.set_em_control_status("Radar", "Passive")
            # 是否自动规避
            aircraft_doctrine.evade_automatically("false")
        # # 删除航路点
        # aircraft.delete_coursed_point(point_index=None, clear=True)
        # count_num+=1
        # env.step()
        # 设置飞机航速档
        # aircraft.set_throttle(3)


def run():
    while True:
        interactive = interactive_test()
        interactive_1 = interactive.mozi_test()
        if interactive_1 is None:
            continue
        else:
            interactive.mozi_test()


if __name__ == "__main__":
    run()
