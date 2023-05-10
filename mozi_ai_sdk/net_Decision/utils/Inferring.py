from pickle import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax
from sklearn import tree
import joblib
import math
import numpy as np
from matplotlib import font_manager as fm, rcParams

from mozi_ai_sdk.net_Decision.utils import threat_cal
from mozi_ai_sdk.net_Decision.utils.demo import get_A_value
from mozi_ai_sdk.net_Decision.utils.geomutil import ConvertLLA2XYZ

from mozi_ai_sdk.net_Decision.utils.Integer_Line import get_line
import mozi_ai_sdk.net_Decision.utils.threat_cal

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

tmp = '类型-节点名称-体系价值'
AREA = [('20.78', '127.35'), ('23.82', '138.62'), ('18.68', '135.53'), ('17.1', '131.2')]


def load_model(file_path):
    loaded_RFmodel = joblib.load(file_path)
    return loaded_RFmodel


def RandomForestInfer(RFmodel, Infer_input_x):
    InferOut = RFmodel.predict(Infer_input_x)
    InferOut_proba = RFmodel.predict_proba(Infer_input_x)
    return InferOut, InferOut_proba


def Prior_fuse(Prior, Infer_proba):
    PostProba = Prior + Infer_proba
    return PostProba


def With_Prior_Infer(RFmodel, Infer_input_x, Prior=1, ):
    InferOut_proba = RFmodel.predict_proba(Infer_input_x)
    Prior = np.asarray(Prior)
    PostProba = Prior_fuse(Prior, InferOut_proba)
    postOut = PostProba.argsort()[:, ::-1][:, 0]
    return postOut, PostProba


def IsPtInPoly(aLon, aLat, pointList):
    """
    :param aLon: double 经度
    :param aLat: double 纬度
    :param pointList: list [(lon, lat)...] 多边形点的顺序需根据顺时针或逆时针，不能乱
    """
    aLon = float(aLon)
    aLat = float(aLat)
    iSum = 0
    iCount = len(pointList)

    if iCount < 3:
        return False

    for i in range(iCount):
        pLon1 = float(pointList[i][0])
        pLat1 = float(pointList[i][1])
        if i == iCount - 1:
            pLon2 = float(pointList[0][0])
            pLat2 = float(pointList[0][1])
        else:
            pLon2 = float(pointList[i + 1][0])
            pLat2 = float(pointList[i + 1][1])

        if ((aLat >= pLat1) and (aLat < pLat2)) or ((aLat >= pLat2) and (aLat < pLat1)):

            if abs(pLat1 - pLat2) > 0:
                pLon = pLon1 - ((pLon1 - pLon2) * (pLat1 - aLat)) / (pLat1 - pLat2)
                if pLon < aLon:
                    iSum += 1

    if iSum % 2 != 0:
        return True
    else:
        return False


UP_AREA = [('23.95', '136.54'), ('23.53', '137.47'), ('23.20', '137.32'), ('23.84', '136.28')]
DOWN_AREA = [('23.85', '136.64'), ('19.66', '135.15'), ('19.43', '135.23'), ('19.74', '134.17')]
lan = '23.678'
lon = '136.987'

a = IsPtInPoly(lan, lon, UP_AREA)
print(a)
input_features = [[1, 1, 3, 2, 5, 0, 2, 10, 10, 10]]


def create_input(contacts):
    carrier = [v for _, v in contacts.items() if
               '航空母舰' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    cruiser = [v for _, v in contacts.items() if
               '巡洋舰' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    destroyer = [v for _, v in contacts.items() if
                 '驱逐舰' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    replenishment_oiler = [v for _, v in contacts.items() if
                           '补给油船' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    fighter = [v for _, v in contacts.items() if '战斗机' in v.strName]  # 战斗机只有一种  800*800区域内所有
    aew = [v for _, v in contacts.items() if '预警机' in v.strName]  # 800*800区域内所有的预警机
    helicopter = [v for _, v in contacts.items() if
                  '直升机' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    fuel_dispenser = [v for _, v in contacts.items() if
                      '加油机' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    amphibious_assault_ship = [v for _, v in contacts.items() if
                               '两栖' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    amphibious_transport_dock = [v for _, v in contacts.items() if
                                 '坞登' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    # goods= [v for _, v in side.contacts.items() if
    #                        '干货船' in v.strName and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]

    # fighter = [v for _, v in side.aircrafts.items() if v.m_Type == 2001 and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    # aew = [v for _, v in side.aircrafts.items() if v.m_Type == 4002 and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    # helicopter = [v for _, v in side.aircrafts.items() if
    #               v.m_Category == 2003 and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    # fuel_dispenser = [v for _, v in side.aircrafts.items() if
    #                   v.m_Type == 8001 and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    # amphibious_assault_ship = [v for _, v in side.ships.items() if
    #                            v.m_Category == 2003 and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    # amphibious_transport_dock = [v for _, v in side.facilities.items() if
    #                              v.m_Category == 3005 and IsPtInPoly(v.dLatitude, v.dLongitude, AREA)]
    lst = [carrier, cruiser, destroyer, replenishment_oiler, fighter, aew, helicopter, fuel_dispenser,
           amphibious_assault_ship, amphibious_transport_dock]
           # amphibious_assault_ship, amphibious_transport_dock, goods]
    unit_lst = [item for type_units in lst for item in type_units]
    input_lst = []
    for i in lst:
        input_lst.append(len(i))
    return [input_lst], unit_lst


def select_nearest_unit(name, unit_lst, attach_unit):
    """
    选择最近的单元为攻击的目标
    :param name: 从单元的名字一部分（阿里伯克#1，阿里伯克#2 中的阿里伯克）
    :param unit_lst: 所有的目标单元
    :param attach_unit: 攻击的zb飞机导弹
    :return: 最近单元的guid
    """
    id_to_distance = {}
    small_lst = [unit for unit in unit_lst if name in unit.strName]  # 选出同名不同编号的所有单元
    for unit in small_lst:
        distance = attach_unit.get_range_to_contact(unit.strGuid)
        id_to_distance[unit.strGuid] = float(distance)
    distance_order = sorted(id_to_distance.items(),
                            key=lambda x: x[1])  # [(id1, distance1),(id2, distance2)]
    if distance_order:
        return distance_order[0][0]  # 返回guid


# 2D-array, shape ==> (n_samples, n_features) Template, input [Carrier, Cruiser, Destroyer, Replenishment oiler,
# fighter, Airborne Early Warning and Control, helicopter, Fuel dispenser, Amphibious assault ship, Amphibious
# transport dock]
#         [[0, 1, 3, 1, 0, 0, 0, 0, 0, 0]]
# Template [航母、巡洋舰、驱逐舰、补给油船、战斗机，机载预警和控制，直升机，加油机，两栖攻击舰，两栖运输码头]
# Template, output [S-CSG, M-CSG, ARG, SAG]
type_names = ["S-CSG", "M-CSG", "ARG", "SAG"]


def excel_convert_dic(file_name):
    value_DF = pd.read_excel(file_name, header=0)
    type2val = {}
    for type_name in value_DF['类型'].unique():
        sub_DF = value_DF[value_DF['类型'] == type_name]

        ls_avg_val = []
        for name in sub_DF["节点名称"].unique():
            ls_avg_val.append((name, sub_DF[sub_DF["节点名称"] == name]['体系价值'].mean()))

        ls_avg_val.sort(key=lambda x: x[1], reverse=True)
        type2val[type_name] = ls_avg_val
    return type2val


def sorte_values(out, type2val):  # out 只有一个值
    for out_ind in out:
        t_n = type_names[out_ind]
        sorted_value_ls = type2val[t_n]
        return sorted_value_ls, t_n
'''
def show_prob(tn, now_time, target_lst, all_type_units):
    """
    :param tn: 目标体系分类名称
    :param now_time: 想定的当前时间
    :param target_lst: 目标类型及其权重, 例子：[('提康德罗加', 1.0), ('油料补给舰 ', 0.453854782117174), ('阿利伯克', 0.369293325336399), ('SH-60直升机 ', 0.342069192032989)]
    :param all_type_units: A/b/c类目标对象列表:驱逐舰和巡洋舰
    :return:
    """
    # plt.figure(20)
    grid = plt.GridSpec(1, 5, wspace=0.8)
    plt.subplot(grid[0, 0:2])
    # plt.subplot(121)
    plt.title('目标体系分类：' + tn + '(' + now_time + ')')

    risk = []
    labels = []
    for i in target_lst:
        risk.append(i[1])
        labels.append(i[0])
    plt.xticks(rotation=300, fontsize=8)
    plt.bar(range(len(risk)), risk, tick_label=labels)
    # plt.tight_layout(w_pad=2)
    # 画热力图
    plt.subplot(grid[0, 2:5])
    # plt.subplot(122)
    grid_range = 800

    # 初始化网格矩阵
    a = np.zeros([grid_range, grid_range])  # 重新设置为0，不会影响下一次调用的结果
    if all_type_units:
        for unit in all_type_units:
            point = unit.dLatitude, unit.dLongitude
            point_x = int(ConvertLLA2XYZ(point).x)
            point_y = int(ConvertLLA2XYZ(point).y)
            a = get_A_value(a, point_x, point_y)
        X = np.arange(0, grid_range, 1)
        Y = np.arange(0, grid_range, 1)
        X, Y = np.meshgrid(X, Y)
        plt.axis('equal')
        plt.contourf(X, Y, a)
        plt.colorbar()  # 图例
        plt.title('目标区威胁势场')
        plt.xlabel("X")
        plt.ylabel("Y")

    plt.draw()
    plt.pause(1)  # 推演会暂停
    plt.close()


def plot_map_heat(type_obj_lst, type_name):
    # 网格范围
    # 暂时没用到 2021/12/13

    heat_data = None
    grid_range = 800

    # 初始化网格矩阵
    a = np.zeros((grid_range, grid_range))  # 重新设置为0，不会影响下一次调用的结果
    if type_obj_lst:
        for unit in type_obj_lst:
            point = unit.dLatitude, unit.dLongitude
            point_x = int(ConvertLLA2XYZ(point).x)
            point_y = int(ConvertLLA2XYZ(point).y)
            heat_data = get_A_value(a, point_x, point_y)
        X = np.arange(0, grid_range, 1)
        Y = np.arange(0, grid_range, 1)
        X, Y = np.meshgrid(X, Y)
        plt.axis('equal')
        plt.contourf(X, Y, heat_data)
        plt.title(f'{type_name}类目标区威胁势场')
        plt.xlabel("X")  ## x轴标签
        plt.ylabel("Y")  ## y轴标签
'''

################################# 画图 ############################


def add_mssn(side, v):
    point_list = []
    rp1 = side.add_reference_point('core_in_' + 'rp1', v.dLatitude + 0.04, v.dLongitude - 0.04)
    rp2 = side.add_reference_point('core_in_' + 'rp2', v.dLatitude + 0.04, v.dLongitude + 0.04)
    rp3 = side.add_reference_point('core_in_' + 'rp3', v.dLatitude - 0.04, v.dLongitude + 0.04)
    rp4 = side.add_reference_point('core_in_' + 'rp4', v.dLatitude - 0.04, v.dLongitude - 0.04)
    point_list.append([rp1.strName, rp2.strName, rp3.strName, rp4.strName])
    mssn = side.add_mission_patrol(f'攻击波次-{v.strName}', 1, point_list)
    return mssn
    # core_pat_mssn.assign_units(v)


def set_ref(side, v):
    side.set_reference_point('core_in_' + 'rp1', v.dLatitude + 0.04, v.dLongitude - 0.04)
    side.set_reference_point('core_in_' + 'rp2', v.dLatitude + 0.04, v.dLongitude + 0.04)
    side.set_reference_point('core_in_' + 'rp3', v.dLatitude - 0.04, v.dLongitude + 0.04)
    side.set_reference_point('core_in_' + 'rp4', v.dLatitude - 0.04, v.dLongitude - 0.04)

def value_risk(unit,unit_lst):
    ##威胁案例，8个目标坐标以及自身目标
    grid_size = 800 / 5
    thread_ins = threat_cal.ThreatField(grid_size=int(np.ceil(grid_size)))
    # 自身座标
    self_pos = np.array([unit.dLatitude, unit.dLongitude]) / 5
    target_pos = [[i.dLatitude, i.dLongitude] for i in unit_lst]
    if target_pos:
        target_pos = np.array(target_pos)/5
        ss = [i.strName for i in unit_lst]
        catogory = list(map(func1, ss))
        target_pos = np.array(
            [[350, 200], [450, 200], [250, 250], [550, 250], [350, 350], [450, 350], [400, 600], [400, 200]]) / 5
        catogory = ["A", "A", "A", "A", "A", "A", "B", "C", ]
        thread_ins.CalThreatField(target_pos, catogory)
        ThreatFieldArray = thread_ins.ThreatFieldArray

        # 画图
        # X = np.arange(0, grid_size, 1)
        # Y = np.arange(0, grid_size, 1)
        # X, Y = np.meshgrid(X, Y)
        # plt.axis('equal')
        # plt.contourf(X, Y, ThreatFieldArray)
        # plt.colorbar()  # 图例
        # plt.title('目标区威胁势场')
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.draw()
        # plt.pause(1)
        # plt.close()
        tar2risk = []
        for postuple in target_pos:

            points = get_line(self_pos, postuple)
            Risk = 0
            for pt in points:
                Risk = Risk + ThreatFieldArray[pt]
            tar2risk.append(Risk)
        tar2risk = np.asarray(tar2risk)
        tar2risk = (tar2risk - tar2risk.min()) / (tar2risk.max() - tar2risk.min())
        ### 输出威胁列表，对准对应8个目标
        print(tar2risk)
        # 把威胁值最大的单元找出guid
        tar2risk = tar2risk.tolist()
        unit_id = tar2risk.index(max(tar2risk))
        unit = unit_lst[unit_id]
        return unit





if __name__ == "__main__":
    RFmodel = load_model('RFmodel_saved.joblib')
    input_features = [[1, 1, 3, 2, 5, 0, 2, 0, 0, 0]]

    # 威胁案例，8个目标坐标以及自身目标
    grid_size = 800 / 5
    thread_ins = threat_cal.ThreatField(grid_size=int(np.ceil(grid_size)))
    # 自身目标
    self_pos = np.array([700, 700]) / 5
    # 8个目标坐标
    target_pos = np.array(
        [[350, 200], [450, 200], [250, 250], [550, 250], [350, 350], [450, 350], [400, 600], [400, 200]]) / 5
    catogory = ["A", "A", "A", "A", "A", "A", "B", "C", ]

    ## 计算威胁，输出关于所有目标的威胁求和，已归一化，maxmin形式。
    thread_ins.CalThreatField(target_pos, catogory)
    ThreatFieldArray = thread_ins.ThreatFieldArray
    tar2risk = []
    for postuple in target_pos:

        points = get_line(self_pos, postuple)
        Risk = 0
        for pt in points:
            Risk = Risk + ThreatFieldArray[pt]
        tar2risk.append(Risk)
    tar2risk = np.asarray(tar2risk)
    tar2risk = (tar2risk - tar2risk.min()) / (tar2risk.max() - tar2risk.min())
    ### 输出威胁列表，对准对应8个目标
    print(tar2risk)

    ## 2D-array, shape ==> (n_samples, n_features)
    ## Template, input [Carrier, Cruiser, Destroyer, Replenishment oiler, fighter, Airborne Early Warning and Control, helicopter, Fuel dispenser, Amphibious assault ship, Amphibious transport dock]
    ## Template, output [S-CSG, M-CSG, ARG, SAG]
    type_names = ["S-CSG", "M-CSG", "ARG", "SAG"]

    # out, out_prob =RandomForestInfer(RFmodel, input_features)
    ## prior先验概率分布，与舰队类型匹配。
    out, out_prob = With_Prior_Infer(RFmodel=RFmodel, Infer_input_x=input_features, Prior=[0.25, 0.25, 0.5, 0], )
    type2val = excel_convert_dic('./Value_data.xls')
    # type2val = {}
    # for type_name in value_DF['类型'].unique():
    #     sub_DF = value_DF[value_DF['类型'] == type_name]
    #
    #     ls_avg_val = []
    #     for name in sub_DF["节点名称"].unique():
    #         ls_avg_val.append((name, sub_DF[sub_DF["节点名称"] == name]['体系价值'].mean()))
    #
    #     ls_avg_val.sort(key=lambda x: x[1], reverse=True)
    #     type2val[type_name] = ls_avg_val

    for out_ind in out:
        t_n = type_names[out_ind]
        sorted_value_ls = type2val[t_n]
        ## 输出价值列表
        print(sorted_value_ls)
        ### 列表输出，对应每次决策识别结果。


def func1(name):
    if '驱逐舰' in name or '巡洋舰' in name:  # if ''驱逐舰' or '巡洋舰' in name
        return 'A'
    elif '预警机' in name:
        return 'B'
    elif '航空母舰' in name or '两栖' in name:
        return 'C'


