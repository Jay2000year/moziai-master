# 时间 ： 2020/7/20 17:03
# 作者 ： sean
# 文件 ： main_versus.py
# 项目 ： moziAIBT
# 版权 ： 北京华戍防务技术有限公司
from flask import Flask
import requests
import time
import pickle

from mozi_ai_sdk.net_Decision.env.env import Environment
from mozi_ai_sdk.net_Decision.env import etc
from mozi_ai_sdk.net_Decision.utils.Inferring import create_input, sorte_values, IsPtInPoly, add_mssn, \
    select_nearest_unit, set_ref, excel_convert_dic
import os

app = Flask(__name__)

os.environ['MOZIPATH'] = 'D:\\mozi_server\\Mozi\\MoziServer\\bin'
AREA = [('20.78', '127.35'), ('23.82', '138.62'), ('18.68', '135.53'), ('17.1', '131.2')]
UP_AREA = [('23.63', '136.20'), ('23.08', '137.22'), ('23.38', '136.0'), ('22.82', '137.01')]  # 在划定的区域内-目标距离近

type2val = excel_convert_dic('Value_data.xls')
DOWN_AREA = [('20.69', '134.30'), ('20.02', '135.33'), ('19.92', '135.14'), ('20.42', '134.21')]
weapon_db_guid = 'hsfw-dataweapon-00000001003826'

# url = "http://127.0.0.1:5000/http/query"
url = "http://192.168.3.118:5000/http/query"


def run():
    # 给服务端发送数据，同时接收返回数据
    input_features = [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    send_data = {'探测态势': input_features}
    r = requests.post(url, json=send_data)
    time.sleep(1)
    out = r.content
    out = pickle.loads(out)
    print('收到服务端返回数据是：', out)

#     pass


# def run(env, side_name=None):
#     """
#        行为树运行的起始函数
#        :param env: 墨子环境
#        :param side_name: 推演方名称l
#        :return:
#        """
#     if not side_name:
#         side_name = '红方'
#     # 连接服务器，产生mozi_server
#     env.start()
#
#     # 重置函数，加载想定,拿到想定发送的数据
#     env.scenario = env.reset()
#     side = env.scenario.get_side_by_name(side_name)
#     step_count = 0
#
#     # 已经攻击过的zb导弹
#     zb_attacked = []
#     for i, point in enumerate(UP_AREA):
#         side.add_reference_point(f'rf{i}', point[0], point[1])
#     while True:
#         zb = [v for _, v in side.aircrafts.items() if
#               'ZB导弹' in v.strName and v not in zb_attacked and '在空' in v.strActiveUnitStatus]  # v.strActiveUnitStatus] 在空不影响
#         patrol_mssns = [v for v in side.patrolmssns.values() if '攻击波次' in v.strName]
#         contacts = side.contacts
#         input_features, activeunits = create_input(contacts)  # 在可以攻击的四边形区域内的所有单元-单元对象
#         # 列表嵌套
#         active_dic = {v.strGuid: v for v in activeunits}
#
#         # 给服务端发送数据，同时接收返回数据
#         send_data = {'探测态势': input_features}
#         r = requests.post(url, json=send_data)
#         time.sleep(1)
#         out = r.content
#         out = pickle.loads(out)
#         print('收到服务端返回数据是：', out)
#
#         target_lst, tn = sorte_values(out, type2val)  # [('提康德罗加', 1.0), ('油料补给舰 ', 0.453854782117174),...]
#         # if step_count % 10 == 0:
#         #     show_prob(tn, now_time, target_lst, all_type_units)
#         for k, v in active_dic.items():
#             if len([mssn for mssn in patrol_mssns if v.strName in mssn.strName]) == 0:
#                 add_mssn(side, v)
#             else:
#                 set_ref(side, v)
#         flag = False
#         for target in target_lst:  # 算法给出的打击目标 ,但是打击目标需要在单元里
#             for v in activeunits:
#                 if target[0] in v.strName:  # 节点名称
#                     for air in zb:  # 已经攻击过的就不再攻击
#                         # 在划定区域内，打击目标，如果相同目标有多个，选择最短距离的目标
#                         if IsPtInPoly(air.dLatitude, air.dLongitude, UP_AREA):
#                             k = select_nearest_unit(target[0], activeunits, air)
#                             print(f'====打击目标是{v.strName}=======')
#
#                             air.cancel_assign_unit_to_mission()  # 删除掉原来的任务
#                             # 攻击波次-E-2D型“先进鹰眼”舰载预警机 #753 这个是单元名称做的任务名称，和target[0]
#                             # # 初始的时候没有任务，但是不影响，因为初始也不会走到这里
#                             the_mssn = [item for item in patrol_mssns if target[0] in item.strName][0]
#                             the_mssn.assign_units({air.strGuid: air})
#                             doct = air.get_doctrine()
#                             doct.set_weapon_control_status('weapon_control_status_surface', 0)  # 对海自由开火
#                             air.allocate_weapon_to_target(k, weapon_db_guid, 1)
#                             zb_attacked.append(air)
#                             if flag:
#                                 continue
#                             flag = True
#                     if flag:
#                         break
#             if flag:
#                 break
#
#         env.step()
#         print(f"'推演步数：{step_count},本方得分：{side.iTotalScore}")
#         step_count += 1
#         if env.is_done():
#             send_data = {'探测态势': '推演已结束！'}
#             r = requests.post(url, json=send_data)
#             print('推演已结束！')
#             # sys.exit(0)
#             break
#         else:
#             pass


def main():
    # env = Environment(etc.SERVER_IP, etc.SERVER_PORT, etc.PLATFORM, etc.SCENARIO_NAME, etc.SIMULATE_COMPRESSION,
    #                   etc.DURATION_INTERVAL, etc.SYNCHRONOUS, etc.app_mode)
    # run(env)
    run()

main()
