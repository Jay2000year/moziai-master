# -*- coding:utf-8 -*-
# File name : brigade_command.py
# Create date : 2022/9/20
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.0

import pyproj
from mozi_ai_sdk.AA_command_agent.battalion_command.battalion_config import *

# 高精度地球地理数据计算模型,投影坐标系为‘WGS_1984_UTM_Zone_50N’,坐标系编号为'EPSG 32650'
crs = pyproj.CRS.from_epsg(32650)
# <Derived Projected CRS: EPSG:32650>
# Name: WGS 84 / UTM zone 50N
# Axis Info [cartesian]:
# - E[east]: Easting (metre)
# - N[north]: Northing (metre)
# Area of Use:
# - name: Between 114°E and 120°E, northern hemisphere between equator and 84°N, onshore and offshore.
#         Brunei. China. Hong Kong. Indonesia. Malaysia - East Malaysia - Sarawak. Mongolia.
#         Philippines. Russian Federation. Taiwan.
# - bounds: (114.0, 0.0, 120.0, 84.0)
# Coordinate Operation:
# - name: UTM zone 50N
# - method: Transverse Mercator
# Datum: World Geodetic System 1984 ensemble
# - Ellipsoid: WGS 84
# - Prime Meridian: Greenwich
earth_model = crs.get_geod()


# 旅级规则体类，负责某个区域中的所有营级规则体的协同与任务目标分配
class BrigadeCommand:
    def __init__(self, name):
        # 这些属性为固定属性，不随态势或自身状态更新而更新
        self.name = name
        self.superior_command = None  # 上级规则体
        self.subordinate_command = []  # 下级规则体
        # self.init_situation = None  # 初始态势

        # 这些属性是动态属性，在特定情况或条件下更新
        self.subordinate_data_receive = {}  # 下级规则体上报的数据
        self.protect_target = {}  # 保卫目标
        self.local_situation = None  # 旅级局部态势
        self.unable = {}  # 各营无法攻击的目标
        self.target_allocated = []  # 已分配营进行拦截的目标
        self.target_not_allocated = []  # 未分配营进行拦截的目标
        self.battalion_munition_status = {}  # 各营弹药状况

    def target_prediction(self):
        """
        预测来袭目标的攻击目的
        """
        # todo:根据航向预测来袭目标攻击的单元，并根据目标速度与距离估算剩余拦截时间窗口

    def self_update(self):
        """
        更新自身状态
        """
        # 汇总各营交战情况，筛选出已分配火力的目标与未分配火力的目标
        self.target_allocated.clear()
        self.target_not_allocated.clear()
        # for guid, contact in self.local_situation.contacts.items():
        #     if contact.m_ContactType == 0:
        #         self.target_not_allocated.append(contact)
        #     elif contact.m_ContactType == 1 and 'VAMPIRE' in contact.strName:
        #         self.target_not_allocated.append(contact)
        #     elif contact.m_ContactType == 1 and 'GuidedWeapon' in contact.strName:
        #         self.target_not_allocated.append(contact)
        # for name, bn_status in self.subordinate_data_receive.items():
        #     self.target_not_allocated = [x for x in self.target_not_allocated if x.m_ActualUnit not in bn_status['attacking_list']]
        #     self.target_allocated = self.target_allocated + [x for x in self.target_not_allocated if x.m_ActualUnit in bn_status['attacking_list']]
        # for tgt in bn_status['attacking_list']:
        # self.target_not_allocated.remove(tgt)
        # self.target_allocated.append(tgt)

    def step(self):
        """
        随推演步进运行规则体
        """
        self.self_update()
        # self.target_prediction()
        self.subordinate_command[0].set_combat_mode(CombatMode.GENERAL)
        self.subordinate_command[0].target_list = self.target_not_allocated
        self.subordinate_command[0].local_situation = self.local_situation
        self.subordinate_command[0].step()
