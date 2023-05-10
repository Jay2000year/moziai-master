# -*- coding:utf-8 -*-
# File name : theater_agent.py
# Create date : 2022/9/22
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.0

import pyproj

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


# 战区级规则体类，负责全局态势分析与各旅级规则体的责任区划分及作战目标安排
class TheaterCommand:
    def __init__(self, name):
        # 这些属性为固定属性，不随态势或自身状态更新而更新
        self.name = name
        self.subordinate_command = []  # 下级规则体

        # 这些属性是动态属性，在特定情况或条件下更新
        self.situation = None  # 全局态势

    def step(self, side):
        """
        随推演步进运行规则体
        """
        self.situation = side
        self.subordinate_command[0].local_situation = self.situation
        self.subordinate_command[0].step()
