#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import numpy as np
from math import cos
from math import radians
import os

from mozi_utils import pylog
from mozi_utils.geo import get_point_with_point_bearing_distance
from mozi_utils.geo import get_degree
from mozi_utils.geo import get_two_point_distance

from mozi_ai_sdk.base_env import BaseEnvironment
from . import etc

"""
作者：刘占勇
日期：2020.05.04
功能：无人机反坦克想定环境类，UAT=UAV Anti Tank
"""


def _get_waypoint_heading(last_heading, action_value):
    """
    获取航路点朝向
    """

    current_heading = last_heading + action_value
    if current_heading < 0:
        current_heading += 360
    if current_heading > 360:
        current_heading -= 360
    return current_heading


class EnvUavAntiTank(BaseEnvironment):
    """
    作者：刘占勇
    日期：2020.05.04
    功能：构造函数
    参数：无
    返回：无
    """

    def __init__(
        self,
        IP,
        AIPort,
        platform,
        scenario_name,
        simulate_compression,
        duration_interval,
        synchronous,
        app_mode,
        platform_mode=None,
    ):
        super().__init__(
            IP,
            AIPort,
            platform,
            scenario_name,
            simulate_compression,
            duration_interval,
            synchronous,
            app_mode,
            platform_mode,
        )

        self.SERVER_PLAT = platform
        self.state_space_dim = 14  # 状态空间维度
        self.action_space_dim = 1
        self.action_max = 1

        self.red_unit_list = None
        self.observation = None
        self.red_side_name = "红方"
        self.blue_side_name = "蓝方"

    def reset(self, app_mode=None):
        """
        重置    Signature of method ‘ret()’ does not match signature of base method in class ‘base_env.reset()’
        返回：当前状体及回报值
        """
        # 调用父类的重置函数
        super(EnvUavAntiTank, self).reset()

        # 构建各方实体
        self._construct_side_entity()
        self._init_unit_list()

        self.m_StartTime = self.scenario.m_StartTime  # 想定开始时间
        self.m_Time = self.scenario.m_Time  # 想定当前时间

        # state_now = self.get_observation()
        state_now = self.get_related_state()
        reward_now = self.get_reward(None)
        return state_now, reward_now

    def get_related_state(self):
        """将飞机的状态，由绝对经纬度和朝向，改为相对目标点的经纬度和朝向"""
        # lat2, lon2 = self.get_target_point()
        obs = self.get_observation()
        # obj_coords = self.get_obj_coord()
        obj_coords = np.zeros(self.state_space_dim, dtype=np.float32)

        contact_ = {}
        for k, v in self.redside.contacts.items():
            contact_[v.m_ActualUnit] = {"lat": v.dLatitude, "lon": v.dLongitude}
        contacts_actual_guid = sorted(contact_)
        i = 0
        for actual_guid in contacts_actual_guid:
            obj_coords[i] = obs[0] - contact_[actual_guid]["lon"]
            obj_coords[i + 1] = obs[1] - contact_[actual_guid]["lat"]
            i += 2
        obj_coords[10] = obs[2] / 360.0
        time_delta = self.m_Time - self.m_StartTime
        obj_coords[11] = time_delta / 3600.0
        obj_coords[12] = time_delta / 7200.0
        obj_coords[13] = time_delta / 14400.0

        state_ = np.array(obj_coords)
        # print(f"state: {state_}")
        return state_

    def execute_action(self, action_value):
        super(EnvUavAntiTank, self).step()
        self.m_Time = self.scenario.m_Time  # 想定当前时间

        waypoint = self._get_aircraft_waypoint(action_value)  # 根据动作计算飞机的期望路径点

        longitude = self.observation[0]  # 当前的位置
        latitude = self.observation[1]
        distance = self.get_target_distance(latitude, longitude)

        airs = self.redside.aircrafts
        for guid in airs:
            aircraft = airs[guid]
            if distance < etc.target_radius:
                # 如果目标距离小于打击距离，且已发现目标，则自动攻击之 {name='坦克排(T-72型主战坦克 x 4) #4', guid='d8686a56-6e55-4b82-953e-4b9d124e5579'}
                if self._check_is_find_target():
                    # target_guid = self._get_target_guid()
                    target_guid = self._get_contact_target_guid()
                    print("%s：自动攻击目标" % datetime.time())
                    # aircraft.auto_attack_target(target_guid)
                    cmd = (
                        f"ScenEdit_AttackContact('{guid}', '{target_guid}',"
                        + "{mode = 0})"
                    )
                    self.scenario.mozi_server.send_and_recv(cmd)
            else:
                # 如果目标距离大于打击距离，则继续机动
                lon, lat = self._deal_point_data(waypoint)
                # print("set waypoint:%s %s" % (lon, lat))
                aircraft.set_waypoint(lon, lat)

        # 动作下达了，该仿真程序运行，以便执行指令（），许怀阳 2020050716:58
        self.mozi_server.run_grpc_simulate()

        # 更新数据时，会被阻塞，实现与仿真的同步
        self._update()

        # # 动作执行完了，该继续仿真了
        # self.mozi_server.run_simulate()

        # obs = self.get_observation()
        obs = self.get_related_state()

        reward, distance = self.get_reward(action_value)

        done = self.check_done()
        if distance >= 50 * 1000:
            done = True
            reward += -1.5
        return np.array(obs), reward, done

    def get_reward(self, action_value):
        """
        获取奖励
        """
        reward = 0.0
        distance = 0
        if action_value is not None:
            # 距离目标越近，奖励值越大
            distance_reward, distance = self._get_distance_reward(action_value)
            reward += distance_reward

            # 不需要额外的击杀奖励，直接使用距离和朝向奖励！

            if distance > etc.target_radius:  # 如果进入了一个距离范围
                reward -= 0.1
            if not self._check_aircraft_exist():  # 飞机被打死，则降低奖赏值
                reward += -1.5
            if not self._check_target_exist():  # 目标被打死，则增加奖赏值
                reward += 1.0
        return reward, distance

    def _get_distance_reward(self, action_value):
        """
        获取距离奖励
        """
        obs = self.observation
        longitude = obs[0]
        latitude = obs[1]
        heading = obs[2]
        distance = self.get_target_distance(latitude, longitude)
        action_change_heading = action_value[0].item() * 10  # 许怀阳 202005062308，由90改为20
        reward = self.get_distance_reward(
            latitude, longitude, heading, action_change_heading
        )
        return reward, distance

    def _init_red_unit_list(self):
        """
        初始化红方单元列表
        """
        ret_lt = []
        aircraft_list_dic = self.redside.aircrafts
        for key in aircraft_list_dic:
            ret_lt.append(key)
        return ret_lt

    def _get_a_side_observation(self, unit_list):
        """
        获取一方的观察
        """
        obs_lt = [0.0 for x in range(0, self.state_space_dim)]
        for key in unit_list:
            aircraft_list_dic = self.redside.aircrafts
            unit = aircraft_list_dic.get(key)
            if unit:
                obs_lt[0] = unit.dLongitude
                obs_lt[1] = unit.dLatitude
                obs_lt[2] = unit.fCurrentHeading
        return obs_lt

    def _get_red_observation(self):
        """
        获取红方的观察
        """
        unit_list = self.red_unit_list
        obs_lt = self._get_a_side_observation(unit_list)
        return obs_lt

    def _get_new_waypoint(self, heading, lat, lon, distance=20.0):
        """
        根據朝向，設置飛機的下一個路徑點
        """
        dic = get_point_with_point_bearing_distance(lat, lon, heading, distance)
        return dic

    def _deal_point_data(self, waypoint):
        """
        处理航路店数据
        """
        lon = str(waypoint["longitude"])
        lat = str(waypoint["latitude"])
        return lon, lat

    """
    作者：刘占勇
    日期：2020.05.04
    功能：检查飞机是否存在，用于判断是否结束推演，如果飞机没有了，就不用再推演了
    参数：无
    返回：无
    """

    def _get_aircraft_waypoint(self, action_value):
        """
        根据智能体的动作指令，获取飞机的期望的航路点
        """
        obs = self.observation
        longitude = obs[0]  # 当前的位置
        latitude = obs[1]
        heading = obs[2]  # 朝向
        waypoint_heading = _get_waypoint_heading(
            heading, action_value[0] * 10
        )  # 许怀阳 20200505 2306 由90改为20
        waypoint = self._get_new_waypoint(waypoint_heading, latitude, longitude)

        return waypoint

    def check_done(self):
        """
        检查是否可以结束
        """
        if not self._check_aircraft_exist():
            return True
        if not self._check_target_exist():
            return True
        return False

    def _check_aircraft_exist(self):
        """
        作者：刘占勇
        日期：2020.05.04
        功能：检查飞机是否存在，用于判断是否结束推演，如果飞机没有了，就不用再推演了
        """
        obs = self.observation
        for i in range(len(obs)):
            if obs[i] != 0.0:
                return True
        return False

    def _check_target_exist(self):
        """
        作者：刘占勇
        日期：2020.05.04
        功能：检查是否还有目标存在
        """
        for k, v in self.blueside.facilities.items():
            if "T-72" in v.strName:
                if float(v.strDamageState) >= 10:
                    return False

        ret = self.scenario.get_units_by_name(etc.target_name)
        for key in ret:
            ret = self.scenario.unit_is_alive(key)
            if not ret:
                # pylog.info("target is not exist")
                pass
            else:
                # pylog.info("target is exist")
                pass
            return ret
        return False

    def _get_target_guid(self):
        """
        获取目标guid
        """
        target_name = etc.target_name
        for key in self.blueside.facilities:
            pylog.info("%s" % self.blueside.facilities[key])
            if etc.target_name == self.blueside.facilities[key].strName:
                target_guid = key
                return target_guid
        return None

    def _get_contact_target_guid(self):
        target_name = etc.target_name
        if self.redside.contacts:
            for key in self.redside.contacts:
                pylog.info("contact guid:%s" % key)
                dic = self.redside.contacts[key].__dict__
                actual_guid = self.redside.contacts[key].m_ActualUnit
                if etc.target_name == self.blueside.facilities[actual_guid].strName:
                    return key

    def _check_is_contact_target(self):
        """
        作者：刘占勇
        日期：2020.05.04
        功能：检查是否还有目标存在
        """
        target_name = etc.target_name
        if self.redside.contacts:
            for key in self.redside.contacts:
                dic = self.redside.contacts[key].__dict__
                actual_guid = self.redside.contacts[key].m_ActualUnit
                for k in self.blueside.facilities:
                    if etc.target_name == self.blueside.facilities[k].strName:
                        target_guid = k
                        return target_guid
        return False

    def _check_is_find_target(self):
        """
        检查是否发现目标
        """
        target_name = etc.target_name
        target_guid = self._check_is_contact_target()
        if target_guid:
            pylog.info("find target and the guid is:%s" % target_guid)
            return True

        return False

    def _update(self):
        """
        更新
        """
        self.mozi_server.update_situation(self.scenario, self.app_mode)
        self.redside.static_update()
        self.blueside.static_update()

    def get_observation(self):
        """
        获取观察
        """
        red_obs_lt = self._get_red_observation()
        self.observation = red_obs_lt
        return red_obs_lt

    def _construct_side_entity(self):
        """
        构造各方实体
        """
        self.redside = self.scenario.get_side_by_name(self.red_side_name)
        self.redside.static_construct()
        self.blueside = self.scenario.get_side_by_name(self.blue_side_name)
        self.blueside.static_construct()

    def _init_unit_list(self):
        """
        初始化单元列表
        """
        self.red_unit_list = self._init_red_unit_list()

    def _get_timesteps(self, action):
        """
        获取单步数据
        """
        obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.check_done()
        info = ""
        return np.array(obs), reward, done, info

    def get_target_point(self):
        """
        获取目标点
        """
        lat2 = etc.task_end_point["latitude"]
        lon2 = etc.task_end_point["longitude"]
        return lat2, lon2

    def get_target_distance(self, lat, lon):
        """
        获取目标距离
        """
        lat2, lon2 = self.get_target_point()
        distance = get_two_point_distance(lon, lat, lon2, lat2)
        return distance

    def get_reward_value(self, task_heading, current_heading, distance):
        """
        获取奖励值
        """
        angel = abs(task_heading - current_heading)
        cos_value = cos(radians(angel))
        reward = (cos_value + 1.0) ** 2 * 1e4 / distance
        if reward < 0:
            reward = 2 * reward
        return reward

    def get_distance_reward(self, lat, lon, last_heading, heading_change):
        """
        获取距离奖励值
        """
        lat2, lon2 = self.get_target_point()
        distance = get_two_point_distance(lon, lat, lon2, lat2)
        task_heading = get_degree(lat, lon, lat2, lon2)
        current_heading = last_heading + heading_change
        return self.get_reward_value(task_heading, current_heading, distance)
