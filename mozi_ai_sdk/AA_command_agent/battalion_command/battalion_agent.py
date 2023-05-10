# -*- coding:utf-8 -*-
# File name : battalion_agent.py
# Create date : 2022/8/30
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.0

import re
import pyproj
import pandas as pd
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


def get_distance(tgt, unit):
    """
    返回tgt与unit之间的距离，单位为米
    """
    return earth_model.inv(
        tgt.dLongitude, tgt.dLatitude, unit.dLongitude, unit.dLatitude
    )[2]


def get_azimuth(tgt, unit):
    """
    返回tgt指向unit的指向角，正北为0
    """
    azimuth = earth_model.inv(
        tgt.dLongitude, tgt.dLatitude, unit.dLongitude, unit.dLatitude
    )[0]
    return azimuth if azimuth >= 0 else azimuth + 360.0


# 营级规则体类，对应想定中单个地导营的操作智能体
class BattalionCommand:
    def __init__(self, name, activeunit):
        # 这些属性为固定属性，不随态势或自身状态更新而更新
        self.name = name
        self.superior_command = None  # 上级规则体
        self.activeunit_obj = activeunit  # 规则体所属活动单元
        self.strName = activeunit.strName  # 显示名称
        self.guid = activeunit.strGuid  # guid
        self.fire_control_channel = None  # 火控通道数量
        self.primary_munition_velocity = None  # 主要弹种的飞行速度
        # self.reload_time_cost = None  # 重装填耗时

        # 这些属性是动态属性，在特定情况或条件下更新
        self.emcon_state = ""  # 电磁管控
        self.combat_mode = ""  # 交战模式
        # self.move_dest = []  # 机动航路点
        self.attacking_dict = {}  # 正在攻击的目标清单 目标m_ActualUnit：弹药种类@分配数
        self.allocated_wpn_dict = {}  # 已分配弹药的目标字典 目标m_ActualUnit：弹药种类@分配数@已分配时间
        self.unable_atk_list = []  # 无法攻击的目标清单 目标m_ActualUnit
        self.epd = None  # 交战优先级条令
        self.idle_fcc = self.fire_control_channel  # 空闲火控通道数
        self.munition_remain = {}  # 剩余弹药
        self.munition_ready = {}  # 准备就绪的弹药
        self.munition_reloading = {}  # 装填中的弹药
        self.damage_state = activeunit.strDamageState  # 毁伤
        self.local_situation = None  # 营级局部态势
        self.target_list = []  # 目标清单
        self.target_priority = {}  # 目标优先级字典
        self.protect_tgt = []  # 保卫目标清单
        self.target_allocated = []  # 已分配营进行拦截的目标
        self.target_not_allocated = []  # 未分配营进行拦截的目标

        # 正在攻击的目标
        # self.activeunit_attack = []
        # self.ooda_time = {}  # ooda时间字典

        self.set_parameter()

    def set_parameter(self):
        """
        设定营属性参数
        """
        if self.activeunit_obj.strDBGUID == "hsfw-datafacility-000001003395":
            self.fire_control_channel = FireControlChannels["S-400_40N6"]
            self.epd = EPD["S-400_40N6"]
            self.primary_munition_velocity = MunitionVelocity["40N6"]
        # elif self.activeunit_obj.strDBGUID == 'hsfw-datafacility-000000001668':
        #     self.fire_control_channel = FireControlChannels['S-400_48N6DM']
        #     self.epd = EPD['S-400_48N6DM']

    def set_combat_mode(self, mode):
        """
        设置营交战模式
        """
        if mode == CombatMode.AMBUSH:
            self.combat_mode = CombatMode.AMBUSH
        elif mode == CombatMode.GENERAL:
            self.combat_mode = CombatMode.GENERAL
        elif mode == CombatMode.TGTPROTECT:
            self.combat_mode = CombatMode.TGTPROTECT

    def set_EMCON(self, emcon):
        """
        设置营电磁管控状态
        """
        if emcon == EMCONMode.RADARONLY:
            self.emcon_state = EMCONMode.RADARONLY
            self.activeunit_obj.set_radar_shutdown(True)
            self.activeunit_obj.set_oecm_shutdown(False)
        elif emcon == EMCONMode.OECMONLY:
            self.emcon_state = EMCONMode.OECMONLY
            self.activeunit_obj.set_radar_shutdown(False)
            self.activeunit_obj.set_oecm_shutdown(True)
        elif emcon == EMCONMode.BOTH:
            self.emcon_state = EMCONMode.BOTH
            self.activeunit_obj.set_radar_shutdown(True)
            self.activeunit_obj.set_oecm_shutdown(True)
        elif emcon == EMCONMode.SILENCE:
            self.emcon_state = EMCONMode.SILENCE
            self.activeunit_obj.set_radar_shutdown(False)
            self.activeunit_obj.set_oecm_shutdown(False)

    def self_update(self):
        """
        更新自身状态
        """
        if not self.activeunit_obj:
            self.damage_state = 100
            return "unit_lost"
        else:
            self.damage_state = self.activeunit_obj.strDamageState

        # 记录总剩余弹药
        def munition_matcher(string):
            pattern = re.compile(r"(\d+)x.+(\$hsfw-dataweapon-\d+)")
            return {pattern.match(string).group(2): pattern.match(string).group(1)}

        self.munition_remain.clear()
        if not self.activeunit_obj.m_UnitWeapons:
            pass
        else:
            unitweapons = self.activeunit_obj.m_UnitWeapons.split("@")
            for wp in unitweapons:
                self.munition_remain.update(munition_matcher(wp))

        # 记录装填中的弹药与就绪弹药
        mounts = self.activeunit_obj.get_mounts()
        self.munition_reloading.clear()
        self.munition_ready.clear()
        for munition_dbid in self.munition_remain.keys():
            self.munition_reloading.update({get_munition_name(munition_dbid): []})
            self.munition_ready.update({get_munition_name(munition_dbid): []})
            for mount in mounts.values():
                if munition_dbid in mount.m_LoadRatio:
                    if "重装载" in mount.strWeaponFireState:
                        self.munition_reloading[
                            get_munition_name(munition_dbid)
                        ].append(mount.strWeaponFireState)
                    elif "就绪" in mount.strWeaponFireState:
                        self.munition_ready[get_munition_name(munition_dbid)].append(
                            mount.strLoadWeaponCount
                        )

        # 筛选出属于该地导营的在空导弹
        def unable_to_attack(weapon_time):
            lst = weapon_time.split('@')
            time_update = int(lst[2]) + 5     # 5秒为推演步长
            return lst[0] + '@' + lst[1] + "@" + str(time_update)

        def judge(weapon_time):
            lst = weapon_time.split('@')
            if int(lst[1])>=15:
                return True
            else:
                return False

        # 从当前目标中删除已攻击目标
        self.weapon_in_the_air = [x.m_PrimaryTargetGuid for k, x in self.local_situation.weapons.items()]
        if self.allocated_wpn_dict and self.weapon_in_the_air:
            dic = {}
            for k, x in self.allocated_wpn_dict.items():
                for weapon in self.weapon_in_the_air:
                    if weapon in k:
                        dic[k] = unable_to_attack(x)
            for k in dic.keys():
                del self.allocated_wpn_dict[k]
        else:
            self.allocated_wpn_dict = {
                k: self.unable_to_attack(x) for k, x in self.allocated_wpn_dict.items()
            }
        self.unable_atk_list = [k for k, x in self.allocated_wpn_dict.items() if judge(x)]  # 反馈列表   不能攻击的目标的真实guid

        # 更新正在攻击的目标清单
        self.attacking_list_dict = {k + x.m_ActualUnit: x for k, x in self.local_situation.contacts.items()}
        if [x for k, x in self.local_situation.weapons.items()]:
            for guid, weapon in self.local_situation.weapons.items():
                if weapon.m_FiringUnitGuid == self.guid:
                    attacked = []
                    for k, x in self.attacking_list_dict.items():
                        if weapon.m_PrimaryTargetGuid in k:
                            attacked.append(x.m_ActualUnit)
                    self.attacking_list = self.attacking_list + attacked
        else:
            self.attacking_list = [re.findall(r"(.*)@", k)[0] for k, x in self.allocated_wpn_dict.items()]
        self.attacking_list = list(set(self.attacking_list))

        # 更新营级目标清单
        for guid, contact in self.local_situation.contacts.items():
            if contact.m_ContactType == 0:
                self.target_not_allocated.append(contact)
            elif contact.m_ContactType == 1 and 'VAMPIRE' in contact.strName:
                self.target_not_allocated.append(contact)
            elif contact.m_ContactType == 1 and 'GuidedWeapon' in contact.strName:
                self.target_not_allocated.append(contact)
        self.target_not_allocated = [x for x in self.target_not_allocated if x.m_ActualUnit not in self.attacking_list]
        self.target_allocated = self.target_allocated + [x for x in self.target_not_allocated if x.m_ActualUnit in self.attacking_list]
        self.target_list = self.target_allocated
        # 更新空闲火控通道数量
        self.idle_fcc = self.fire_control_channel - len(self.attacking_list)

    def EngagePriorityAlgorithm(self):
        """
        根据交战优先级条令为目标标注交战优先级
        """
        self.target_priority = {k: 0 for k in self.target_list}
        for tgt in self.target_list:
            if tgt.m_ContactType == 0:
                self.target_priority[tgt] += self.epd["Aircraft"]
            elif tgt.m_ContactType == 1:
                self.target_priority[tgt] += self.epd["Missile"]
        for tgt in self.target_list:
            if tgt.fAltitude_AGL > self.epd["HighAltThreshold"]:
                self.target_priority[tgt] += self.epd["HighAlt"]
            elif tgt.fAltitude_AGL < self.epd["LowAltThreshold"]:
                self.target_priority[tgt] += self.epd["LowAlt"]
            else:
                self.target_priority[tgt] += self.epd["MidAlt"]
        for tgt in self.target_list:
            if tgt.fCurrentSpeed * 1.852 > self.epd["HighSpdThreshold"]:
                self.target_priority[tgt] += self.epd["HighSpd"]
            elif tgt.fCurrentSpeed * 1.852 < self.epd["LowSpdThreshold"]:
                self.target_priority[tgt] += self.epd["LowSpd"]
            else:
                self.target_priority[tgt] += self.epd["MidSpd"]
        for tgt in self.target_list:
            if not tgt.strAge:
                strAge = 0
            elif "分" in tgt.strAge:
                strAge = -1
            else:
                strAge = int(tgt.strAge.split("秒")[0])

            if strAge == -1 or strAge >= 20:
                self.target_priority[tgt] += self.epd["UnknowRange"]
            elif (
                get_distance(tgt, self.activeunit_obj) >= self.epd["FarRangeThreshold"]
            ):
                self.target_priority[tgt] += self.epd["FarRange"]
            elif (
                get_distance(tgt, self.activeunit_obj)
                <= self.epd["CloseRangeThreshold"]
            ):
                self.target_priority[tgt] += self.epd["CloseRange"]
            else:
                self.target_priority[tgt] += self.epd["MidRange"]
        for tgt in self.target_list:
            if self.protect_tgt:
                for protect_tgt in self.protect_tgt:
                    if (
                        tgt.fCurrentHeading + 0.2
                        > get_azimuth(tgt, protect_tgt)
                        > tgt.fCurrentHeading - 0.2
                    ):
                        self.target_priority[tgt] += self.epd["HeadingProtect"]
                        break
                    if (
                        tgt.fCurrentHeading + 0.2
                        > get_azimuth(tgt, self.activeunit_obj)
                        > tgt.fCurrentHeading - 0.2
                    ):
                        self.target_priority[tgt] += self.epd["HeadingSelf"]
            elif (
                tgt.fCurrentHeading + 0.2
                > get_azimuth(tgt, self.activeunit_obj)
                > tgt.fCurrentHeading - 0.2
            ):
                self.target_priority[tgt] += self.epd["HeadingSelf"]
            else:
                self.target_priority[tgt] += self.epd["OtherHeading"]

    def fire_control(self):
        """
        火力控制，按交战模式和交战优先级分配火力
        """
        if self.combat_mode == CombatMode.GENERAL:
            tgt_df = pd.DataFrame(
                {
                    "contact": list(self.target_priority.keys()),
                    "priority": list(self.target_priority.values()),
                }
            )
            tgt_df = tgt_df.sort_values(by="priority", ascending=False)
            # tgt_df.drop(index=tgt_df[tgt_df['priority'] < 20].index, inplace=True)      # 在营级端直接删除不打击的目标
            # weapon_dbguid = 'hsfw-dataweapon-00000000002104'
            # for k, contact in enumerate(tgt_df['contact']):
            #     self.activeunit_obj.manual_attack(contact.strGuid, weapon_dbguid, 1)
            #     self.attacking_list.append(contact.m_ActualUnit)
            #     # self.distribute_contact[contact.m_ActualUnit] = weapon_dbguid + '@0'
            #     sign = contact.m_ActualUnit + '@' + contact.strGuid
            #     self.distribute_contact[sign] = weapon_dbguid + '@0'
            #     # self.munition_ready.popitem()
            for row in tgt_df.iterrows():
                #
                allocate_num = 1
                self.activeunit_obj.manual_attack(
                    row["contact"].strGuid, get_munition_dbguid("40N6"), allocate_num
                )
                self.allocated_wpn_dict.update(
                    {row["contact"].m_ActualUnit + '@' + row['contact'].strGuid: "40N6@" + str(allocate_num) + "@0"}
                )

        # todo:按交战模式和交战优先级分配火力



    def status_report(self):
        """
        向上级回报自身状态
        """
        return self.superior_command.subordinate_data_receive.update(
            {
                self.name: {
                    "munition_ready": self.munition_ready,
                    "munition_reloading": self.munition_reloading,
                    "attacking_list": self.attacking_list,
                    "unable_atk_list": self.unable_atk_list,
                    "damage_state": self.damage_state,
                }
            }
        )

    def step(self):
        """
        随推演步进运行规则体
        """
        self.EngagePriorityAlgorithm()
        self.self_update()
        self.fire_control()
        self.status_report()
