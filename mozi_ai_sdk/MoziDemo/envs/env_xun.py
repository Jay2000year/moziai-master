#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:33:22 2020

@author: dixit
"""

import random
import itertools
import uuid
from collections import namedtuple
from itertools import chain
from mozi_simu_sdk.mssnpatrol import CPatrolMission
from mozi_simu_sdk.mssnstrike import CStrikeMission
# from mozi_ai_sdk.test.dppo.envs.common.utils import *
from envs.common.utils import *
from MoziDemo.envs.env import Environment
from MoziDemo.envs import etc

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box, Dict
from mozi_ai_sdk.remote_handle_docker import restart_mozi_container

import sys
import re
import zmq
import time

# zmq init
zmq_context = zmq.Context()
# ray request port
restart_requestor = zmq_context.socket(zmq.REQ)
Function = namedtuple('Function', ['type', 'function'])
FEATS_MAX_LEN = 350
MAX_DOCKER_RETRIES = 3

class RLENV(MultiAgentEnv):
    def __init__(self, env_config):
        self.steps = None
        self.reward_accum = None
        self.env_config = env_config
        self.reset_nums = 0
        self.side_name = env_config['side_name']
        self.key_word = '地空导弹中队'
        self._init_env()
        
    def get_rl_reset_info(self, agent_name):
        obs, r, done, info = None, None, None, None
        if agent_name == 'auto':
            pass
        else:
            obs = self._get_features()
            done = False
            r = 0
            info = 0
        return obs, r, done, info
        
    def reset(self, agent_name):
        self._init_env()
        return self.get_rl_reset_info(agent_name)
        
    def auto_step(self):
        done = False
        self._get_features()
        self.scenario = self.env.step()  # 墨子环境step
        self.side = self.scenario.get_side_by_name(self.side_name)
        done = self._is_done()
        return done
    
    def _return_weapon_list(self, weaponsvalid):
        name, num, dbid = list(), list(), list()
        for item in weaponsvalid:
            dbid.append(item[1])
            name.append(item[0].split(' ')[1])
            num.append(int(item[0].split(' ')[0][0:-1]))
        return name, num, dbid
    
    def _turn_distance_to_float(self, num):
        res = -1
        try:
            res = float(num)
        except:
            pass
        return res
            

    def _get_features(self):
        obs = {}
        contacts = self.side.get_contacts()
        if len(contacts)>0:
            facilities = self.side.get_facilities()
            can_fire_key = None
            for key, facility in facilities.items():
                facility_info = facility.get_summary_info()
                if self.key_word in facility_info['name']:
                    can_fire_key = key
                    self.can_fire_facilities[key] = {}
                    self.can_fire_facilities[key]['longitude'] = facility_info['longitude']
                    self.can_fire_facilities[key]['latitude'] = facility_info['latitude']
                    self.can_fire_facilities[key]['altitude'] = facility_info['altitude']
                    self.can_fire_facilities[key]['weaponsValid'] = {}
                    self.can_fire_facilities[key]['weaponsValid']['name'], self.can_fire_facilities[key]['weaponsValid']['num'], self.can_fire_facilities[key]['weaponsValid']['dbid'] = self._return_weapon_list(facility_info['weaponsValid'])
            
            for key, contact in contacts.items():
                contact_info = contact.get_contact_info()
                obs[key] = {}
                obs[key]['longitude'] = contact_info['longitude']
                obs[key]['latitude'] = contact_info['latitude']
                obs[key]['altitude'] = contact_info['altitude']
                obs[key]['heading'] = contact_info['heading']
                obs[key]['speed'] = contact_info['speed']
                obs[key]['fg'] = contact_info['fg']
                obs[key]['guid'] = contact_info['guid']
                ###################################################
                obs[key]['facility_info'] = {}
                obs[key]['facility_info']['distance'] = self._turn_distance_to_float(facilities[can_fire_key].get_range_to_contact(contact_info['guid']))
                obs[key]['facility_info']['longitude'] = self.can_fire_facilities[can_fire_key]['longitude']
                obs[key]['facility_info']['latitude'] = self.can_fire_facilities[can_fire_key]['latitude']
                obs[key]['facility_info']['altitude'] = self.can_fire_facilities[can_fire_key]['altitude']
                obs[key]['facility_info']['weaponsValid'] = {}
                obs[key]['facility_info']['weaponsValid'] = self.can_fire_facilities[can_fire_key]['weaponsValid']
                ##################################################
                obs[key]['weapon_info'] = {}
                obs[key]['weapon_info']['name'] = list()
                
            
            weapons = self.side.get_weapons()
            if len(weapons) > 0:
                for key, weapon in weapons.items():
                    #print(key)
                    try:
                        weapon_info = weapon.get_summary_info()
                        #print(weapon_info['target'], obs[key], obs[key]['fg'])
                        if weapon_info['target'] in obs.keys():
                            obs[key]['weapon_info']['name'].append(weapon_info['name'])
                    except:
                        #pass
                        print('[ERROR]: weapon.get_summary_info()')              
        else:
            pass
        return obs
    
    def _action_mapping(self, idx):
        action_mapping = {
            0: [0, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, 2],
            4: [2, 0],
            5: [1, 1],
            6: [3, 0],
            7: [0, 3],
            8: [1, 2],
            9: [2, 1],
            10: [4, 0],
            11: [0, 4],
            12: [2, 2],
            13: [1, 3],
            14: [3, 1]
        }
        return action_mapping[idx]   
    
    def _get_can_fire_facility_at_fire(self):
        facilities = self.side.get_facilities()
        can_fire_facility = None
        patriot_2_weapon_dbid, patriot_3_weapon_dbid = "null", "null"
        for _, facility in facilities.items():
            facility_info = facility.get_summary_info()
            if self.key_word in facility_info['name']:
                can_fire_facility = facility
                for item in facility_info['weaponsValid']:
                    if '爱国者-2' in item[0]:
                        patriot_2_weapon_dbid = item[1]
                    if '爱国者-3' in item[0]:
                        patriot_3_weapon_dbid = item[1]  
        return can_fire_facility, patriot_2_weapon_dbid, patriot_3_weapon_dbid
                        
    
    def rl_step(self, action_record):
        #############################################################       
        for fg, action in action_record.items():
            can_fire_facility, patriot_2_weapon_dbid, patriot_3_weapon_dbid = self._get_can_fire_facility_at_fire()
            patriot_2_num, patriot_3_num = self._action_mapping(action)
            #print(patriot_3_weapon_dbid, patriot_3_num, patriot_2_weapon_dbid, patriot_2_num)
            if patriot_2_weapon_dbid == "null":
                pass
            else:
                #print('fire 2 -> ', patriot_2_num, self._action_mapping(action))
                can_fire_facility.manual_attack(fg, patriot_2_weapon_dbid, patriot_2_num)
            if patriot_3_weapon_dbid == "null":
                pass
            else:
                #print('fire 3 -> ', patriot_3_num, self._action_mapping(action))
                can_fire_facility.manual_attack(fg, patriot_3_weapon_dbid, patriot_3_num)
        #############################################################
        self.scenario = self.env.step()  # 墨子环境step
        self.side = self.scenario.get_side_by_name(self.side_name)
        #############################################################
        obs = self._get_features()
        ############################################################
        done = self._is_done()
        r, info = 0, 0
        return obs, done, r, info 
    
        
    def _init_env(self):
        self._create_env(etc.PLATFORM, scenario_name=etc.SCENARIO_NAME)
        self.scenario = self.env.reset(self.side_name)
        self.side = self.scenario.get_side_by_name(self.side_name)
        self.can_fire_facilities = {}
        self.reset_nums += 1
    
    def _create_env(self, platform, scenario_name=None):
        """
        每5局重启墨子，获取初始态势
        """
        docker_ip_port = '123.57.142.120:6063'
        for _ in range(MAX_DOCKER_RETRIES):
            try:
                if self.reset_nums % 5 == 0:
                    print('[RESTART] Restart Container')
                    restart_mozi_container(docker_ip_port)
                self.env = Environment(etc.SERVER_IP,
                                      etc.SERVER_PORT,
                                      platform,
                                      scenario_name,
                                      etc.SIMULATE_COMPRESSION,
                                      etc.DURATION_INTERVAL,
                                      etc.SYNCHRONOUS)
                
                if self.env_config['avail_docker_ip_port']:
                    self.avail_ip_port_list = self.env_config['avail_docker_ip_port']
                else:
                    raise Exception('no avail port!')
                self.ip_port = self.avail_ip_port_list[0]
                self.ip = self.avail_ip_port_list[0].split(":")[0]
                self.port = self.avail_ip_port_list[0].split(":")[1]
                self.ip_port = f'{self.ip}:{self.port}'            
                self.env.start(self.ip, self.port)
                break
            except:
                print("第%d次重启docker失败！！！"%(_))
        
    def set_doctrine(self, name):
        command = "2"
        if name == 'auto' or name == 'debug':
            command = "0"
        doctrine = self.side.get_doctrine()
        doctrine.set_weapon_control_status_air(command)
        if command == "0":
            print('[COMMAND]: 自由开火')
        else:
            print('[COMMAND]: Agent控制')
            
    def _is_done(self):
        # 对战平台
        response_dic = self.scenario.get_responses()
        for _, v in response_dic.items():
            if v.Type == 'EndOfDeduction':
                print('打印出标记：EndOfDeduction')
                return True
        return False

    '''
    def _update(self, scenario):
        self.side = scenario.get_side_by_name(self.side_name)
        self.reward = self._get_win_score() - self.reward_accum + self.temp_reward
        self.reward_accum = self._get_win_score() + self.temp_reward
        self.temp_reward = 0
        self.m_Time = self.scenario.m_Time  # 想定当前时间
        self.asuw = {k: v for k, v in self.side.aircrafts.items()
                     if int(re.sub('\D', '', v.strLoadoutDBGUID)) == 3004}  # 可用反舰空战飞机
        self.asup = {k: v for k, v in self.side.aircrafts.items()
                     if int(re.sub('\D', '', v.strLoadoutDBGUID)) == 19361}  # 可用空战飞机
        self.target = {k: v for k, v in self.side.contacts.items() if v.m_ContactType == 2 and 'DDG' in v.strName}



    def step(self, action):
        done = False
        mission_unit = self._assign_available_unit(action['agent_0'])
        if self.env_config['mode'] in ['train', 'development']: 
            force_done = self.safe_step(action, mission_unit)
            if force_done:
                done = force_done
                self.reset_nums = 4  # 下一局会重启墨子docker(每5局重启一次docker)
                print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，强制重启墨子！！！")
            else:
                self._update(self.scenario)
                done = self._is_done()
        elif self.env_config['mode'] in ['versus', 'eval']:
            if mission_unit:
                self._action_func_list[action['agent_0']].function(self.side, mission_unit)
            self.scenario = self.env.step()  # 墨子环境step
            self._update(self.scenario)
            done = self._is_done()
        reward = {'agent_0': self.reward}
        obs = {'agent_0': {"obs": self._generate_features()}}
        self.steps += 1
        if self.steps % 10 == 0:
            print(self.ip_port + '-' + f'reward is {self.reward}' + '-' + f'action is {action}')
        if done:
            print('++++Score:', self.reward_accum, 'step:', self.steps)
        return obs, reward, {'__all__': done, 'agent_0': done}, {'agent_0': {'score': self.side.iTotalScore}}

    def safe_step(self, action, mission_unit):
        force_done = False
        # noinspection PyBroadException
        try:
            if mission_unit:
                self._action_func_list[action['agent_0']].function(self.side, mission_unit)
        except Exception:
            print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，执行lua超时！！！")
            force_done = True
            return force_done
        # noinspection PyBroadException
        try:
            self.scenario = self.env.step()  # 墨子环境step
        except Exception:
            print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，更新态势超时！！！")
            force_done = True
            return force_done
        if self.scenario and self.scenario.get_side_by_name(self.side_name):
            return force_done
        else:
            # 态势更新失败会抛出异常
            print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，更新态势失败！！！")
            force_done = True
            return force_done                       

    def reset(self):
        self._get_initial_state()
        self.steps = 0
        self.reward_accum = self._get_win_score()
        self._update(self.scenario)
        obs = {'agent_0': {
            "obs": self._generate_features()
        }}
        print('env_reset finished!!!')
        return obs

    def _generate_features(self):
        feats = []

        contacts = {k: v for k, v in self.side.contacts.items() if v.m_ContactType != 1}
        # s_contacts = sorted(contacts.items(), key=lambda value: value[1].dLongitude)
        h_feats = [0.0 for _ in range(6)]
        div = 0.0
        for k, v in contacts.items():
            div += 1.0
            temp_feats = [0.0 for _ in range(6)]
            if v.m_ContactType:
                temp_feats[0] = v.m_ContactType/22.0
            if v.m_IdentificationStatus:
                temp_feats[1] = v.m_IdentificationStatus/4.0
            if v.fCurrentHeading:
                temp_feats[2] = v.fCurrentHeading/180.0
            if v.fCurrentSpeed:
                temp_feats[3] = v.fCurrentSpeed/1000.0
            if v.dLongitude and v.dLatitude:
                temp_feats[4] = v.dLongitude/180.0
                temp_feats[5] = v.dLatitude/180.0
            h_feats = map(lambda x, y: x + y, h_feats, temp_feats)
        if div == 0.0:
            feats.extend(h_feats)
        else:
            h_feats = [i/div for i in h_feats]
            feats.extend(h_feats)

        # aircraft = sorted(self.side.aircrafts.items(), key=lambda value: value[1].dLongitude)
        red_air_feats = [0.0 for _ in range(9)]
        div = 0.0
        for k, v in self.side.aircrafts.items():
            div += 1.0
            temp_red_air_feats = [0.0 for _ in range(9)]
            if v.iFireIntensityLevel:
                temp_red_air_feats[0] = v.iFireIntensityLevel/4.0
            if v.iFloodingIntensityLevel:
                temp_red_air_feats[1] = v.iFloodingIntensityLevel/4.0
            if v.strAirOpsConditionString:
                temp_red_air_feats[2] = v.strAirOpsConditionString/26.0
            if v.dLongitude and v.dLatitude:
                temp_red_air_feats[3] = v.dLongitude/180.0
                temp_red_air_feats[4] = v.dLatitude/180.0

            weapon_list = self._get_unit_weapon(v)
            # 诱饵弹 2051-通用红外干扰弹；564-通用箔条；3386-AN/ALE70
            temp_red_air_feats[5] = self._get_weapon_num(weapon_list, [564, 2051, 3386])/10.0
            # 空空导弹  51-AIM120D;945-AIM9X
            temp_red_air_feats[6] = self._get_weapon_num(weapon_list, [51, 945])/10.0
            # 反舰导弹  826-AGM154C
            temp_red_air_feats[7] = self._get_weapon_num(weapon_list, [826, ])/10.0
            # 防空导弹 15-RIM162A
            temp_red_air_feats[8] = self._get_weapon_num(weapon_list, [15, ])/10.0

            red_air_feats = map(lambda x, y: x + y, red_air_feats, temp_red_air_feats)

        if div == 0.0:
            feats.extend(red_air_feats)
        else:
            red_air_feats = [i/div for i in red_air_feats]
            feats.extend(red_air_feats)

        # ships = sorted(self.side.ships.items(), key=lambda value: value[1].dLongitude)
        red_ship_feats = [0.0 for _ in range(3)]
        div = 0.0
        for k, v in self.side.ships.items():
            div += 1.0
            temp_red_ship_feats = [0.0 for _ in range(3)]
            if v.dFuelPercentage:
                temp_red_ship_feats[0] = v.dFuelPercentage/100.0
            if v.dLongitude and v.dLatitude:
                temp_red_ship_feats[1] = v.dLatitude/180.0
                temp_red_ship_feats[2] = v.dLongitude/180.0

            red_ship_feats = map(lambda x, y: x + y, red_ship_feats, temp_red_ship_feats)

        if div == 0.0:
            feats.extend(red_ship_feats)
        else:
            red_ship_feats = [i/div for i in red_ship_feats]
            feats.extend(red_ship_feats)

        red_patrol_mission_feats = [self.side.patrolmssns.__len__()/10.0, ]
        feats.extend(red_patrol_mission_feats)
        red_strike_mission_feats = [self.side.strikemssns.__len__()/10.0, ]
        feats.extend(red_strike_mission_feats)

        time_delta = self.m_Time - self.m_StartTime
        feats.append(time_delta / 3600.0)
        feats.append(time_delta / 7200.0)
        feats.append(time_delta / 14400.0)
        # print(f'+++feats:{feats}')
        # if feats.__len__() > FEATS_MAX_LEN:
        #     feats = feats[:FEATS_MAX_LEN]
        # else:
        #     feats.extend([0.0 for _ in range(FEATS_MAX_LEN - feats.__len__())])
        return feats

    @staticmethod
    def _get_unit_weapon(unit):
        """
        :param unit: aircraft, ship
        :return:
        """
        weapon = list(map(lambda x: x.split('$'), unit.m_UnitWeapons.split('@')))
        weapon_list = list(map(lambda x, y: x + [y[-1]], list(map(lambda x: x[0].split('x '), weapon)), weapon))
        return weapon_list

    @staticmethod
    def _get_weapon_num(weapon_list, weapon_type):
        num = 0
        for weapon in weapon_list:
            if weapon[0] != '' and weapon[-1] != '':
                if int(re.sub('\D', '', weapon[-1])) in weapon_type:
                    num += int(weapon[0])
        return num

    def _get_env(self):
        if self.env_config['mode'] == 'train':
            self.schedule_addr = self.env_config['schedule_addr']
            self.schedule_port = self.env_config['schedule_port']
            scenario_name = etc.SCENARIO_NAME
            platform = 'linux'
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config['mode'] == 'development':
            scenario_name = etc.SCENARIO_NAME
            platform = 'linux'
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config['mode'] == 'versus':
            scenario_name = etc.SCENARIO_NAME
            platform = 'linux'
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config['mode'] == 'eval':
            scenario_name = etc.EVAL_SCENARIO_NAME
            platform = 'windows'
            self._create_env(platform, scenario_name=scenario_name)

            # platform = 'linux'
            # self._create_env(platform)
        else:
            raise NotImplementedError

    def _create_env(self, platform, scenario_name=None):
        for _ in range(MAX_DOCKER_RETRIES):
            # noinspection PyBroadException
            try:
                self.env = Environment(etc.SERVER_IP,
                                       etc.SERVER_PORT,
                                       platform,
                                       scenario_name,
                                       etc.SIMULATE_COMPRESSION,
                                       etc.DURATION_INTERVAL,
                                       etc.SYNCHRONOUS)
                # by dixit
                if self.env_config['avail_docker_ip_port']:
                    self.avail_ip_port_list = self.env_config['avail_docker_ip_port']
                else:
                    raise Exception('no avail port!')
                # self.self.reset_nums = 0
                self.ip_port = self.avail_ip_port_list[0]
                print(self.ip_port)
                self.ip = self.avail_ip_port_list[0].split(":")[0]
                self.port = self.avail_ip_port_list[0].split(":")[1]
                self.ip_port = f'{self.ip}:{self.port}'
                self.env.start(self.ip, self.port)
                break
            except Exception:
                continue

    def _get_initial_state(self):
        """
        dixit 2021/3/22
        每5局重启墨子，获取初始态势
        """

        self.reset_nums += 1
        if self.env_config['mode'] in ['train', 'development']:
            if self.reset_nums % 5 == 0:
                docker_ip_port = self.avail_ip_port_list[0]
                for _ in range(MAX_DOCKER_RETRIES):
                    # noinspection PyBroadException
                    try:
                        if self.env_config['mode'] == 'train':
                            restart_container(self.schedule_addr,
                                              self.schedule_port,
                                              self.env_config['training_id'],
                                              docker_ip_port)
                        else:
                            restart_mozi_container(docker_ip_port)
                        self.env = Environment(etc.SERVER_IP,
                                               etc.SERVER_PORT,
                                               'linux',
                                               etc.SCENARIO_NAME,
                                               etc.SIMULATE_COMPRESSION,
                                               etc.DURATION_INTERVAL,
                                               etc.SYNCHRONOUS)
                        self.env.start(self.ip, self.port)
                        break
                    except Exception:
                        print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，第{_}次重启docker失败！！！")
                        continue
                print('开始mozi reset!!!')
                self.scenario = self.env.reset(self.side_name)
                print('结束mozi reset!!!')
            else:
                print('开始mozi reset!!!')
                self.scenario = self.env.reset(self.side_name)
                print('结束mozi reset!!!')
        else:
            self.scenario = self.env.reset(self.side_name)

    def _is_done(self):
        # 对战平台
        response_dic = self.scenario.get_responses()
        for _, v in response_dic.items():
            if v.Type == 'EndOfDeduction':
                print('打印出标记：EndOfDeduction')
                return True
        return False
'''