import argparse
import collections
import sys
import gym
import pandas as pd
import json
import os
from pathlib import Path
import datetime
import time
import os
import numpy as np
import pickle
from tqdm import trange
import time
import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from gym.spaces import Discrete, Box, Dict
from ray.rllib.agents.ppo import PPOTrainer
import datetime
import time
# from ray.managers.zmq_manager import g_zmq_manager
from ray.managers.utils import *
import datetime



class UnifyTrainer:
    
    def __init__(self, N_STATES, N_ACTIONS, agent_name, env):
        if agent_name == 'ppo':
            pass
        elif agent_name == 'auto':
            # 此时用墨子默认的智能
            pass
        self.agent_name = agent_name
        self.env = env
        self.length = 650
        self.fire_record = {}
        self.N_STATES = N_STATES
        self.read_files = list()
        self.reward_idx = 4 
        self.done_idx = 5
        self.next_state_idx = 3
        self.action_idx = 7
        self.state_idx = 0
        #[s, a, action_prob, s_, r, done, key]
        self.mylog = open('C:/Users/3-5/Desktop/moziai-master/mozi_ai_sdk/MoziDemo/res' + agent_name + '_output.log', mode = 'a',encoding='utf-8')
         
    def exception_output(self):
        flag = True
        if self.reset_error:
            print('[ERROR] Reset Error', file = self.mylog, flush = True)
            print('[ERROR] Reset Error')
            flag = False
        if self.process_error: 
            print('[ERROR] Process Error', file = self.mylog, flush = True)
            print('[ERROR] Process Error')
            flag = False
        if self.log_generate_error: 
            print('[ERROR] Log Generate Error', file = self.mylog, flush = True)
            print('[ERROR] Log Generate Error')
            flag = False   
        if self.log_csv_error: 
            print('[ERROR] Log CSV Error', file = self.mylog, flush = True)
            print('[ERROR] Log CSV Error')
            flag = False  
        return flag
    

     
    def _whether_take_action(self, key, ob):
        if key in self.fire_record.keys():
            #print('len: ', len(ob['weapon_info']['name']))
            cur_distance = float(ob['facility_info']['distance'])
            if (cur_distance < 0) or (self.fire_record[key]['dis'][-1] < 0) or (self.fire_record[key]['num'] > 2) or (cur_distance > 40.0) or (self.fire_record[key]['missile_num'] > 6):
                return False
            else:
                if (cur_distance > self.fire_record[key]['dis'][-1]) and (self.fire_record[key]['num'] == 0):
                    return True
                if len(self.fire_record[key]['point']) > 0:
                    if (self.fire_record[key]['num'] <= 2) and (abs(float(self.fire_record[key]['point'][-1]) - float(ob['facility_info']['distance'])) > 4.0):
                        return True
            return False
        else:
            self.fire_record[key] = {}
            self.fire_record[key]['num'] = 0
            self.fire_record[key]['point'] = list()
            self.fire_record[key]['dis'] = list()
            self.fire_record[key]['missile_num'] = 0
            return False
        
        
    def _obs_2_s(self, ob, key):
        s = list()
        ## dif 
        s.append( round( np.exp(float(ob['latitude']) - float(ob['facility_info']['latitude'])),2) )
        s.append( round( np.exp(float(ob['longitude']) - float(ob['facility_info']['longitude'])),2) )
        s.append( round( np.log(float(ob['altitude']) - float(ob['facility_info']['altitude'])),2) )
        s.append( round( np.log(float(ob['heading'])),2) )
        s.append( round( np.log(float(ob['facility_info']['distance'])),2))
        if key in self.fire_record.keys():
            s.append(1)
        else:
            s.append(0)
        w_type = ob['facility_info']['weaponsValid']['name']
        left_num = ob['facility_info']['weaponsValid']['num']
        # 剩余弹药
        patriot_2_num, patriot_3_num = 0, 0
        for i in range(len(w_type)):
            if '爱国者-2' in w_type[i]:
                patriot_2_num = int(left_num[i])
            if '爱国者-3' in w_type[i]:
                patriot_3_num = int(left_num[i])
        s.append(patriot_2_num % 12)
        s.append(int(patriot_2_num /12 ))
        s.append(patriot_3_num % 16)
        s.append(int(patriot_3_num /16 ))
        # 当前来袭多少发
        s.append(len(self.fire_record) % 4)
        s.append(int(len(self.fire_record) / 4))
                
        return s
    
    def _recover_reward(self, memory_record):
        file = self._get_ready_file()
        try:
            red_endgame_pd = pd.read_csv('/root/MoziLog/Analysis/' + file + '/1/' +'红方(防御)_WeaponEndgame.csv')
            # endgame 只能记录result
            targetID = red_endgame_pd['TargetID'].tolist()
            weaponID  = red_endgame_pd['WeaponID'].tolist()
            results = red_endgame_pd['Result'].tolist()
            key_reward = {}
            for i in range(len(targetID)):
                if weaponID[i] == targetID[i]:
                    continue
                if targetID[i] not in key_reward.keys():
                    key_reward[targetID[i]] = {}
                    key_reward[targetID[i]]['num'] = 0
                    key_reward[targetID[i]]['result'] = 0
                if results[i] == 'KILL':
                    key_reward[targetID[i]]['result'] = 1
            red_fired_pd = pd.read_csv('/root/MoziLog/Analysis/' + file + '/1/' + '红方(防御)_WeaponFired.csv')
            targetID = red_fired_pd['TargetContactActualUnitID'].tolist()
            for i in range(len(targetID)):
                if targetID[i] in key_reward.keys():
                    key_reward[targetID[i]]['num'] = key_reward[targetID[i]]['num'] + 1
                else:
                    key_reward[targetID[i]] = {}
                    key_reward[targetID[i]]['num'] = 1
                    key_reward[targetID[i]]['result'] = 0
            ###############################################################
            for i in range(len(memory_record)):
                guid = memory_record[i][-1]
                if guid in key_reward.keys():
                    #print(' in !')
                    if key_reward[guid]['result'] == 1:
                        if key_reward[guid]['num'] != 0:
                            memory_record[i][self.reward_idx] = 3.0/key_reward[guid]['num']  # 效费比有关
                        else:
                            memory_record[i][self.reward_idx] = 0
                    else:
                        memory_record[i][self.reward_idx] = -0.5 * key_reward[guid]['num']
                else:
                    memory_record[i][self.reward_idx] = 0

                if i != len(memory_record) - 1:
                    memory_record[i][self.next_state_idx] = memory_record[i+1][self.state_idx]
                else:
                    memory_record[i][self.done_idx] = 1            
            print('[FILES]: %s'%(file), file = self.mylog, flush = True)
            print('[FILES]: %s'%(file))
            self.get_each_info(os.path.join('/root/MoziLog/Analysis', file, '1'))
        except:
            self.log_csv_error = True
        return memory_record

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
    
    
    def train_each_round(self, i_episode):
        self.reset_error, self.process_error, self.log_generate_error, self.log_csv_error = False, False, False, False
        try:
            obs, _, _, _ = self.env.reset(self.agent_name)
            self.env.set_doctrine(self.agent_name) # 设置开火模式
        except:
            self.reset_error = True
            
        try:
            done = False
            self.fire_record = {}
            memory_record = list()
            for count in trange(self.length):
                if self.agent_name == 'auto':
                    done = self.env.auto_step()
                else:
                    cur_action_record = {}
                    for key, ob in obs.items():
                        if self._whether_take_action(key, ob) == False:
                            pass
                        else:
                            s = self._obs_2_s(ob, key)
                            a, action_prob = self.agent.choose_action(s)
                            s_, r, done = [0] * self.N_STATES, 0, 0
                            memory_record.append([s, a, action_prob, s_, r, done, ob['guid']])
                            cur_action_record[ob['fg']] = a
                            if a != 0:
                                self.fire_record[key]['num'] = self.fire_record[key]['num'] + 1
                                self.fire_record[key]['point'].append(float(ob['facility_info']['distance']))
                                self.fire_record[key]['missile_num'] = self.fire_record[key]['missile_num'] + self._action_mapping(a)[0] + self._action_mapping(a)[1]
                        self.fire_record[key]['dis'].append(float(ob['facility_info']['distance']))
                    next_obs, done, r, info = self.env.rl_step(cur_action_record)
                    obs = next_obs
                if done:
                    break
        except:
            self.process_error = True
        memory_record = self._recover_reward(memory_record)
        if self.exception_output():
            if self.agent_name == 'auto':
                pass
            else:
                self.agent.store_transition(memory_record)
                print('[BUFFER_SIZE]: %d'%(len(self.agent.buffer)))
                if len(self.agent.buffer) > self.agent.batch_size and self.agent_name != 'offline':
                    self.agent.learn(self.mylog)
                self.agent.record_ep_r(i_episode, memory_record, self.mylog)
                self.agent.save_model(i_episode)
        else:
            print('[ERROR] ROUND %d MISS'%(i_episode), file = self.mylog, flush = True)
            print('[ERROR] ROUND %d MISS'%(i_episode))
        
    def get_each_info(self, path):
        red_endgame_pd = pd.read_csv(os.path.join(path, '红方(防御)_WeaponEndgame.csv'))
        red_weaponfired_pd = pd.read_csv(os.path.join(path, '红方(防御)_WeaponFired.csv'))
        blue_weaponfired_pd = pd.read_csv(os.path.join(path, '蓝方(进攻)_WeaponFired.csv'))
        # 统计爱国者2
        patriot_2_bool_index = red_endgame_pd.WeaponName.str.contains('爱国者-2')
        patriot_2_data = red_endgame_pd[patriot_2_bool_index]
        patriot_2_num = len(patriot_2_data)
        # 统计爱国者3的数量
        patriot_3_bool_index = red_endgame_pd.WeaponName.str.contains('爱国者-3')
        patriot_3_data = red_endgame_pd[patriot_3_bool_index]
        patriot_3_num = len(patriot_3_data)
        # 统计每枚导弹的拦截结果
        contact_info = {}
        blue_total_contact_list = blue_weaponfired_pd['WeaponID'].tolist()
        # init
        for id in blue_total_contact_list:
            contact_info[id] = {}
            contact_info[id]['weaponIDs'] = list()
            contact_info[id]['pariot_2_num'] = 0
            contact_info[id]['pariot_3_num'] = 0
            contact_info[id]['result'] = 0 # 0表示未知，1表示拦截成功
            contact_info[id]['name'] = list()
        # 统计红方的拦截组合
        red_weapondID = red_weaponfired_pd['WeaponID'].tolist()
        red_TargetContactActualUnitID = red_weaponfired_pd['TargetContactActualUnitID'].tolist()
        red_weaponName = red_weaponfired_pd['WeaponClass'].tolist()
        for i in range(len(red_TargetContactActualUnitID)):
            contact_info[red_TargetContactActualUnitID[i]]['weaponIDs'].append(red_weapondID[i])
            contact_info[red_TargetContactActualUnitID[i]]['name'].append(red_weaponName[i])
            if '爱国者-3' in red_weaponName[i]:
                contact_info[red_TargetContactActualUnitID[i]]['pariot_3_num'] = contact_info[red_TargetContactActualUnitID[i]]['pariot_3_num'] + 1
            elif '爱国者-2' in red_weaponName[i]:
                contact_info[red_TargetContactActualUnitID[i]]['pariot_2_num'] = contact_info[red_TargetContactActualUnitID[i]]['pariot_2_num'] + 1
            else:
                print('[ERROR !!!]')
        red_TargetID = red_endgame_pd['TargetID'].tolist()
        red_Result = red_endgame_pd['Result'].tolist()
        for i in range(len(red_TargetID)):
            if red_Result[i] == 'KILL':
                contact_info[red_TargetID[i]]['result'] = 1
        
        #显示结果
        for key in contact_info.keys():
            if key not in red_TargetContactActualUnitID:
                contact_info[key]['result'] = -1
        null_record = 0
        defeat_record = 0
        success_record = 0
        for key in contact_info.keys():
            if contact_info[key]['result'] == -1:
                null_record = null_record + 1
            if contact_info[key]['result'] == 0:
                defeat_record = defeat_record + 1
            if contact_info[key]['result'] == 1:
                success_record = success_record + 1
        print('[MoziLog]: 爱国者2: %d\t 爱国者3: %d\t 效费比: %f\t 拦截率: %f\t 有拦截记录: %d\t 来袭数量: %d\t 记录拦截成功: %d\t 记录拦截失败: %d\t 无拦截记录: %d'%(patriot_2_num, patriot_3_num, success_record*1.0/(patriot_2_num + patriot_3_num), success_record * 1.0 / len(contact_info), success_record + defeat_record, len(contact_info), success_record, defeat_record, null_record), file = self.mylog, flush = True)  
        print('[MoziLog]: 爱国者2: %d\t 爱国者3: %d\t 效费比: %f\t 拦截率: %f\t 有拦截记录: %d\t 来袭数量: %d\t 记录拦截成功: %d\t 记录拦截失败: %d\t 无拦截记录: %d'%(patriot_2_num, patriot_3_num,  success_record*1.0/(patriot_2_num + patriot_3_num), success_record * 1.0 / len(contact_info), success_record + defeat_record, len(contact_info), success_record, defeat_record, null_record))      
        return contact_info
 
    def _get_ready_file(self):
        res = None  
        try:
            cur_files = os.listdir('/root/MoziLog/Analysis')
            dif = list(set(cur_files) - set(self.read_files))
            for item in dif:
                self.read_files.append(item)
            if len(dif) != 1:
                print('[ERROR]: %d ready_file'%(len(dif)), file = self.mylog, flush = True)
                print('[ERROR]: %d ready_file'%(len(dif)))
                self.log_generate_error = True
            else:
                res = dif[0]
        except:
            self.log_generate_error = True
        return res

    def train(self, episode):
        start_time = time.time()
        print('Training ...')
        for i_episode in range(episode):
            pre_time = time.time()
            print('第%s轮训练',i_episode)
            self.train_each_round(i_episode)
            cur_time = time.time()
            print('[ROUND #%d/%d] 当前推演耗时%ds, 累计耗时%ds'%(i_episode + 1, episode, cur_time-pre_time, cur_time-start_time), file = self.mylog, flush = True)
            print('[ROUND #%d/%d] 当前推演耗时%ds, 累计耗时%ds'%(i_episode + 1, episode, cur_time-pre_time, cur_time-start_time))
            #break
        self.mylog.close()
        