# 时间 : 2021/2/5 21:20 
# 作者 : Dixit
# 文件 : handle_ray_result.py 
# 项目 : 墨子联合作战智能体研发平台
# 版权 : 北京华戍防务技术有限公司

import json
import os
from pathlib import Path


def read_json(path):
    '''
    by dixit
    2021.02.06
    读取result.json文件
    :param path:
    :return:
    '''

    tmp_result = []
    tmp_training_iteration = []
    with open(path, 'r') as fp:
        # content = fp.readlines()
        while True:
            line = fp.readline()
            if line:
                line_json = json.loads(line)
                if line_json and line_json['episode_reward_mean'] and line_json['training_iteration']:
                    tmp_result.append(line_json['episode_reward_mean'])
                    tmp_training_iteration.append(line_json['training_iteration'])
            else:
                break
    if tmp_result and tmp_training_iteration:
        tmp_result_iter = zip(tmp_result, tmp_training_iteration)
        tmp_sorted_result_iter = sorted(list(tmp_result_iter), reverse=False)
        return tmp_sorted_result_iter
    else:
        return None


def find_checkpoint(tmp_sorted_result_iter, dirs):
    '''
    by dixit
    2021.02.06
    搜索一个trial最优的结果
    :param tmp_sorted_result_iter:
    :param dirs:
    :return:
    '''
    length = len(tmp_sorted_result_iter)
    for _ in range(length):
        tmp_max_result = tmp_sorted_result_iter.pop()
        if dirs:
            if tmp_max_result and f'checkpoint_{tmp_max_result[-1]}' in dirs:
                return tmp_max_result
            else:
                pass
        else:
            return None


def find_best_checkpoint_dir(train_dir):
    '''
    by dixit
    2021.02.06
    寻找单个训练场景所有trial的最优checkpoint
    :param train_dir:
    :return:
    '''
    lists = os.listdir(train_dir)  # 列出目录的下所有文件和文件夹保存到lists
    lists.sort(key=lambda fn: os.path.getmtime(train_dir + "/" + fn), reverse=True)  # 按时间排序
    max_result = -float('inf')
    tmp_max_result = None
    best_checkpoint_dir = None
    param_dir = None
    tmp_training_iteration = None
    for i in range(len(lists)):
        if os.path.isdir(os.path.join(train_dir, lists[i])):
            file_new = os.path.join(train_dir, lists[i])  # 获取一个trial结果文件夹
            for root_dir, dirs, files in os.walk(file_new):
                # print(root_dir)
                # print(root, dirs, files)
                if "result.json" in files:
                    result_dir = os.path.join(root_dir, "result.json")  # 一个trial结果
                    tmp_sorted_result_iter = read_json(result_dir)
                    if tmp_sorted_result_iter:
                        tmp_max_result = find_checkpoint(tmp_sorted_result_iter, dirs)
                    else:
                        break
                if tmp_max_result and f'checkpoint-{tmp_max_result[-1]}' in files:
                    if tmp_max_result[0] > max_result:
                        max_result = tmp_max_result[0]
                        best_checkpoint_dir = os.path.join(root_dir, f'checkpoint-{tmp_max_result[-1]}')
                        break
                    else:
                        break
    if best_checkpoint_dir:
        best_trial_path = Path(best_checkpoint_dir).parent.parent
        param_dir = os.path.join(best_trial_path, 'params.json')
        for file_name in os.listdir(best_trial_path):
            if 'events.out.tfevents' in file_name:
                tf_events_dir = os.path.join(best_trial_path, file_name)

    return tf_events_dir, best_checkpoint_dir, param_dir


if __name__ == '__main__':
    # 训练ID
    train_scen = '1cdab424-7447-11eb-b32a-00163e34c521'
    # 默认结果文件夹
    local_dir = r'/root/mozi_ai/1cda5af6-7447-11eb-b32a-00163e34c521/result'
    dir_test = os.path.join(local_dir, train_scen)
    if os.path.exists(dir_test):
        # tf_events_dir          tensor board文件路径
        # best_checkpoint_dir    最优checkpoint
        # param_dir  最优超参路径
        tf_events_dir, best_checkpoint_dir, param_dir = find_best_checkpoint_dir(dir_test)
        print(tf_events_dir, '\n', best_checkpoint_dir, '\n', param_dir)
    else:
        print('文件不存在')
