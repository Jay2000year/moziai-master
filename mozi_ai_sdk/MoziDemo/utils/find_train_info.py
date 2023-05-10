# 时间 : 2021/2/6 9:37 
# 作者 : Dixit
# 文件 : find_train_info.py 
# 项目 : 墨子联合作战智能体研发平台
# 版权 : 北京华戍防务技术有限公司

import os
import json


def read_json(path):
    '''
    by dixit
    2021.02.06
    读取result.json文件
    :param path:
    :return:
    '''

    new_result = 0.0
    training_iteration = 0
    traing_time = 0.0
    filesize = os.path.getsize(path)
    print(filesize)
    if filesize == 0:
        return new_result, training_iteration, traing_time
    else:
        with open(path, 'rb') as fp:
            offset = -50  # 设置偏移量
            while -offset < filesize:
                fp.seek(offset, 2)  # seek(off, 2)表示文件指针：从文件末尾(2)开始向前50个字符(-50)
                lines = fp.readlines()  # 读取文件指针范围内所有行
                if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                    last_line = lines[-1]  # 取最后一行
                    line_json = json.loads(last_line)
                    if line_json and line_json['episode_reward_mean'] and line_json['training_iteration'] and line_json['time_total_s']:
                        new_result = line_json['episode_reward_mean']
                        training_iteration = line_json['training_iteration']
                        traing_time = line_json['time_total_s']
                        return new_result, training_iteration, traing_time
                else:
                    # 如果off为50时得到的readlines只有一行内容，那么不能保证最后一行是完整的
                    # 所以off翻倍重新运行，直到readlines不止一行
                    offset *= 2
            fp.seek(0)
            lines = fp.readlines()
            last_line = lines[-1]  # 取最后一行
            line_json = json.loads(last_line)
            if line_json and line_json['episode_reward_mean'] and line_json['training_iteration'] and line_json['time_total_s']:
                new_result = line_json['episode_reward_mean']
                training_iteration = line_json['training_iteration']
                traing_time = line_json['time_total_s']
    return new_result, training_iteration, traing_time


def find_traing_info(test_report):
    '''
    by dixit
    2021.02.06
    返回当前正在训练trial的信息
    :param test_report:
    :return:
    '''
    lists = os.listdir(test_report)  # 列出目录的下所有文件和文件夹保存到lists
    lists.sort(key=lambda fn: os.path.getmtime(test_report + "/" + fn), reverse=True)  # 按时间排序
    mean_result = 0.0
    training_iteration = 0
    traing_time = 0.0
    for i in range(len(lists)):
        if os.path.isdir(os.path.join(test_report, lists[i])):
            file_new = os.path.join(test_report, lists[i])  # 获取最新trial结果文件夹
            for root_dir, dirs, files in os.walk(file_new):
                if "result.json" in files:
                    result_dir = os.path.join(root_dir, "result.json")  # 一个trial结果
                    mean_result, training_iteration, traing_time = read_json(result_dir)
                    break
            break
    return mean_result, training_iteration, traing_time


if __name__ == '__main__':
    train_scen = 'cartpole-ppo'
    local_dir = r'C:\Users\zhanghong\ray_results'
    dir_test = os.path.join(local_dir, train_scen)
    mean_result, training_iteration, traing_time = find_traing_info(dir_test)
    print(mean_result, training_iteration, traing_time)