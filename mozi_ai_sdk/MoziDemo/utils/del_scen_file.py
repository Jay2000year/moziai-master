# 时间 : 2021/2/6 11:35 
# 作者 : Dixit
# 文件 : del_scen_file.py 
# 项目 : 墨子联合作战智能体研发平台
# 版权 : 北京华戍防务技术有限公司

import os
import shutil


# def del_scen_file(filepath):
#     """
#     删除某一目录下的所有文件或文件夹
#     :param filepath: 路径
#     :return:
#     """
#     del_list = os.listdir(filepath)
#     for f in del_list:
#         file_path = os.path.join(filepath, f)
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#

if __name__ == '__main__':
    train_scen = 'default'
    local_dir = r'C:\Users\zhanghong\ray_results'
    dir_test = os.path.join(local_dir, train_scen)
    if os.path.exists(dir_test):
        shutil.rmtree(dir_test)
    else:
        print('文件不存在')
