# 时间 ： 2020/7/20 17:13
# 作者 ： Dixit
# 文件 ： etc.py
# 项目 ： moziAIBT
# 版权 ： 北京华戍防务技术有限公司

import os

APP_ABSPATH = os.path.dirname(__file__)

#######################
SERVER_IP = "127.0.0.1"
SERVER_PORT = "6060"
PLATFORM = 'windows'
SCENARIO_NAME = "军科想定(对海)-6条航线VS航母战斗群-6.scen"
SIMULATE_COMPRESSION = 4
DURATION_INTERVAL = 30  # 90太大，运行到判断点的时候，跳过去了
SYNCHRONOUS = True
#######################
MAX_EPISODES = 5000
MAX_BUFFER = 1000000
MAX_STEPS = 30
#######################
# app_mode:
# 1--local windows 本地windows模式
# 2--linux mode    linux模式
# 3--evaluate mode 比赛模式
app_mode = 1