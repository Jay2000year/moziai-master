# 时间 ： 2020/9/8 21:30
# 作者 ： Dixit
# 文件 ： etc.py
# 项目 ： moziAIBT2
# 版权 ： 北京华戍防务技术有限公司


import os

APP_ABSPATH = os.path.dirname(__file__)

#######################
# SERVER_IP = "127.0.0.1"
SERVER_IP = "123.57.142.120"
SERVER_PORT = "6062"
# SERVER_PORT = "8848"
PLATFORM = 'linux'
# SCENARIO_NAME = "bt_test.scen"  # 距离近，有任务
#SCENARIO_NAME = "hxfb-multitask"
SCENARIO_NAME = "test_csv.scen.xml"
EVAL_SCENARIO_NAME = "海峡风暴-资格选拔赛-蓝方任务随机方案.scen"
# SCENARIO_NAME = "海峡风暴单机AI对抗.scen"
# 0-1倍速，1-2倍速，2-5倍速，3-15倍速，
# 4-30倍速，5-60倍速，6-300倍速，7-900倍速，8-1800倍速
SIMULATE_COMPRESSION = 2
DURATION_INTERVAL = 5
SYNCHRONOUS = True
#######################
# app_mode
# 1--windows的墨子初始化态势
# 2--linux docker容器中的墨子初始化态势
# 3--remote windows evaluate mode
# 4--local windows evaluate mode
app_mode = 2
#######################
MAX_EPISODES = 5000
MAX_BUFFER = 1000000
MAX_STEPS = 30
#######################

#######################
TMP_PATH = "%s/%s/tmp" % (APP_ABSPATH, SCENARIO_NAME)
OUTPUT_PATH = "%s/%s/output" % (APP_ABSPATH, SCENARIO_NAME)

CMD_LUA = "%s/cmd_lua" % TMP_PATH
PATH_CSV = "%s/path_csv" % OUTPUT_PATH
MODELS_PATH = "%s/Models/" % OUTPUT_PATH
EPOCH_FILE = "%s/epochs.txt" % OUTPUT_PATH
#######################

TRANS_DATA = True
