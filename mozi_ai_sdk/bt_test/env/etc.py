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
PLATFORM = "windows"
# SCENARIO_NAME = "bt_test.scen"  # 距离近，有任务
# SCENARIO_NAME = "海峡风暴-资格选拔赛.scen"  # 没有任务
SCENARIO_NAME = "海峡风暴-资格选拔赛-蓝方任务随机方案-给周国进测试.scen"
# 0-1倍速，1-2倍速，2-5倍速，3-15倍速，
# 4-30倍速，5-60倍速，6-300倍速，7-900倍速，8-1800倍速
SIMULATE_COMPRESSION = 3
# 决策步长，单位秒
# 环境类里有个step()方法，DURATION_INTERVAL是执行这个方法之后，向前推进的想定时间。
# 执行这个step()方法后，会有一个态势更新的动作，我们可以根据更新的态势做决策，所有叫决策步长。
# 设置的值应大于等于推演档位倍速的值，否则会影响推进速度
DURATION_INTERVAL = 20
# 同步模式：调用step方法后，推进暂停，直到智能体客户端调用下一个step才向下推进
# 异步模式：当前版本不支持
SYNCHRONOUS = True
#######################
MAX_EPISODES = 5000
MAX_BUFFER = 1000000
MAX_STEPS = 30
#######################
# APP_MODE:
# 1 -- 本地windows模式
# 2 -- linux模式
# 3 -- 比赛模式
# 三种模式在创建env对象时有所区别
# app_mode=1或2
# env = Environment(etc.SERVER_IP, etc.SERVER_PORT, etc.PLATFORM, etc.SCENARIO_NAME_DEMO_01,
#                   etc.SIMULATE_COMPRESSION, etc.DURATION_INTERVAL, etc.SYNCHRONOUS, etc.APP_MODE)
# app_mode=3
# env = Environment(ip, port, duration_interval=etc.DURATION_INTERVAL, app_mode=3,
#                   agent_key_event_file=args.agent_key_event_file)
app_mode = 1
# Windows下墨子安装目录下bin目录
MOZI_PATH = "D:\\ZT-mozi\\Mozi\\MoziServer\\bin"
