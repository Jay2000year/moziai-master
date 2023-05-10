
from mozi_ai_sdk.base_env import BaseEnvironment
from mozi_ai_sdk import etc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='123.57.142.120')
parser.add_argument("--platform", type=str, default='linux')
parser.add_argument("--scenario_name", type=str, default="test_csv.scen.xml")
parser.add_argument("--port", type=str, default='6061')
# parser.add_argument("--scenario_name", type=str, default="test_csv.scen.xml")
parser.add_argument("--mozi_server_path", type=str, default='D:\\mozi_4p\\mozi\\Mozi\\MoziServer\\bin')
# os.system('rm -rf /root/MoziLog/Analysis/*')


class Environment(BaseEnvironment):
    def __init__(
            self,
            ip,
            port,
            platform=None,
            scenario_name=None,
            simulate_compression=4,
            duration_interval=None,
            synchronous=True,
            app_mode=None
    ):
        super().__init__(ip, port, platform, scenario_name, simulate_compression, duration_interval, synchronous,
                         app_mode)
        self.aircraft = None
        self.env = None
        self.blue_side = None
        self.scenario = None

    def start(self, ip=None, port=None):
        super(Environment, self).start(ip, port)

    def reset(self):
        """
        重置函数
        主要用途：加载想定，
        """
        self.scenario = super(Environment, self).reset()
        return self.scenario

    def step(self):
        self.scenario = super(Environment, self).step()

    def run(self, init_env):
        self.env = init_env
        self.scenario = self.env.reset()
        self.blue_side = self.scenario.get_side_by_name('蓝方')
        while not self.env.is_done():

            """
            决策
            """
            self.aircraft = self.blue_side.get_aircrafts()
            if self.aircraft:
                self.env.step()
                self.aircraft = self.blue_side.get_aircrafts()
            else:
                break


if __name__ == "__main__":
    args = parser.parse_args()
    count = 0
    env = Environment(ip=args.ip,
                      port=args.port,
                      platform=etc.PLATFORM,
                      scenario_name=args.scenario_name,
                      simulate_compression=etc.SIMULATE_COMPRESSION,
                      duration_interval=etc.DURATION_INTERVAL,
                      synchronous=etc.SYNCHRONOUS,
                      app_mode=etc.app_mode)
    env.start(args.ip, args.port)
    while count < 100:
        print('----------------------第{}次加载推演想定---------------------------'.format(count+1))
        env.run(env)
        count += 1
