import math

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ThreatField:
    def __init__(self, Detect_weight=0.2, Attack_weight=0.8, grid_size=800, *args, **kwargs):
        self.Detect_weight = Detect_weight
        self.Attack_weight = Attack_weight
        self.grid_size = grid_size
        self.ThreatFieldArray = np.zeros((grid_size, grid_size))

    def _Circle(self, targetX, targetY, field_radius, value):
        start = max(targetX - field_radius, 0)
        end = min(targetX + field_radius + 1, self.grid_size)

        for i in np.arange(int(np.ceil(start)), int(np.ceil(end))):
            arc_l = (math.sqrt(np.abs((field_radius ** 2 - (i - targetX) ** 2))))

            arc_l = round(arc_l)

            y_start = max(targetY - arc_l, 0)

            y_end = min(targetY + arc_l + 1, self.grid_size)

            y_start = int(np.ceil(y_start))
            y_end = int(np.ceil(end))

            self.ThreatFieldArray[i][y_start:y_end] += value

        return self.ThreatFieldArray

    '''
    公里      威胁
    0-60公里 1 最高级
    60-150 0.6  
    150 350 0.3
    350+   0
    '''

    # 探测A类
    def _surveyA(self, targetX, targetY):
        value = [[15, 0.2], [30, 0.3], [50, 0.2], [80, 0.3]]
        for i in range(len(value)):
            self.ThreatFieldArray = self._Circle(targetX, targetY, value[i][0] * 10, value[i][1] * self.Detect_weight)

        return self.ThreatFieldArray

    # 探测 B类  600km处， <0.1
    def _surveyB(self, targetX, targetY):
        temp = 0
        for i in range(600, -1, -1):
            value = (-1 + 0.1) / 600 * i + 1 - temp
            temp = temp + value
            self.ThreatFieldArray = self._Circle(targetX, targetY, i, value * self.Detect_weight)

        return self.ThreatFieldArray

    # 探测 C类
    def _surveyC(self, targetX, targetY):
        value = [[10, 0.5], [20, 0.5]]
        self.ThreatFieldArray = self._Circle(targetX, targetY, value[0][0] * 10, value[0][1] * self.Detect_weight)
        self.ThreatFieldArray = self._Circle(targetX, targetY, value[1][0] * 10, value[1][1] * self.Detect_weight)
        return self.ThreatFieldArray

    # A类攻击场
    def _attackA(self, targetX, targetY):
        temp1 = temp2 = 0
        for i in range(450, -1, -5):
            value = (-0.4 + 0.1) / 450 * i + 0.4 - temp1
            temp1 = temp1 + value
            self.ThreatFieldArray = self._Circle(targetX, targetY, i, value * self.Attack_weight)

        for i in range(60, -1, -1):
            value = -(0.6 - 0.1) / 60 * i + 1 - temp2
            temp2 = temp2 + value
            self.ThreatFieldArray = self._Circle(targetX, targetY, i, value * self.Attack_weight)
        return self.ThreatFieldArray

    '''
    攻击场 C类
    '''

    def _attackC(self, targetX, targetY):
        temp = 0
        for i in range(60, -1, -1):
            value = (-1 + 0.1) / 60 * i + 1 - temp
            temp = temp + value
            self.ThreatFieldArray = self._Circle(targetX, targetY, i, value * self.Attack_weight)
        return self.ThreatFieldArray

    # 三类节点在矩阵中刻画
    def _get_A_value(self, targetX, targetY):
        self.ThreatFieldArray = self._surveyA(targetX, targetY)
        self.ThreatFieldArray = self._attackA(targetX, targetY)
        return self.ThreatFieldArray

    def _get_B_value(self, targetX, targetY):
        self.ThreatFieldArray = self._surveyB(targetX, targetY)
        return self.ThreatFieldArray

    def _get_C_value(self, targetX, targetY):
        self.ThreatFieldArray = self._surveyC(targetX, targetY)
        self.ThreatFieldArray = self._surveyC(targetX, targetY)
        return self.ThreatFieldArray

    # 按照给定的坐标以及类型在矩阵中进行数据添加
    def CalThreatField(self, Target_pos=None, Catogory=None):
        for i in range(len(Target_pos)):
            if Catogory[i] == "A":
                self.ThreatFieldArray = self._get_A_value(Target_pos[i][0], Target_pos[i][1])
            elif Catogory[i] == "B":
                self.ThreatFieldArray = self._get_B_value(Target_pos[i][0], Target_pos[i][1])
            elif Catogory[i] == "C":
                self.ThreatFieldArray = self._get_C_value(Target_pos[i][0], Target_pos[i][1])
        # return self.ThreatFieldArray


if __name__ == "__main__":
    grid_size = 800 / 5

    thread_ins = ThreatField(grid_size=int(np.ceil(grid_size)))
    # 目标的坐标
    positions = np.array(
        [[350, 200], [450, 200], [250, 250], [550, 250], [350, 350], [450, 350], [400, 600], [400, 200]]) / 5
    catogory = ["A", "A", "A", "A", "A", "A", "B", "C", ]
    thread_ins.CalThreatField(positions, catogory)
    ThreatFieldArray = thread_ins.ThreatFieldArray

    print(ThreatFieldArray.shape)

    # X = np.arange(0, grid_size, 1)
    # Y = np.arange(0, grid_size, 1)
    # X, Y = np.meshgrid(X, Y)

    # fig = plt.figure()
    # plt.axis('equal')
    # plt.contourf(X,Y,ThreatFieldArray)

    # # 三维画图
    # ax = Axes3D(fig)
    # fig.add_axes(ax)
    # ax.plot_surface(X, Y,ThreatFieldArray, rstride=1, cstride=1, cmap='rainbow')

    # plt.ylabel("X") ## y轴标签
    # plt.xlabel("Y")  ## x轴标签
    # plt.show()
