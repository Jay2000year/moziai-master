#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName: demo.py
# @Time : 2021/11/16 16:16
# @Author : wcowboy
import math

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def circle(a, r, x0, y0, value):
    """
    最底层函数:矩阵圆形复制
    a:矩阵
    r:场半径
    x0,y0:节点位置
    value：数值
    返回值：a
    """
    start = max(x0 - r, 0)
    end = min(x0 + r + 1, l)

    for i in range(start, end):
        arc_l = (math.sqrt(np.abs((r ** 2 - (i - x0) ** 2))))

        arc_l = round(arc_l)

        y_start = max(y0 - arc_l, 0)

        y_end = min(y0 + arc_l + 1, l)
        a[i][y_start:y_end] += value

    return a


# 探测A类 驱逐舰，巡洋舰
def surveyA(a, x0, y0):
    '''
    公里      威胁
    0-60公里 1 最高级
    60-150 0.6
    150 350 0.3
    350+   0
    '''
    value = [[15, 0.2], [30, 0.3], [50, 0.2], [80, 0.3]]
    for i in range(len(value)):
        a = circle(a, value[i][0] * 10, x0, y0, value[i][1] * t1)

    return a


# 探测 B类  预警机 600km处， <0.1
def surveyB(a, x0, y0):
    temp = 0
    for i in range(600, -1, -1):
        value = (-1 + 0.1) / 600 * i + 1 - temp
        temp = temp + value
        a = circle(a, i, x0, y0, value * t1)
    return a


# 探测 C类 航母，两栖
def surveyC(a, x0, y0):
    value = [[10, 0.5], [20, 0.5]]
    a = circle(a, value[0][0] * 10, x0, y0, value[0][1] * t1)
    a = circle(a, value[1][0] * 10, x0, y0, value[1][1] * t1)
    return a


# A类攻击场
def attackA(a, x0, y0):
    temp1 = temp2 = 0
    for i in range(450, -1, -5):
        value = (-0.4 + 0.1) / 450 * i + 0.4 - temp1
        temp1 = temp1 + value
        a = circle(a, i, x0, y0, value * t2)

    for i in range(60, -1, -1):
        value = -(0.6 - 0.1) / 60 * i + 1 - temp2
        temp2 = temp2 + value
        a = circle(a, i, x0, y0, value * t2)

    return a


# 攻击场 C类
def attackC(a, x0, y0):
    temp = 0
    for i in range(60, -1, -1):
        value = (-1 + 0.1) / 60 * i + 1 - temp
        temp = temp + value
        a = circle(a, i, x0, y0, value * t2)
    return a


# 三类节点在矩阵中刻画
def get_A_value(a, x0, y0):
    a = surveyA(a, x0, y0)
    a = attackA(a, x0, y0)
    return a


def get_B_value(a, x0, y0):
    a = surveyB(a, x0, y0)
    return a


def get_C_value(a, x0, y0):
    a = surveyC(a, x0, y0)
    a = attackC(a, x0, y0)
    return a


# 探测:攻击
t1, t2 = 0.2, 0.8
# 网格范围
l = 800

# 初始点
x0, y0 = 450, 450
# 初始化网格矩阵
a = np.zeros((l, l))

# 按照给定的坐标以及类型在矩阵中进行数据添加
a = get_A_value(a, 350, 200)
a = get_A_value(a, 450, 200)
a = get_A_value(a, 250, 250)
a = get_A_value(a, 550, 250)
a = get_A_value(a, 350, 350)
a = get_A_value(a, 450, 350)

a = get_B_value(a, 400, 600)
a = get_C_value(a, 400, 200)

fig = plt.figure()

# X = np.arange(0, l, 1)
# Y = np.arange(0, l, 1)
# X, Y = np.meshgrid(X, Y)
# plt.axis('equal')
# plt.contourf(X, Y, a)

# 三维画图
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# ax.plot_surface(X, Y, a, rstride=1, cstride=1, cmap='rainbow')

# plt.ylabel("X")  ## y轴标签
# plt.xlabel("Y")  ## x轴标签
# plt.show()

