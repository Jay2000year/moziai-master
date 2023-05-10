#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName: tset.py
# @Time : 2021/10/27 22:55
# @Author : wcowboy


import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)



# safeguard = np.loadtxt('networks/fakedata/保障网.txt')
# scout = np.loadtxt('networks/fakedata/信息网.txt')
# control = np.loadtxt('networks/fakedata/指控网.txt')
# info = np.loadtxt('networks/fakedata/info.txt')
#
# name = ["航母1", "航母2", "E-2D预警机", "F/A-18F战斗机1", "F/A-18F战斗机2",
#         "F-35C战斗机1", "F-35C战斗机2", "MQ-25加油机", "提康德罗加1",
#         "提康德罗加2", "提康德罗加3", "阿利伯克1", "阿利伯克2", "阿利伯克3",
#         "阿利伯克4", "阿利伯克5", "阿利伯克6", "阿利伯克7", "干货补给舰1",
#         "干货补给舰2", "油料补给舰1", "油料补给舰2"]

# safeguard = np.loadtxt('networks/safeguard.txt')
# scout = np.loadtxt('networks/scout.txt')
# control = np.loadtxt('networks/control.txt')
# info = np.loadtxt('networks/info.txt')
#


# name = ["航母", "油料补给舰", "MQ-25加油机", "干货补给舰",
#         "提康德罗加1", "提康德罗加2", "阿利伯克1", "阿利伯克2",
#         "阿利伯克3", "E-2D", "FA-18", "F-35"
#         ]
#
# safeguard = np.loadtxt('networks//fakedata2/1CVBG/保障网.txt')
# scout = np.loadtxt('networks/fakedata2/1CVBG/信息网.txt')
# control = np.loadtxt('networks/fakedata2/1CVBG/指控网.txt')
# info = np.loadtxt('networks/fakedata2/1CVBG/info.txt')
# name = ["航母", "E-2D预警机", "F/A-18F战斗机", "F-35C战斗机", "MQ-25加油机",
#         "提康德罗加1", "提康德罗加2", "阿利伯克1", "阿利伯克2",
#         "阿利伯克3", "干货补给舰", "油料补给舰"]

# 1CVBG

# name = [
#
#     "航母", "E-2D预警机", "F/A-18F战斗机", "F-35C战斗机",
#     "MQ-25加油机", "提康德罗加1", "提康德罗加2", "阿利伯克1",
#     "阿利伯克2", "阿利伯克3", "干货补给舰", "油料补给舰"
#
# ]
#
# safeguard = np.loadtxt('networks//fakedata2/1CVBG/保障网.txt')
# scout = np.loadtxt('networks//fakedata2/1CVBG/信息网.txt')
# control = np.loadtxt('networks//fakedata2/1CVBG/指控网.txt')
# info = np.loadtxt('networks//fakedata2/1CVBG/info.txt')


# ARG
name = [

"两攻",
"坞登1",
"坞登2",
"SH-60直升机",
"F-35B战斗机1",
"F-35B战斗机2",
"阿利伯克1",
"阿利伯克2",
"阿利伯克3",
"油料补给舰"


]

safeguard = np.loadtxt('networks//fakedata2/ARG/保障网.txt')
scout = np.loadtxt('networks//fakedata2/ARG/信息网.txt')
control = np.loadtxt('networks//fakedata2/ARG/指控网.txt')
info = np.loadtxt('networks//fakedata2/ARG/info.txt')


# SSG
# name = [
#     "提康德罗加1",
#     "阿利伯克1",
#     "阿利伯克2",
#     "阿利伯克3",
#     "油料补给舰",
#     "SH-60直升机"]
#
# safeguard = np.loadtxt('networks//fakedata2/SSG/保障网.txt')
# scout = np.loadtxt('networks//fakedata2/SSG/信息网.txt')
# control = np.loadtxt('networks//fakedata2/SSG/指控网.txt')
# info = np.loadtxt('networks//fakedata2/SSG/info.txt')

#
# print(scout.shape)
# print(safeguard.shape)
# print(control.shape)


info_ = []
n, m = len(info), len(info[0])
print(n, m)
df = pd.DataFrame(info)
# print(df)

# 下面是标准化
# 正指标 (x-min)/(max-min)
# 负指标 (max-x)/(max-min)
for i in list(df.columns):
    # 获取各个指标的最大值和最小值
    Max = np.max(df[i])
    Min = np.min(df[i])
    # 标准化
    df[i] = (Max - df[i]) / (Max - Min)


df_ = np.array(df)


# print(df_)

# 下面求指标比重
def bizhong(df_bizhong):
    for column in df_bizhong.columns:
        sigma_xij = sum(df_bizhong[column])
        df_bizhong[column] = df_bizhong[column].apply(lambda x_ij: x_ij / sigma_xij if x_ij / sigma_xij != 0 else 1e-6)

    return df_bizhong


df_bizhong = bizhong(df)

# print(df_bizhong)


# 下面求熵值Hi
# 先算K值
k = 1 / np.log(12)  # z为12
# print(k)
h_j = (-k) * np.array([sum([pij * np.log(pij) for pij in df_bizhong[column]]) for column in df_bizhong.columns])
h_js = pd.Series(h_j, index=df_bizhong.columns)
# 下面求差异系数
df_bianyi = pd.Series(1 - h_j, index=df_bizhong.columns)

# 下面计算指标权重
df_Weight = df_bianyi / sum(df_bianyi)
df_Weight.name = '指标权重'

weight = np.array(df_Weight).T

# print(weight)
# print("权重\n",df_Weight)

result = np.dot(df_, weight.T)

# TODO
'''
归一化完成，写网络聚合
'''

def getNetworkValue(scout, control, safeguard, singleNodeValue, k1, k2):
    n, m = len(scout), len(scout[0])
    # 节点的连接数

    # 子网聚类数
    clustering = [[], [], []]
    for i in range(n):
        clustering[0].append((np.sum(scout[i])) * singleNodeValue[i])
        clustering[1].append((np.sum(control[i])) * singleNodeValue[i])
        clustering[2].append((np.sum(safeguard[i])) * singleNodeValue[i])

    # 侦察 控制 保障
    w = [np.sum(i) for i in clustering]

    # print("子网聚类数:",w)

    nodes = [[], [], []]

    for i in range(n):
        if np.sum(scout[i]):
            nodes[0].append(i)

        if np.sum(control[i]):
            nodes[1].append(i)
        #
        if np.sum(safeguard[i]):
            nodes[2].append(i)

    # print("子网节点", nodes)

    G_od = 0
    # 侦察控制中的边，节点是否在侦察子网中间
    for i in range(n):
        for j in range(n):
            # print(i, j, scout[i][j],end="\t")
            if control[i][j] > 0:
                if i in nodes[0] or j in nodes[0]:
                    G_od += 1
    G_s = 0
    for i in range(n):
        for j in range(n):
            # print(i, j, scout[i][j],end="\t")
            if safeguard[i][j] > 0:
                if i in nodes[1] or j in nodes[1] or i in nodes[0] or j in nodes[0]:
                    G_s += 1

    # print("侦察-控制：", G_od,"侦察-控制-保障：", G_s)

    X = k1 * (w[0] + w[1]) * G_od + k2 * w[2] * G_s

    return X


# 入口

X = getNetworkValue(scout, control, safeguard, result, 0.5, 0.5)

# 初始化归一化矩阵
X_ = np.zeros(n)

# 设置网络权重比
k1, k2 = 0.5, 0.5
for index in range(n):
    t = (getNetworkValue(np.delete(np.delete(scout, [index], axis=1), [index], axis=0),
                         np.delete(np.delete(control, [index], axis=1), [index], axis=0),
                         np.delete(np.delete(safeguard, [index], axis=1), [index], axis=0),
                         np.delete(result, [index], axis=0),
                         k1, k2))
    X_[index] = (X - t) / X

print(X_)

# 接收数据
final = []
print("控制\侦察:保障子网\t", k1, k2, "\n节点名称\t体系价值")
for i in range(n):
    final.append(list([name[i], X_[i]]))

# 数据排序
final.sort(key=lambda x: x[1], reverse=True)

for i in final:
    print(i[0], "\t", i[1])
