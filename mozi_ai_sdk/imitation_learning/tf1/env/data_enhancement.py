# -*- coding: utf-8 -*-
"""
# 数据增强实现
"""
import tensorflow as tf
import cv2
import numpy as np
from scipy import misc
import random
import matplotlib.pyplot as plt

# from mozi_ai_sdk.imitation_learning.tf1.A_data_load_test import construction_matrix


def data_cutting():
    # img_path = "C:/Users/3-5/Desktop/26004211cd623464c3a6676f8e0e7c59_1.jpg"
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # h, w, _ = img.shape
    matrix_3D, air_data, batch_list = construction_matrix()
    for index in range(len(batch_list)):
        img = matrix_3D[index].transpose(0, 2, 3, 1)
        action = air_data[index].reshape(batch_list[index], 3, 1)
        _, h, w, _ = img.shape
        new_h1, new_h2 = np.random.randint(0, h - 48, 2)
        new_w1, new_w2 = np.random.randint(0, w - 48, 2)
        img_crop1 = img[:, new_h1 : new_h1 + 512, new_w1 : new_w1 + 512, :]
        img_crop2 = img[:, new_h2 : new_h2 + 512, new_w2 : new_w2 + 512, :]
    # 显示
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1), plt.imshow(img)
    plt.axis("off")
    plt.title("原图")
    plt.subplot(1, 3, 2), plt.imshow(img_crop1)
    plt.axis("off")
    plt.title("水平镜像")
    plt.subplot(1, 3, 3), plt.imshow(img_crop2)
    plt.axis("off")
    plt.title("垂直镜像")
    plt.show()


# data_cutting()


def data_rotate():
    img_path = "C:/Users/3-5/Desktop/26004211cd623464c3a6676f8e0e7c59_1.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 去除黑边的操作
    crop_image = lambda img, x0, y0, w, h: img[
        y0 : y0 + h, x0 : x0 + w
    ]  # 定义裁切函数，后续裁切黑边使用

    def rotate_image(img, angle, crop):
        """
        angle: 旋转的角度
        crop: 是否需要进行裁剪，布尔向量
        """
        w, h = img.shape[:2]
        # 旋转角度的周期是360°
        angle %= 360
        # 计算仿射变换矩阵
        M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        # 得到旋转后的图像
        img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

        # 如果需要去除黑边
        if crop:
            # 裁剪角度的等效周期是180°
            angle_crop = angle % 180
            if angle > 90:
                angle_crop = 180 - angle_crop
            # 转化角度为弧度
            theta = angle_crop * np.pi / 180
            # 计算高宽比
            hw_ratio = float(h) / float(w)
            # 计算裁剪边长系数的分子项
            tan_theta = np.tan(theta)
            numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

            # 计算分母中和高宽比相关的项
            r = hw_ratio if h > w else 1 / hw_ratio
            # 计算分母项
            denominator = r * tan_theta + 1
            # 最终的边长系数
            crop_mult = numerator / denominator

            # 得到裁剪区域
            w_crop = int(crop_mult * w)
            h_crop = int(crop_mult * h)
            x0 = int((w - w_crop) / 2)
            y0 = int((h - h_crop) / 2)
            img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
        return img_rotated

    rotate_image(img, 45, False)
    # 水平镜像
    h_flip = cv2.flip(img, 1)
    # 垂直镜像
    v_flip = cv2.flip(img, 0)
    # 水平垂直镜像
    hv_flip = cv2.flip(img, -1)
    # 90度旋转
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    rotation_45 = cv2.warpAffine(img, M, (cols, rows))
    # 45度旋转
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 135, 2)
    rotation_135 = cv2.warpAffine(img, M, (cols, rows))
    # 去黑边旋转45度
    image_rotated = rotate_image(img, 45, True)

    # 显示

    # plt.figure(figsize=(15, 10))
    # plt.subplot(2, 3, 1), plt.imshow(img)
    # plt.axis('off');
    plt.title("原图")
    plt.subplot(2, 3, 2), plt.imshow(h_flip)
    plt.axis("off")
    plt.title("水平镜像")
    plt.subplot(2, 3, 3), plt.imshow(v_flip)
    plt.axis("off")
    plt.title("垂直镜像")
    plt.subplot(2, 3, 4), plt.imshow(hv_flip)
    plt.axis("off")
    plt.title("水平垂直镜像")
    plt.subplot(2, 3, 5), plt.imshow(rotation_45)
    plt.axis("off")
    plt.title("旋转45度")
    plt.subplot(2, 3, 6), plt.imshow(rotation_135)
    plt.axis("off")
    plt.title("去黑边旋转45度")
    plt.show()


# data_rotate()


def data_zoom():
    img_path = "C:/Users/3-5/Desktop/26004211cd623464c3a6676f8e0e7c59_1.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    img_2 = cv2.resize(img, (int(h * 1.5), int(w * 1.5)))
    img_2 = img_2[
        int((h - 300) / 2) : int((h + 300) / 2),
        int((w - 300) / 2) : int((w + 300) / 2),
        :,
    ]
    img_3 = cv2.resize(img, (300, 300))

    # 显示
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1), plt.imshow(img)
    plt.axis("off")
    plt.title("原图")
    plt.subplot(1, 3, 2), plt.imshow(img_2)
    plt.axis("off")
    plt.title("向外缩放")
    plt.subplot(1, 3, 3), plt.imshow(img_3)
    plt.axis("off")
    plt.title("向内缩放")
    plt.show()


def gasuss_noise(image, mean=0, var=0.001):
    """
    添加高斯噪声
    mean : 均值
    var : 方差
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.0
    else:
        low_clip = 0.0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1), plt.imshow(image)
    plt.axis("off")
    plt.title("原图")
    plt.subplot(1, 3, 2), plt.imshow(out)
    plt.axis("off")
    plt.show()
    return out


def sp_noise(image, prob):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1), plt.imshow(image)
    plt.axis("off")
    plt.title("原图")
    plt.subplot(1, 3, 2), plt.imshow(output)
    plt.axis("off")
    plt.show()
    return output


def Gasuss_and_sp():
    img_path = "C:/Users/3-5/Desktop/26004211cd623464c3a6676f8e0e7c59_1.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = air_missile_3D
    # 添加高斯噪声 ，噪声比例分别是
    # img_s1 = gasuss_noise(img, 0, 0.005)
    img_s2 = gasuss_noise(img, 0, 0.05)


# Gasuss_and_sp()


def Sp_noise():
    img_path = "C:/Users/3-5/Desktop/26004211cd623464c3a6676f8e0e7c59_1.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 添加椒盐噪声，噪声比例为 0.02
    out1 = sp_noise(img, prob=0.02)


Sp_noise()
# data_zoom()
