# 时间 : 2022/4/12 17:17
# 作者 : zhait
# 文件 : model_train.py
# 项目 : moziai
# 版权 : 北京华戍防务技术有限公司
# 训练模型

import tensorflow as tf
import numpy as np
from mozi_ai_sdk.imitation_learning.PCGrad_tf import PCGrad
import matplotlib.pyplot as plt
from mozi_ai_sdk.imitation_learning.tf1.env.load_data import ConstructionMatrix


def multi_layer_cnn(rnn_type):
    data = ConstructionMatrix()
    config = {
        "conv_activation": tf.nn.relu,
        "conv_filters": [[16, 8, 4], [32, 4, 2], [256, 11, 1]],
    }
    # obs_space = spaces.Box(low=-10, high=10, shape=(84, 84, 3))
    # action_space = spaces.Box(low=-10, high=10, shape=(3,))
    input_1 = tf.placeholder(tf.float32, [None, 128, 128, 3])
    label = tf.placeholder(tf.float32, [None, 3, 1])
    # input_1 = tf.placeholder(tf.float32, [None, 100, 100, 3])
    # label = tf.placeholder(tf.float32, [None, 3, 1])

    cnn_layer_1 = conv(input_1, 4, 16, 3, "cnn_1")
    cnn_layer_2 = conv(cnn_layer_1, 4, 32, 3, "cnn_2")
    cnn_layer_3 = conv(cnn_layer_2, 8, 256, 5, "cnn_3", padding="VALID")

    y = tf.layers.flatten(cnn_layer_3)
    fc_1 = fc(y, 256, "fc_1")  # output: [batch_size, hidden_size]
    cnn_share_net = tf.expand_dims(fc_1, 0)
    lstm_hidden_size = 256
    if rnn_type == "lstm":
        # num_units 参数 代表lstm单元数
        cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=lstm_hidden_size, state_is_tuple=True
        )
    else:
        cell = tf.contrib.rnn.GRUCell(num_units=lstm_hidden_size)

    rnn_outputs, _ = tf.nn.dynamic_rnn(
        cell=cell, dtype=tf.float32, sequence_length=[200], inputs=cnn_share_net
    )
    multi_head_in = tf.squeeze(rnn_outputs)
    # a11_head_in = fc(multi_head_in, 256, "a11_head", in_channels=lstm_hidden_size)
    a1_head = fc(
        fc_1, 1, "a1_head", in_channels=lstm_hidden_size
    )  # output: [batch_size, hidden_size]
    a1_embedding = fc(a1_head, 256, "a1_embedding", in_channels=1)

    a2_head_in = tf.add(fc_1, a1_embedding)
    a2_head = fc(
        a2_head_in, 1, "a2_head", in_channels=lstm_hidden_size
    )  # output: [batch_size, hidden_size]
    a2_embedding = fc(a2_head, 256, "a2_embedding", in_channels=1)

    a3_head_in = tf.add(a2_head_in, a2_embedding)
    a3_head = fc(
        a3_head_in, 1, "a3_head", in_channels=lstm_hidden_size
    )  # output: [batch_size, hidden_size]

    mse_1 = tf.reduce_mean(tf.square(a1_head - label[:, 0]))  # (a1_head, label[:, 0])
    mse_2 = tf.reduce_mean(tf.square(a2_head - label[:, 1]))
    mse_3 = tf.reduce_mean(tf.square(a3_head - label[:, 2]))

    optimizer = PCGrad(tf.train.AdamOptimizer())  # wrap your favorite optimizer
    losses = [mse_1, mse_2, mse_3]  # a list of per-task losses
    a1_loss_value = []
    a2_loss_value = []
    a3_loss_value = []
    epochs = []
    train_op = optimizer.minimize(losses, var_list=tf.trainable_variables())
    #  obtain saver target
    saver = tf.train.Saver()
    #  obtain summary data
    #     merged = tf.contrib.deprecated.merge_all_summaries()
    # train_writer = tf.train.SummaryWriter()
    # with tf.Session() as session:
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    matrix_state, actions, batch_list = data.export()
    for epoch in range(5):
        for index in range(len(batch_list)):
            # for index_1 in range(batch_list):
            state = matrix_state[index]
            # state = tf.tile(state, multiples = [2, 1, 1, 1])
            action = actions[index]
            # action = tf.tile(action, multiples = [2, 1, 1])
            x_data = state
            y_data = action
            # 加载数据随机抽样
            # dataset = tf.contrib.data.shuffle_and_repeat()
            # 检测session中是否增加了新的节点
            session.graph.finalize()
            #  返回图的节点数据
            # tf.io.write_graph(session.graph_def, "C:\\Users\\3-5\\Desktop\\data", 'graph.pb', as_text=True)
            loss_value, res = session.run(
                [losses, train_op], feed_dict={input_1: x_data, label: y_data}
            )
            a1_loss_value.append(loss_value[0])
            a2_loss_value.append(loss_value[1])
            a3_loss_value.append(loss_value[2])
            epochs.append(epoch)
            print(f"epoch: {epoch}, loss_value: {loss_value}")
            # tf.reset_default_graph()
    saver.save(session, "../new128_4_4_8_tmp_5/pcgrad_regression.ckpt")
    # a1, a2, a3 = session.run([a1_head, a2_head, a3_head], feed_dict={input_1: x_data, label: y_data})
    # # 画图
    # x = np.linspace(0, 1, 23)
    # plt.figure(figsize=(6, 6), dpi=80)
    # plt.figure(1)
    # plt.subplot(331)
    # plt.plot(x, a1, color="b", linestyle="--")
    # plt.plot(x, y_data[:, 0], c="r")
    # plt.subplot(332)
    # plt.plot(x, a2, color="b", linestyle="-")
    # plt.plot(x, y_data[:, 1], c="r")
    # plt.subplot(333)

    # plt.plot(x, a3, color="b", linestyle="-")
    # plt.plot(x, y_data[:, 2], c="r")
    # plt.subplot(334)
    # plt.plot(epochs, a1_loss_value, color="b", linestyle="-")
    # plt.xlim((0, 100))
    # plt.subplot(335)
    # plt.plot(epochs, a2_loss_value, color="b", linestyle="-")
    # plt.xlim((0, 100))
    # plt.subplot(336)
    # plt.plot(epochs, a2_loss_value, color="b", linestyle="-")
    # plt.xlim((0, 100))
    # plt.show()


def batch_normalization_layer(inputs, out_channels, name):
    mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2])  # 计算均值和方差
    beta = tf.get_variable(
        name + "_beta", out_channels, tf.float32, initializer=tf.zeros_initializer
    )
    gamma = tf.get_variable(
        name + "_gamma", out_channels, tf.float32, initializer=tf.ones_initializer
    )
    bn_layer = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

    return bn_layer


def conv(in_, kernel_size, out_channels, stride, name, padding="SAME"):
    in_channels = in_.shape[-1]
    # filter weights
    conv_weights = tf.get_variable(
        name=name,
        shape=[kernel_size, kernel_size, in_channels, out_channels],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.00004),
    )
    conv_layer = tf.nn.conv2d(
        in_, conv_weights, [1, stride, stride, 1], padding=padding
    )  # 卷积操作
    batch_norm = batch_normalization_layer(conv_layer, out_channels, name)
    conv_output = tf.nn.relu(batch_norm)  # relu激活函数
    return conv_output


def fc(in_, out_channels, name, in_channels=None, activation_func=True):
    if not in_channels:
        in_channels = in_.shape[-1]
    # 创建 全连接权重 变量
    fc_weights = tf.get_variable(
        name=name + "_weights",
        shape=[in_channels, out_channels],
        # shape = [out_channels,in_channels]
        initializer=tf.truncated_normal_initializer(stddev=0.01),
        dtype=tf.float32,
        regularizer=tf.contrib.layers.l2_regularizer(0.01),
    )
    # 创建 全连接偏置 变量
    fc_biases = tf.get_variable(
        name=name + "_biases",
        shape=[out_channels],
        initializer=tf.zeros_initializer,
        dtype=tf.float32,
    )

    fc_layer = tf.matmul(in_, fc_weights)  # 全连接计算
    # fc_layer = tf.multiply(in_, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)  # 加上偏置项
    if activation_func:
        fc_layer = tf.nn.tanh(fc_layer)  # relu激活函数
    return fc_layer


if __name__ == "__main__":
    multi_layer_cnn("lstm")
