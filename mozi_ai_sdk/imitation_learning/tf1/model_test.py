# 时间 : 2022/1/10 17:17
# 作者 : Dixit
# 文件 : test_dynamic_rnn.py
# 项目 : moziai
# 版权 : 北京华戍防务技术有限公司


import tensorflow as tf
import numpy as np
from gym import spaces
from mozi_ai_sdk.imitation_learning.PCGrad_tf import PCGrad
import matplotlib.pyplot as plt

# from moziai.mozi_ai_sdk.imitation_learning.tf1.data_load_test import construction_matrix
# from moziai.mozi_ai_sdk.imitation_learning.tf1.interactive_test import interactive_test
import os


def multi_layer_cnn(rnn_type, matrix, batch):
    tf.reset_default_graph()
    config = {
        "conv_activation": tf.nn.relu,
        "conv_filters": [[16, 8, 4], [32, 4, 2], [256, 11, 1]],
    }
    # obs_space = spaces.Box(low=-10, high=10, shape=(84, 84, 3))
    # action_space = spaces.Box(low=-10, high=10, shape=(3,))
    input_1 = tf.placeholder(tf.float32, [None, 128, 128, 3])
    label = tf.placeholder(tf.float32, [None, 3, 1])
    # batch_size_ph = tf.placeholder(tf.int32, [])
    # input_1 = tf.placeholder(tf.float32, [None, 100, 100, 3])
    # label = tf.placeholder(tf.float32, [None, 3, 1])
    # input_1 = tf.Variable(tf.random_uniform((200, 84, 84, 3), minval=-1, maxval=1, dtype=tf.float32))
    # label = tf.Variable(tf.random_uniform((200, 3, 1), minval=-1, maxval=
    #
    #        1, dtype=tf.float32))

    cnn_layer_1 = conv(input_1, 4, 16, 3, "cnn_1")
    cnn_layer_2 = conv(cnn_layer_1, 4, 32, 3, "cnn_2")
    cnn_layer_3 = conv(cnn_layer_2, 8, 256, 5, "cnn_3", padding="VALID")

    y = tf.layers.flatten(cnn_layer_3)
    fc_1 = fc(y, 256, "fc_1")  # output: [batch_size, hidden_size]
    # cnn_share_net = tf.expand_dims(fc_1, 0)
    #
    lstm_hidden_size = 256
    # if rnn_type == 'lstm':
    #     cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_hidden_size, state_is_tuple=False)
    # else:
    #     cell = tf.contrib.rnn.GRUCell(num_units=lstm_hidden_size)
    #
    # # state_in = cell.zero_state(batch_size_ph, tf.float32)
    # rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
    #     cell=cell,
    #     dtype=tf.float32,
    #     sequence_length=[200],
    #     inputs=cnn_share_net
    #     # initial_state=state_in
    #     )

    # multi_head_in = tf.squeeze(rnn_outputs, 0)
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

    # optimizer = PCGrad(tf.train.AdamOptimizer())  # wrap your favorite optimizer
    losses = [mse_1, mse_2, mse_3]  # a list of per-task losses
    a1_loss_value = []
    a2_loss_value = []
    a3_loss_value = []
    # epochs = []
    # train_op = optimizer.minimize(losses, var_list=tf.trainable_variables())
    # optimizer = tf.train.AdamOptimizer()
    # losses = mse_1 + mse_2 + mse_3
    # losses_value = []
    # epochs = []
    # train_op = optimizer.minimize(losses, var_list=tf.trainable_variables())
    agent_states = {}

    saver = tf.train.Saver()

    with tf.Session() as session:
        if os.path.exists("new128_4_4_8_tmp_5/checkpoint"):
            saver.restore(session, "new128_4_4_8_tmp_5/pcgrad_regression.ckpt")
        # new_saver = tf.train.import_meta_graph('D:\Agent Code\moziai\mozi_ai_sdk\imitation_learning\tf1\')
        # new_saver.restore(session, tf.train.latest_checkpoint('moziai\mozi_ai_sdk\imitation_learning\tf1\checkpoint'))
        # saver.restore(session,r'mozi_ai_sdk/imitation_learning/tf1/tmp/pcgrad_regression.ckpt.data-00000-of-00001')
        # session.run(tf.global_variables_initializer())
        # session.run()
        # tf.get_default_graph()
        # x_data = np.random.uniform(-1, 1, (200, 100, 100, 3))
        # matrix_3D,air_data, batch_list= construction_matrix()
        # matrix_3D,air_data = interactive_test()
        state = matrix

        # for epoch in range(100):
        # for index in range(len(batch_list)):

        # action = air_data.reshape(1, 3, 1)
        x_data = state
        # y_data = action
        # y_data = np.random.uniform(-1, 1, (200, 3, 1))
        #         print(tf.global_variables())
        #     batch = batch
        #     states = np.empty([1, 512])
        #     default = np.zeros([512])
        #     if agent_states == {}:
        #         pass
        #     else:
        #         states = agent_states.get(0, default)

        # loss_value = session.run([losses], feed_dict={input_1:x_data})
        loss_value = session.run(
            [a1_head, a2_head, a3_head], feed_dict={input_1: x_data}
        )

        # agent_states[0] = rnn_state
        # a1_loss_value.append(loss_value[0][0])
        # a2_loss_value.append(loss_value[0][1])
        # a3_loss_value.append(loss_value[0][2])
        # epochs.append(epoch)
        # print(f"epoch: {''}, loss_value: {loss_value}")
        a = loss_value[0]
        b = loss_value[0][:, 0]
        print(loss_value[0][:, 0][0], loss_value[1][:, 0][0], loss_value[2][:, 0][0])
        return loss_value[0][:, 0][0], loss_value[1][:, 0][0], loss_value[2][:, 0][0]

        # saver.save(session,'pcgrad_regression.ckpt')
        # a1, a2, a3 = session.run([a1_head, a2_head, a3_head], feed_dict={input_1: x_data, label: y_data})
        # # 画图
        # x = np.linspace(0, 10, 200)
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

    # fc_layer = tf.multiply(in_, fc_weights)  # 全连接计算
    fc_layer = tf.matmul(in_, fc_weights)  # 全连接计算
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)  # 加上偏置项
    if activation_func:
        fc_layer = tf.nn.tanh(fc_layer)  # relu激活函数
    return fc_layer


# if __name__ == '__main__':
# dynamic_rnn(rnn_type='lstm')
# multi_layer_cnn('lstm')
