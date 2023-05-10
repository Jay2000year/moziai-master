import numpy as np


# 求每一列(即每个时刻)中最大值对应的softmax值
def softmax(logits):
    # 注意这里求e的次方时，次方数减去max_value其实不影响结果，因为最后可以化简成教科书上softmax的定义
    # 次方数加入减max_value是因为e的x次方与x的极限(x趋于无穷)为无穷，很容易溢出，所以为了计算时不溢出，就加入减max_value项
    # 次方数减去max_value后，e的该次方数总是在0到1范围内。
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


def remove_blank(labels, blank=0):
    new_labels = []
    # 合并相同的标签
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels


def beam_decode(y, beam_size=10):
    # y是个二维数组，记录了所有时刻的所有项的概率
    T, V = y.shape
    # 将所有的y中值改为log是为了防止溢出，因为最后得到的p是y1..yn连乘，且yi都在0到1之间，可能会导致下溢出
    # 改成log(y)以后就变成连加了，这样就防止了下溢出
    log_y = np.log(y)
    # 初始的beam
    beam = [([], 0)]
    # 遍历所有时刻t
    for t in range(T):
        # 每个时刻先初始化一个new_beam
        new_beam = []
        # 遍历beam
        for prefix, score in beam:
            # 对于一个时刻中的每一项(一共V项)
            for i in range(V):
                # 记录添加的新项是这个时刻的第几项，对应的概率(log形式的)加上新的这项log形式的概率(本来是乘的，改成log就是加)
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]
                # new_beam记录了对于beam中某一项，将这个项分别加上新的时刻中的每一项后的概率
                new_beam.append((new_prefix, new_score))
        # 给new_beam按score排序
        new_beam.sort(key=lambda x: x[1], reverse=True)
        # beam即为new_beam中概率最大的beam_size个路径
        beam = new_beam[:beam_size]

    return beam


np.random.seed(1111)
y_test = softmax(np.random.random([20, 6]))
beam_chosen = beam_decode(y_test, beam_size=100)
for beam_string, beam_score in beam_chosen[:20]:
    print(remove_blank(beam_string), beam_score)
