# -*- coding: utf-8 -*-
# @Time    : 2021-01-13 13:21
# @Author  : zxl
# @FileName: ablation.py

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

print(matplotlib.get_backend())

if __name__ == "__main__":


    a = [4, 5, 0]
    b = [3, 2, 0]
    c = [4, 3, 0]
    d = [3, 2, 0]

    a_2 = [0, 0, 0.13]
    b_2 = [0, 0, 0.18]
    c_2 = [0, 0, 0.18]
    d_2 = [0, 0, 0.18]
    labels = ['A', 'B', 'C']

    plt.rcParams['axes.labelsize'] = 16  # xy轴label的size
    plt.rcParams['xtick.labelsize'] = 12  # x轴ticks的size
    plt.rcParams['ytick.labelsize'] = 14  # y轴ticks的size

    # 设置柱形的间隔
    width = 0.2  # 柱形的宽度
    x1_list = []
    x2_list = []
    x3_list = []
    x4_list = []
    for i in range(len(a)):
        x1_list.append(i)
        x2_list.append(i + width)
        x3_list.append(i + 2 * width)
        x4_list.append(i + 3 * width)

    # 创建图层
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('X')
    ax1.set_ylim(0, 6)
    ax1.bar(x1_list, a, width=width, label="a", color='mediumspringgreen', align='edge')
    plt.bar(x2_list, b, width=width, label="b", color='cornflowerblue', align='edge', tick_label=labels)
    plt.bar(x3_list, c, width=width, label="c", color='aqua', align='edge', tick_label=labels)
    plt.bar(x4_list, d, width=width, label="d", color='coral', align='edge', tick_label=labels)
    ax6 = ax1.twinx()
    ax6.set_ylim(0, 0.3)
    ax6.set_ylabel('Y')
    b1 = plt.bar(x1_list, a_2, width=width, label="a", color='mediumspringgreen', align='edge')
    b2 = plt.bar(x2_list, b_2, width=width, label="b", color='cornflowerblue', align='edge', tick_label=labels)
    b3 = plt.bar(x3_list, c_2, width=width, label="c", color='aqua', align='edge', tick_label=labels)
    b4 = plt.bar(x4_list, d_2, width=width, label="d", color='coral', align='edge', tick_label=labels)
    plt.legend(handles=[b1, b2, b3, b4], loc="upper center", ncol=4)
    plt.show()