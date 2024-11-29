"""
    因为本优化解决了6个工程优化的问题，所以对于不同的工程问题有不同的参数，这里会根据立具体的工程问题选择不同的参数
"""
import numpy as np


def get_h(g):
    if g < 0:
        return 0
    else:
        return 1


def get_functions_details(function_name):
    # 拉/压弹簧设计
    if function_name == "F1":
        fobj = lambda x: F1(x)
        lb = [0.05, 0.25, 2]
        ub = [2, 1.3, 15]
        dim = 3
        return lb, ub, dim, fobj
    # 压力容器设计
    if function_name == "F2":
        fobj = lambda x: F2(x)
        lb = [0, 0, 10, 10]
        ub = [99, 99, 200, 200]
        dim = 4
        return lb, ub, dim, fobj
    # 焊接梁设计
    if function_name == "F3":
        fobj = lambda x: F3(x)
        lb = [0.1, 0.1, 0.1, 0.1]
        ub = [2, 10, 10, 2]
        dim = 4
        return lb, ub, dim, fobj
    # 减速器设计
    if function_name == "F4":
        fobj = lambda x: F4(x)
        lb = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]
        ub = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
        dim = 7
        return lb, ub, dim, fobj
    # 齿轮系设计问题
    if function_name == "F5":
        fobj = lambda x: F5(x)
        lb = [12, 12, 12, 12]
        ub = [60, 60, 60, 60]
        dim = 4
        return lb, ub, dim, fobj
    # 三杆桁架设计问题
    if function_name == "F6":
        fobj = lambda x: F6(x)
        lb = [0, 0]
        ub = [1, 1]
        dim = 2
        return lb, ub, dim, fobj


def F1(x):
    g = [0 for i in range(4)]
    cost = (x[2] + 2) * x[1] * x[0] ** 2
    g[0] = 1 - ((x[2] * (x[1]**3)) / (71785 * (x[0]**4)))
    gtmp = (4 * x[1]**2 - x[0] * x[1]) / (12566 * (x[1] * x[0]**3 - x[0]**4))
    g[1] = gtmp + 1 / (5108 * x[0]**2) - 1
    g[2] = 1 - ((140.45 * x[0]) / ((x[1]**2) * x[2]))
    g[3] = ((x[0] + x[1]) / 1.5) - 1
    lam = 10**15
    z = 0
    for k in range(len(g)):
        z = z + lam * (g[k]**2) * get_h(g[k])
    o = cost + z
    return o


def F2(x):
    g = [0 for i in range(4)]
    cost = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * x[2]**2 + 3.1661 * x[0]**2 * x[3] + 19.84 * x[0]**2 * x[2]
    g[0] = -x[0] + 0.0193 * x[2]
    g[1] = -x[1] + 0.00954 * x[2]
    g[2] = -np.pi * x[2]**2 * x[3] - (4 / 3) * np.pi * x[2]**3 + 1296000
    g[3] = x[3] - 240
    lam = 10**15
    z = 0
    for k in range(len(g)):
        z = z + lam * (g[k]**2) * get_h(g[k])
    o = cost + z
    return o


def F3(x):
    g = [0 for i in range(7)]
    cost = 1.10471 * (x[0] ** 2) * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])
    # 不等式约束
    Q = 6000 * (14 + x[1] / 2)
    D = np.sqrt(x[1]**2 / 4 + (x[0] + x[2])**2 / 4)
    J = 2 * (x[0] * x[1] * np.sqrt(2) * (x[1]**2 / 12 + (x[0] + x[2])**2 / 4))
    alpha = 6000 / (np.sqrt(2) * x[0] * x[1])
    beta = Q * D / J
    tau = np.sqrt(alpha ** 2 + 2 * alpha * beta * x[1] / (2 * D) + beta ** 2)
    sigma = 504000 / (x[3] * x[2]**2)
    delta = 65856000 / (30 * 10 ** 6 * x[3] * x[2]**3)
    F = 4.013 * (30 * 10**6) / 196 * np.sqrt(x[2] ** 2 * x[3]**6 / 36) * (1 - x[2] * np.sqrt(30 / 48) / 28)

    g[0] = tau - 13600
    g[1] = sigma - 30000
    g[2] = x[0] - x[3]
    g[3] = 0.10471 * x[0]**2 + 0.04811 * x[2] * x[3] * (14 + x[1]) - 5.0
    g[4] = 0.125 - x[0]
    g[5] = delta - 0.25
    g[6] = 6000 - F
    lam = 10**15
    z = 0
    for k in range(len(g)):
        z = z + lam * (g[k] ** 2) * get_h(g[k])
    o = cost + z
    return o


def F4(x):
    g = [0 for i in range(11)]
    cost = 0.7854 * x[0] * (x[1] ** 2) * (3.3333 * (x[2] ** 2) + 14.9334 * x[2] - 43.0934) - 1.508 * x[0] * (
            (x[5] ** 2) + (x[6] ** 2)) + 7.4777 * ((x[5] ** 3) + (x[6] ** 3)) + 0.7854 * (
                   x[3] * (x[5] ** 2) + x[4] * (x[6] ** 2))
    g[0] = (27 / (x[0] * (x[1] ** 2) * x[2])) - 1
    g[1] = (397.5 / (x[0] * (x[1] ** 2) * (x[2] ** 2))) - 1
    g[2] = (1.93 * (x[3] ** 3) / (x[1] * (x[5] ** 4) * x[2])) - 1
    g[3] = (1.93 * (x[4] ** 3) / (x[1] * (x[6] ** 4) * x[2])) - 1
    g[4] = (((((745 * x[3] / (x[1] * x[2])) ** 2) + 16.9 * (10 ** 6)) ** (1 / 2)) / (110 * (x[5] ** 3))) - 1
    g[5] = (((((745 * x[4] / (x[1] * x[2])) ** 2) + 157.5 * (10 ** 6)) ** (1 / 2)) / (85 * (x[6] ** 3))) - 1
    g[6] = (x[1] * x[2] / 40) - 1
    g[7] = ((5 * x[1]) / x[0]) - 1
    g[8] = (x[0] / (12 * x[1])) - 1
    g[9] = ((1.5 * x[5] + 1.9) / x[3]) - 1
    g[10] = ((1.1 * x[6] + 1.9) / x[4]) - 1
    lam = 10 ** 15
    z = 0
    for k in range(len(g)):
        z = z + lam * (g[k] ** 2) * get_h(g[k])
    o = cost + z
    return o


def F5(x):
    cost = ((1 / 6.931) - ((x[2] * x[1]) / (x[0] * x[3]))) ** 2
    o = cost
    return o


def F6(x):
    g = [0 for i in range(3)]
    cost = (np.multiply(np.multiply(2, np.sqrt(2)), x[0]) + x[1]) * 100
    g[0] = np.divide(np.multiply(np.sqrt(2), x[0]) + x[1],
                     np.multiply(np.sqrt(2), np.power(x[0], 2)) + np.multiply(np.multiply(2, x[0]), x[1])) * 2 - 2
    g[1] = np.divide(x[1], np.multiply(np.sqrt(2), np.power(x[0], 2)) + np.multiply(np.multiply(2, x[0]), x[1])) * 2 - 2
    g[2] = np.divide(1, np.multiply(np.sqrt(2), x[1]) + x[0]) * 2 - 2
    lam = 10 ** 15
    z = 0
    for k in range(len(g)):
        z = z + lam * (g[k] ** 2) * get_h(g[k])
    o = cost + z
    return o
