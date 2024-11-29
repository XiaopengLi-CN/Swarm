import numpy as np  # 导入NumPy库，用于数值计算

"""
    初始化种群的位置
    该函数用于初始化种群的位置，生成在给定上下界范围内的随机位置
"""

def initialization_position(search_agents_no, dim, ub, lb):
    """
    初始化种群的位置

    参数:
    search_agents_no: int - 种群的大小，即搜索代理的数量
    dim: int - 问题的维度，即每个搜索代理的位置向量的维度
    ub: array-like - 上界，定义每个维度的最大值
    lb: array-like - 下界，定义每个维度的最小值

    返回:
    x: ndarray - 初始化后的种群位置矩阵，每行表示一个搜索代理的位置
    """
    boundary_no = np.shape(ub)[0]  # 获取上界数组的第一个维度的大小，判断是否为单一边界还是多边界
    if boundary_no == 1:  # 如果只有一个边界值，说明所有维度共享相同的上下界
        x = np.random.random_sample((search_agents_no, dim)) * (ub - lb) + lb  # 生成在[lb, ub]范围内的随机数矩阵
        return x  # 返回初始化后的种群位置
    elif boundary_no > 1:  # 如果有多个边界值，说明每个维度有独立的上下界
        x = np.zeros((search_agents_no, dim))  # 初始化种群位置矩阵，所有元素为0
        # 这里是至关重要的，有很多人问我多个变量，每个变量都拥有自己独立的约束的时候，应该如何做。这里就给出了答案
        for i in range(dim):  # 遍历每个维度
            ub_i = ub[i]  # 获取第i个维度的上界
            lb_i = lb[i]  # 获取第i个维度的下界
            x[:, i] = np.reshape(np.random.random_sample((search_agents_no, 1)) * (ub_i - lb_i) + lb_i, (1, search_agents_no))  # 生成在[lb_i, ub_i]范围内的随机数，并赋值给第i列
        return x  # 返回初始化后的种群位置