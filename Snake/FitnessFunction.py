"""
    这个是适应度函数（量子启发算法的视频 里面介绍了什么是 适应度函数）
"""
import numpy as np  # 导入NumPy库，用于数组和矩阵操作

def chung_reynolds(x):
    """
    计算Chung Reynolds函数的适应度值

    参数:
    x: np.matrix, 输入的解向量

    返回:
    f: float, 适应度值
    """
    n = np.shape(x)[1]  # 获取解向量的维度
    f = 0  # 初始化适应度值
    for i in range(n):
        # 计算每个维度的适应度值并累加
        f = f + np.power(np.power(x[0, i], 2), 2)
    return f  # 返回适应度值