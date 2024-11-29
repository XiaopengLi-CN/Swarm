import numpy as np

"""
    初始化种群的位置
"""


def initialization_position(search_agents_no, dim, ub, lb):
    boundary_no = np.shape(ub)[0]
    if boundary_no == 1:
        x = np.random.random_sample((search_agents_no, dim)) * (ub - lb) + lb
        return x
    elif boundary_no > 1:
        x = np.zeros((search_agents_no, dim))
        # 这里是至关重要的，有很多人问我多个变量，每个变量都拥有自己独立的约束的时候，应该如何做。这里就给出了答案
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            x[:, i] = np.reshape(np.random.random_sample((search_agents_no, 1)) * (ub_i - lb_i) + lb_i, (1, search_agents_no))
        return x
