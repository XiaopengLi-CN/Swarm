import numpy as np
from initialization import initialization_position
from levy import levy
"""
    优化思想的主体
"""


def GJO(search_agents_no, max_iteration, lb, ub, dim, fobj):
    # 初始化金豺对
    Male_Jackal_pos = [0 for i in range(dim)]
    Male_Jackal_score = np.inf
    Female_Jackal_pos = [0 for i in range(dim)]
    Female_Jackal_score = np.inf

    # 初始化个体的位置
    positions = initialization_position(search_agents_no, dim, ub, lb)
    convergence_curve = np.zeros((1, max_iteration))
    # 循环的次数
    l = 0

    # 主循环
    while l < max_iteration:
        for i in range(np.shape(positions)[0]):
            # 边界检查
            flag4ub = positions[i, :] > ub
            flag4lb = positions[i, :] < lb
            # 边界检查为什么这样写，我在蛇优化中进行详细的讲解了。主要的思想就是把不合规的先置为0，之后大于边界的就设置为边界的最大值，小于边界的就设置为边界的最小值
            positions[i, :] = np.multiply(positions[i, :], (~(flag4ub + flag4lb))) + np.multiply(ub, flag4ub) + np.multiply(lb, flag4lb)
            # 计算每一个个体的适应度
            fitness = fobj(positions[i, :])
            # 更新雄性豺狼
            if fitness < Male_Jackal_score:
                Male_Jackal_score = fitness
                Male_Jackal_pos = positions[i, :]
            if Male_Jackal_score < fitness < Female_Jackal_score:
                Female_Jackal_score = fitness
                Female_Jackal_pos = positions[i, :]
        E1 = 1.5 * (1 - (l / max_iteration))
        RL = 0.05 * levy(search_agents_no, dim, 1.5)
        Male_Positions = np.zeros((np.shape(positions)[0], np.shape(positions)[1]))
        Female_Positions = np.zeros((np.shape(positions)[0], np.shape(positions)[1]))
        for i in range(np.shape(positions)[0]):
            for j in range(np.shape(positions)[1]):
                r1 = np.random.random()
                E0 = 2 * r1 - 1
                # 逃避的能量
                E = E1 * E0
                if np.abs(E) < 1:
                    D_male_jackal = np.abs((RL[i, j] * Male_Jackal_pos[j] - positions[i, j]))
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((RL[i, j] * Female_Jackal_pos[j] - positions[i, j]))
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal
                else:
                    D_male_jackal = np.abs((Male_Jackal_pos[j] - RL[i, j] * positions[i, j]))
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((Female_Jackal_pos[j] - RL[i, j] * positions[i, j]))
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal
                positions[i, j] = (Male_Positions[i, j] + Female_Positions[i, j]) / 2
        convergence_curve[0, l] = Male_Jackal_score
        l = l + 1
    return Male_Jackal_score, Male_Jackal_pos, convergence_curve
