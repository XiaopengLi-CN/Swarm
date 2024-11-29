import numpy as np  # 导入NumPy库，用于数值计算
from initialization import initialization_position  # 从initialization模块导入initialization_position函数
from levy import levy  # 从levy模块导入levy函数

"""
    优化思想的主体
    该函数实现了金豺优化算法（GJO），用于解决优化问题
"""

def GJO(search_agents_no, max_iteration, lb, ub, dim, fobj):
    """
    金豺优化算法（GJO）

    参数:
    search_agents_no: int - 种群的大小，即搜索代理的数量
    max_iteration: int - 最大迭代次数
    lb: array-like - 下界，定义每个维度的最小值
    ub: array-like - 上界，定义每个维度的最大值
    dim: int - 问题的维度，即每个搜索代理的位置向量的维度
    fobj: function - 目标函数，用于计算适应度

    返回:
    Male_Jackal_score: float - 雄性豺狼的最佳得分
    Male_Jackal_pos: array - 雄性豺狼的最佳位置
    convergence_curve: ndarray - 收敛曲线，记录每次迭代的最佳得分
    """
    # 初始化金豺对
    Male_Jackal_pos = [0 for i in range(dim)]  # 初始化雄性豺狼的位置
    Male_Jackal_score = np.inf  # 初始化雄性豺狼的得分为无穷大
    Female_Jackal_pos = [0 for i in range(dim)]  # 初始化雌性豺狼的位置
    Female_Jackal_score = np.inf  # 初始化雌性豺狼的得分为无穷大

    # 初始化个体的位置
    positions = initialization_position(search_agents_no, dim, ub, lb)  # 调用initialization_position函数初始化种群位置
    convergence_curve = np.zeros((1, max_iteration))  # 初始化收敛曲线数组
    # 循环的次数
    l = 0  # 初始化迭代计数器

    # 主循环
    while l < max_iteration:  # 当迭代次数小于最大迭代次数时
        for i in range(np.shape(positions)[0]):  # 遍历每个搜索代理
            # 边界检查
            flag4ub = positions[i, :] > ub  # 检查是否超过上界
            flag4lb = positions[i, :] < lb  # 检查是否低于下界
            # 边界检查为什么这样写，我在蛇优化中进行详细的讲解了。主要的思想就是把不合规的先置为0，之后大于边界的就设置为边界的最大值，小于边界的就设置为边界的最小值
            positions[i, :] = np.multiply(positions[i, :], (~(flag4ub + flag4lb))) + np.multiply(ub, flag4ub) + np.multiply(lb, flag4lb)  # 边界处理
            # 计算每一个个体的适应度
            fitness = fobj(positions[i, :])  # 计算适应度
            # 更新雄性豺狼
            if fitness < Male_Jackal_score:  # 如果当前适应度优于雄性豺狼的得分
                Male_Jackal_score = fitness  # 更新雄性豺狼的得分
                Male_Jackal_pos = positions[i, :]  # 更新雄性豺狼的位置
            if Male_Jackal_score < fitness < Female_Jackal_score:  # 如果当前适应度介于雄性和雌性豺狼的得分之间
                Female_Jackal_score = fitness  # 更新雌性豺狼的得分
                Female_Jackal_pos = positions[i, :]  # 更新雌性豺狼的位置
        E1 = 1.5 * (1 - (l / max_iteration))  # 计算逃避能量的系数
        RL = 0.05 * levy(search_agents_no, dim, 1.5)  # 计算Levy飞行步长
        Male_Positions = np.zeros((np.shape(positions)[0], np.shape(positions)[1]))  # 初始化雄性豺狼的位置矩阵
        Female_Positions = np.zeros((np.shape(positions)[0], np.shape(positions)[1]))  # 初始化雌性豺狼的位置矩阵
        for i in range(np.shape(positions)[0]):  # 遍历每个搜索代理
            for j in range(np.shape(positions)[1]):  # 遍历每个维度
                r1 = np.random.random()  # 生成随机数r1
                E0 = 2 * r1 - 1  # 计算逃避能量E0
                # 逃避的能量
                E = E1 * E0  # 计算逃避能量E
                if np.abs(E) < 1:  # 如果逃避能量的绝对值小于1
                    D_male_jackal = np.abs((RL[i, j] * Male_Jackal_pos[j] - positions[i, j]))  # 计算雄性豺狼的距离
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal  # 更新雄性豺狼的位置
                    D_female_jackal = abs((RL[i, j] * Female_Jackal_pos[j] - positions[i, j]))  # 计算雌性豺狼的距离
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal  # 更新雌性豺狼的位置
                else:  # 如果逃避能量的绝对值大于等于1
                    D_male_jackal = np.abs((Male_Jackal_pos[j] - RL[i, j] * positions[i, j]))  # 计算雄性豺狼的距离
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal  # 更新雄性豺狼的位置
                    D_female_jackal = abs((Female_Jackal_pos[j] - RL[i, j] * positions[i, j]))  # 计算雌性豺狼的距离
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal  # 更新雌性豺狼的位置
                positions[i, j] = (Male_Positions[i, j] + Female_Positions[i, j]) / 2  # 更新搜索代理的位置
        convergence_curve[0, l] = Male_Jackal_score  # 记录当前迭代的最佳得分
        l = l + 1  # 迭代计数器加1
    return Male_Jackal_score, Male_Jackal_pos, convergence_curve  # 返回雄性豺狼的最佳得分、最佳位置和收敛曲线