import numpy as np  # 导入NumPy库，用于数值计算

def levy(n, m, beta):
    # 分子
    num = np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2)  # 计算分子部分，使用gamma分布和sin函数
    # 分母
    den = np.random.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))  # 计算分母部分，使用gamma分布和beta值
    sigma_u = (num / den)**(1 / beta)  # 计算sigma_u值
    u = np.random.normal(0, sigma_u, (n, m))  # 生成服从正态分布的随机数u
    v = np.random.normal(0, 1, (n, m))  # 生成服从正态分布的随机数v
    z = np.multiply(u, np.power(np.abs(v), 1/beta))  # 计算Levy分布的随机数z
    return z  # 返回生成的Levy分布的随机数