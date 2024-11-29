
import numpy as np


def levy(n, m, beta):
    # 分子
    num = np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    # 分母
    den = np.random.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))
    sigma_u = (num / den)**(1 / beta)
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    z = np.multiply(u, np.power(np.abs(v), 1/beta))
    return z
