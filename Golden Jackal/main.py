import numpy as np
import matplotlib.pyplot as plt
from GetFunctionsDetails import get_functions_details
from GJO import GJO

"""
    金豹优化的代码：
    涉及到以下6个工程优化问题：
        'F1'拉/压弹簧设计；
        'F2' %压力容器设计
        'F3' %焊接梁设计
        'F4' %减速器设计
        'F5' 齿轮系设计问题
        'F6' 三杆桁架设计问题
"""

# 种群的大小
search_agents_no = 50
# 迭代的次数
max_iteration = 3000

# 求解的具体的问题
function_name = "F1"
# 加载所选择的工程问题的具体的细节
lb, ub, dim, fobj = get_functions_details(function_name)
# GJO的最大重新运行次数
run_n = 5
cost = np.zeros((run_n, 1))
pos = np.zeros((run_n, 4))
for i in range(run_n):
    print("当前运行的是第", i, "次")
    male_jackal_score, male_jackal_pos, GJO_cg_courve = GJO(search_agents_no, max_iteration, lb, ub, dim, fobj)
    cost[i, :] = male_jackal_score
mean_cost = np.mean(cost)
min_cost = np.min(cost)
max_cost = np.max(cost)
print("best value GJO：", min_cost, "mean：", mean_cost, "max_cost：", max_cost)


