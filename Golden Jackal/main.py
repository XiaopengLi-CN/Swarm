import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from GetFunctionsDetails import get_functions_details  # 从GetFunctionsDetails模块导入get_functions_details函数
from GJO import GJO  # 从GJO模块导入GJO函数

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
search_agents_no = 50  # 定义种群的大小为50
# 迭代的次数
max_iteration = 3000  # 定义迭代的次数为3000

# 求解的具体的问题
function_name = "F1"  # 定义要求解的工程问题为F1
# 加载所选择的工程问题的具体的细节
lb, ub, dim, fobj = get_functions_details(function_name)  # 调用get_functions_details函数获取问题的细节，包括下界、上界、维度和目标函数
# GJO的最大重新运行次数
run_n = 5  # 定义GJO算法的最大运行次数为5
cost = np.zeros((run_n, 1))  # 初始化一个数组用于存储每次运行的成本，大小为(run_n, 1)
pos = np.zeros((run_n, 4))  # 初始化一个数组用于存储每次运行的最佳位置，大小为(run_n, 4)
for i in range(run_n):  # 循环运行GJO算法run_n次
    print("当前运行的是第", i, "次")  # 打印当前运行的次数
    male_jackal_score, male_jackal_pos, GJO_cg_courve = GJO(search_agents_no, max_iteration, lb, ub, dim, fobj)  # 调用GJO函数进行优化，返回最佳得分、最佳位置和收敛曲线
    cost[i, :] = male_jackal_score  # 将当前运行的最佳得分存储到cost数组中
mean_cost = np.mean(cost)  # 计算所有运行的平均成本
min_cost = np.min(cost)  # 计算所有运行的最小成本
max_cost = np.max(cost)  # 计算所有运行的最大成本
print("best value GJO：", min_cost, "mean：", mean_cost, "max_cost：", max_cost)  # 打印GJO算法的最佳值、平均值和最大值