from FitnessFunction import chung_reynolds  # 导入适应度函数
from SO import snake_optimization  # 导入蛇形优化算法
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图

# 定义维度
dim = 30  # 问题的维度，即解的变量个数
# 定义最大的迭代次数
max_iter = 1000  # 算法的最大迭代次数
# 定义种群的的大小
search_agents_no = 30  # 种群的大小，即搜索代理的数量
# 定义边界
solution_bound = [2, 100]  # 解的边界，表示每个变量的取值范围

# 调用蛇形优化算法进行优化
food, global_fitness, gene_best_fitness = snake_optimization(search_agents_no, max_iter, chung_reynolds, dim, solution_bound)

# 绘制每代的最佳适应度变化曲线
plt.plot([i for i in range(max_iter)], gene_best_fitness)
plt.xlabel('迭代次数')  # 设置x轴标签
plt.ylabel('最佳适应度')  # 设置y轴标签
plt.title('蛇形优化算法的收敛曲线')  # 设置图表标题
plt.show()  # 显示图表

# 输出最佳的解决方案和最佳适应度
print("最佳的解决方案：", food)
print("最佳适应度：", global_fitness)