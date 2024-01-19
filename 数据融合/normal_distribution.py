# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# # 创建一个空列表来保存数值
# data_list = []
#
# # 输入数据
# data = input("请输入若干个数值，用空格分隔：")
# data = list(map(float, data.split()))
#
# # 将输入的数值添加到列表中
# data_list.extend(data)
#
# # 计算均值和标准差
# mean = np.mean(data_list)
# std = np.std(data_list)
#
# # 生成正态分布的x轴数据
# x = np.linspace(mean - 3*std, mean + 3*std, 100)
#
# # 计算正态分布的概率密度函数
# pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))
#
# # 创建一个包含列表数据的DataFrame
# df = pd.DataFrame({'数值': data_list})
#
# # 保存DataFrame为CSV文件
# df.to_csv('data_list.csv', index=False)
# # print("列表已保存为data_list.csv文件")
#
# # 绘制正态分布曲线图
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# plt.plot(x, pdf)
# plt.xlabel('数值')
# plt.ylabel('概率密度')
# plt.title('正态分布曲线图')
# plt.grid(True)
# plt.show()
#
# # 输入新的数值
# new_value = float(input("请输入一个新的数值："))
#
# # 读取列表中的数值计算新的均值和标准差
# data_list.append(new_value)
# mean = np.mean(data_list)
# std = np.std(data_list)
#
# # 计算新数值在不同区间的概率
# prob_less_than_zero = np.sum(pdf[x < 0])
# prob_between_zero_and_two = np.sum(pdf[(x >= 0) & (x <= 2)])
# prob_between_two_and_three = np.sum(pdf[(x > 2) & (x <= 3)])
# prob_greater_than_three = np.sum(pdf[x > 3])
#
# # 创建数据表格
# data_table = pd.DataFrame({
#     '区间': ['新数值小于0', '新数值在0到2之间', '新数值在2到3之间', '新数值大于3'],
#     '概率': [prob_less_than_zero, prob_between_zero_and_two, prob_between_two_and_three, prob_greater_than_three]
# })
#
# # 导出数据表格为CSV文件
# data_table.to_csv('probability_distribution.csv', index=False)
#
# # 打印数据表格
# print(data_table)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取已存在的CSV文件
df = pd.read_csv('data_list.csv')

# 从DataFrame中提取数值列
data_list = df['数值'].tolist()

# 计算均值和标准差
mean = np.mean(data_list)
std = np.std(data_list)

# 生成正态分布的x轴数据
x = np.linspace(mean - 3*std, mean + 3*std, 100)

# 计算正态分布的概率密度函数
pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))

# 创建一个包含列表数据的DataFrame
df = pd.DataFrame({'数值': data_list})

# 保存DataFrame为CSV文件
df.to_csv('data_list.csv', index=False)
# print("列表已保存为data_list.csv文件")

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 绘制正态分布曲线图

# 指定支持中文的字体
plt.rcParams['font.family'] = 'SimHei, sans-serif'

plt.plot(x, pdf)
plt.xlabel('数值')
plt.ylabel('概率密度')
plt.title('正态分布曲线图')
plt.grid(True)
plt.show()

# 输入新的数值
new_value = float(input("请输入一个新的数值："))

# 将新值添加到列表
data_list.append(new_value)

# 重新计算均值和标准差
mean = np.mean(data_list)
std = np.std(data_list)

# 计算新数值在不同区间的概率
prob_less_than_zero = np.sum(pdf[x < 0])
prob_between_zero_and_two = np.sum(pdf[(x >= 0) & (x <= 2)])
prob_between_two_and_three = np.sum(pdf[(x > 2) & (x <= 3)])
prob_greater_than_three = np.sum(pdf[x > 3])

# 创建数据表格
data_table = pd.DataFrame({
    '区间': ['适宜', '轻旱', '中旱', '重旱'],
    '概率': [prob_less_than_zero, prob_between_zero_and_two, prob_between_two_and_three, prob_greater_than_three]
})


# 导出数据表格为CSV文件
data_table.to_csv('probability_distribution.csv', index=False)

# 打印数据表格
print(data_table)
