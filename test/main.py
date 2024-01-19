import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 读取数据
data = pd.read_csv('data_and_results.csv')

# 提取特征和目标变量
features = data[['年份']]
target = data['气象干旱程度']

# 创建随机森林回归模型
model = RandomForestRegressor()
# 拟合模型
model.fit(features, target)

# 预测下一年的气象干旱程度
next_year = int(input("请输入下一年的年份："))
predicted_result = model.predict([[next_year]])
new_row = {'年份': next_year, '气象干旱程度': predicted_result[0]}

data = data.append(new_row, ignore_index=True)
data['年份'] = data['年份'].astype(int)
data.to_csv('data_and_results.csv', index=False)
print(f"预测{next_year}年的气象干旱程度为：{predicted_result[0]}")



import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
data = pd.read_csv('data_and_results.csv')

# 将年份列转换为整数类型
data['年份'] = data['年份'].astype(int)

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
# 创建子图
fig, ax = plt.subplots()

# 绘制历年的气象干旱程度
ax.plot(data['年份'], data['气象干旱程度'], label='历年气象干旱程度')

# 绘制预测结果
ax.scatter(data['年份'].iloc[-1], data['气象干旱程度'].iloc[-1], color='red', label='预测结果')

# 设置标题和轴标签
ax.set_title('历年气象干旱程度及预测结果')
ax.set_xlabel('年份')
ax.set_ylabel('气象干旱程度')

# 添加图例
ax.legend()
# 显示图形
plt.show()
