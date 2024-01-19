import os
#
# 创建目录
# 存放结果文件
os.mkdir('../Image_Classification/output')

# 存放训练得到的模型权重
os.mkdir('../checkpoints')

# 存放生成的图表
os.mkdir('图表')

import matplotlib.pyplot as plt

# windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

plt.plot([1,2,3], [100,500,300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()
