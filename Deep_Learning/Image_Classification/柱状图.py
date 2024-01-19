import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

from PIL import Image, ImageFont, ImageDraw
# 导入中文字体，指定字号
font = ImageFont.truetype('..\SimHei.ttf', 32)

# 定义数据
SPEI = [0.04,0.11,0.18,0.27,0.40]
DenseNet = [0.02,0.09,0.13,0.21,0.76]
S_DNet = [0.034, 0.104, 0.165, 0.252, 0.508]

# 定义x轴标签
x = np.arange(len(SPEI))

# 绘制柱状图
plt.bar(x, SPEI, width=0.2, align='center', label='SPEI')
plt.bar(x + 0.2, DenseNet, width=0.2, align='center', label='DenseNet')
plt.bar(x + 0.4, S_DNet, width=0.2, align='center', label='S-DNet')

# 添加标签
# plt.xlabel('干旱类型')
plt.xlabel('Drought type')
#plt.ylabel('概率')
plt.ylabel('Probability')
#plt.title('不同模态下干旱概率对比图')
plt.title('Comparison of drought probabilities under different modes')
#plt.xticks(x + 0.2, ['适宜', '轻旱', '中旱', '重旱','特旱'])
plt.xticks(x + 0.2, ['OM', 'LD', ' MD', 'SD','ED'])
plt.legend()

# 设置y轴范围为0到1
plt.ylim(0, 1)

# 显示图形
plt.show()
