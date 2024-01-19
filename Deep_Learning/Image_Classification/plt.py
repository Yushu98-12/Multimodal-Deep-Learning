import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

from PIL import Image, ImageFont, ImageDraw
# 导入中文字体，指定字号
font = ImageFont.truetype('..\SimHei.ttf', 32)

# 读取CSV文件
csv_file = 'pred_results.csv'  # 替换为实际的CSV文件路径
data = pd.read_csv(csv_file)

# 提取数据列
class_names = data['Class']
class_ids = data['Class_ID']
confidence = data['Confidence(%)']

# 绘制柱状图
plt.figure(figsize=(20, 10))
plt.bar(class_names, confidence)
plt.xlabel('类别', fontsize=18)
plt.ylabel('置信度', fontsize=18)
plt.title('图像分类预测结果', fontsize=18)
plt.xticks(rotation=0)  # 如果类别名较长，可以旋转 x 轴标签
plt.tight_layout()
plt.tick_params(labelsize=16) # 坐标文字大小

plt.show()
