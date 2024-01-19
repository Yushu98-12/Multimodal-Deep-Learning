import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 载入类别名称和ID
idx_to_labels = np.load('../idx_to_labels_eng.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)

# 载入测试集预测结果表格
df = pd.read_csv('测试集预测结果.csv')

# 绘制某一类别的ROC曲线
specific_class = 'SJP-MD'
# 二分类标注
y_test = (df['标注类别名称'] == specific_class)
# 二分类置信度
y_score = df['SJP-MD-预测置信度']

from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, y_score)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, linewidth=5, label=specific_class)
plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='Stochastic model')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.rcParams['font.size'] = 22
plt.title('{} ROC curve  AUC:{:.3f}'.format(specific_class, auc(fpr, tpr)))
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend()
plt.grid(True)

# plt.savefig('{}-ROC曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
# plt.show()

# yticks = ax.yaxis.get_major_ticks()
# yticks[0].label1.set_visible(False)

auc(fpr, tpr)

# 绘制所有类别的ROC曲线
from matplotlib import colors as mcolors
import random
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']

def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg

get_line_arg()

plt.figure(figsize=(14, 10))
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='Stochastic model')

plt.title('ROC curve for each category')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.rcParams['font.size'] = 22
plt.grid(True)


# 从sklearn.metrics导入roc_curve和auc
from sklearn.metrics import roc_curve, auc

# 其余代码不变

# 创建空列表存储每个类别的FPR、TPR和AUC
all_fpr = []
all_tpr = []
all_auc = []
# 计算每个类别的ROC曲线并存储数据
max_length = 0
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df['{}-预测置信度'.format(each_class)])
    fpr, tpr, _ = roc_curve(y_test, y_score)
    if len(fpr) > max_length:
        max_length = len(fpr)  # 更新最大长度

# 对每个类别的ROC曲线执行插值填充，使其具有相同的长度
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df['{}-预测置信度'.format(each_class)])
    fpr, tpr, _ = roc_curve(y_test, y_score)
    if len(fpr) < max_length:  # 如果长度不足，进行插值填充
        fpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(fpr)), fpr)
        tpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(tpr)), tpr)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(auc(fpr, tpr))

# 计算平均的FPR和TPR
mean_fpr = np.mean(all_fpr, axis=0)
mean_tpr = np.mean(all_tpr, axis=0)

# 计算平均AUC
mean_auc = np.mean(all_auc)

# 绘制整体ROC曲线
plt.figure(figsize=(14, 10))
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC Curve (AUC = {mean_auc:.3f})')

# 绘制对角线，表示随机分类的情况
plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--', label='Stochastic model')

# 设置图例、标题和坐标轴标签
plt.legend(loc='lower right', fontsize=12)
plt.title('Mean ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# 保存图像
plt.savefig('平均ROC曲线.pdf', dpi=120, bbox_inches='tight')



# 假设您有四个模型，模型名称分别为 model1, model2, model3, model4

# 假设您有四个模型，模型名称分别为 model1, model2, model3, model4
model_names = ['model1', 'model2', 'model3', 'model4']

# 创建一个空字典，用于存储不同模型的数据
data_dict = {}

# 计算并保存每个模型的数据
for i, model_name in enumerate(model_names):
    # 假设 mean_fpr、mean_tpr 和 mean_auc 是每个模型的数据
    data_dict[model_name] = {'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'mean_auc': mean_auc}

    # 保存数据到文件中
    np.save(f'mean_roc_data_{model_name}.npy', data_dict[model_name])

# 将所有数据保存到同一个文件中
np.save('all_mean_roc_data.npy', data_dict)


# 显示图形
plt.show()

# 其余代码不变

auc_list = []
for each_class in classes:
    y_test = list((df['标注类别名称'] == each_class))
    y_score = list(df['{}-预测置信度'.format(each_class)])
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, **get_line_arg(), label=each_class)
    plt.legend()
    auc_list.append(auc(fpr, tpr))

plt.legend(loc='best', fontsize=12)
plt.savefig('各类别ROC曲线（2）.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
plt.show()

# 将AUC增加至各类别准确率评估指标表格中
df_report = pd.read_csv('各类别准确率评估指标.csv')

# 计算 AUC值 的 宏平均 和 加权平均
macro_avg_auc = np.mean(auc_list)
weighted_avg_auc = sum(auc_list * df_report.iloc[:-2]['support'] / len(df))

auc_list.append(macro_avg_auc)
auc_list.append(weighted_avg_auc)

df_report['AUC'] = auc_list

# df_report.to_csv('所有类别的准确率评估指标.csv', index=False)
