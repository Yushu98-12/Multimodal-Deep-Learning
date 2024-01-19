import math
from sklearn.metrics import f1_score

def calculate_loss_from_accuracy(accuracy):
    # 假设使用二分类问题，Loss为二元交叉熵损失函数
    # 假设正例和负例的样本数量相等
    # 假设正例的准确率等于Accuracy
    # 假设负例的准确率等于1 - Accuracy
    # 根据这些假设，计算Loss
    loss = - (0.5 * accuracy * math.log(accuracy) + 0.5 * (1 - accuracy) * math.log(1 - accuracy))
    return loss

def calculate_f1_from_accuracy(accuracy):
    # 假设使用二分类问题，根据Accuracy估算F1值
    # 假设正例的召回率等于Accuracy
    # 假设负例的召回率等于1 - Accuracy
    # 根据这些假设，计算F1值
    f1 = 2 * (0.5 * accuracy) / (0.5 * accuracy + 0.5 * (1 - accuracy))
    return f1

# 示例数据
accuracy = 0.7617

# 估算Loss和F1值
loss = calculate_loss_from_accuracy(accuracy)
f1 = calculate_f1_from_accuracy(accuracy)

print("Estimated Loss:", loss)
print("Estimated F1:", f1)
