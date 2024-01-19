
def weighted_sum(arr1, arr2, w1, w2):
    # 检查输入数组长度是否一致
    if len(arr1) != len(arr2):
        return "Error: 输入数组长度不一致"

    # 计算加权求和
    result = []
    for i in range(len(arr1)):
        weighted_sum = arr1[i] * w1 + arr2[i] * w2
        result.append(weighted_sum)

    return result


# 输入两组数组和权重
arr1 = [0.02,0.09,0.13,0.21,0.76]
arr2 = [0.04,0.11,0.18,0.27,0.40]
w1 = 0.3
w2 = 0.7

# 调用函数进行加权求和
result = weighted_sum(arr1, arr2, w1, w2)

# 打印结果
print(result)
