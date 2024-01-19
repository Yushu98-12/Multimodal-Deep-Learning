import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

# 导入中文字体，指定字号
font = ImageFont.truetype('..\SimHei.ttf', 32)

# 载入类别
idx_to_labels = np.load('..\idx_to_labels.npy', allow_pickle=True).item()

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# 导入训练好的模型
model = torch.load('..\checkpoints\ResNet101-best_1-1.000.pth', map_location=device)
model = model.eval()

# 预处理
from torchvision import transforms

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 加载图像并进行预测
img_path = '../Image/Test_Image/001.jpg'
img_pil = Image.open(img_path)
input_img = test_transform(img_pil).unsqueeze(0).to(device)
pred_logits = model(input_img)
pred_softmax = F.softmax(pred_logits, dim=1)

# 模拟SPI计算
# 假设有12个月份的降水和蒸散指数数据
# 这里使用随机数据模拟
np.random.seed(0)
precipitation_data = np.random.randint(0, 100, size=12)
evaporation_data = np.random.randint(0, 100, size=12)

# 计算SPEI
window_size = 3
spi_data = []
for i in range(window_size, len(precipitation_data)):
    spi = (precipitation_data[i] - np.mean(precipitation_data[i-window_size:i])) / np.std(precipitation_data[i-window_size:i])
    spi_data.append(spi)

spi_data = np.array(spi_data)

# 将SPEI与模型预测结果进行加权求和
spi_weights = np.array([0.7, 0.3])  # 设置SPEI的权重
weights = np.concatenate((pred_softmax.cpu().detach().numpy()[0], spi_weights))
weights /= np.sum(weights)
combined_probs = np.sum(weights.reshape(-1, 1) * np.concatenate((pred_softmax.cpu().detach().numpy()[0], spi_data)).reshape(1, -1), axis=1)

# 正则化结果，使其在0到1内并和为1
combined_probs = combined_probs / np.sum(combined_probs)
combined_probs = np.clip(combined_probs, 0, 1)  # 将结果限制在0到1之间

# 将结果转换为百分比形式
combined_probs *= 100




# 可视化加权结果
# plt.figure(figsize=(22, 10))
# x = list(idx_to_labels.values()) + ['SPI']
# y_pred = pred_softmax.cpu().detach().numpy()[0] * 100
# width = 0.45
# y_spi = np.array([spi_data] * len(x))
# y_combined = np.concatenate((y_pred, y_spi), axis=0)
# ax = plt.bar(x, y_combined, width)
# plt.bar_label(ax, fmt='%.2f', fontsize=15)
# plt.tick_params(labelsize=14)
# plt.title('图像分类预测结果', fontsize=18)
# plt.xticks(rotation=45)
# plt.xlabel('类别', fontsize=18)
# plt.ylabel('置信度', fontsize=18)
# plt.show()





# 绘制图像和加权结果
draw = ImageDraw.Draw(img_pil)
for i, (class_id, prob) in enumerate(zip(idx_to_labels.keys(), combined_probs)):
    class_name = idx_to_labels[class_id]
    confidence = prob * 100
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    draw.text((50, 100 + 50 * i), text, font=font, fill=(0, 0, 0, 1))

# 显示图像
img_pil.show()
