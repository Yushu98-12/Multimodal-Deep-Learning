from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image

# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 图像预处理
from torchvision import transforms

# # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
# train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                      ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

# 导入训练好的模型
model = torch.load('checkpoints/Drought_Wheat_23.3.30_best-0.990.pth')
model = model.eval().to(device)

# 抽取模型中间层输出结果作为语义特征
from torchvision.models.feature_extraction import create_feature_extractor
model_trunc = create_feature_extractor(model, return_nodes={'avgpool': 'semantic_feature'})

#计算单张图像的语义特征
img_path = 'Drought_Wheat_23.3.30/test/1Suitable/IMG_20230330_103404.jpg'
img_pil = Image.open(img_path)
input_img = test_transform(img_pil) # 预处理
input_img = input_img.unsqueeze(0).to(device)
# 执行前向预测，得到指定中间层的输出
pred_logits = model_trunc(input_img)

pred_logits['semantic_feature'].squeeze().detach().cpu().numpy().shape

# 载入测试集图像分类结果
df = pd.read_csv('../Algorithm evaluation result graph/AlexNet-best_1-0.558/测试集预测结果.csv')
# 计算测试集每张图像的语义特征
encoding_array = []
img_path_list = []

for img_path in tqdm(df['图像路径']):
    img_path_list.append(img_path)
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
    feature = model_trunc(input_img)['semantic_feature'].squeeze().detach().cpu().numpy() # 执行前向预测，得到 avgpool 层输出的语义特征
    encoding_array.append(feature)
encoding_array = np.array(encoding_array)

encoding_array.shape

# 保存为本地的.npy文件
np.save('测试集语义特征.npy', encoding_array)
