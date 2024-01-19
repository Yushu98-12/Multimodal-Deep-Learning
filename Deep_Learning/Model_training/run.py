import multiprocessing
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")

# windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 图像预处理
from torchvision import transforms

# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),      # 随机缩放裁剪
                                      transforms.RandomHorizontalFlip(),      # 图像增强（水平翻转）
                                      transforms.ToTensor(),                  # 把图片转成torch的tensor数据
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])                                       # 归一化的均值、标准差（约定俗成）

# 验证集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化            不需要图像增强
test_transform = transforms.Compose([transforms.Resize(256),         # 缩放成256*256的正方形图像
                                     transforms.CenterCrop(224),     # 从中心裁剪成224*224的正方形图像
                                     transforms.ToTensor(),          # 把图片转成torch的tensor数据
                                     transforms.Normalize(           # 归一化
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

# 《载入图像分类数据集》

# 数据集文件夹路径
dataset_dir = '..\Image\Train_Image'
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'test')

print('训练集路径', train_path)
print('测试集路径', test_path)

from torchvision import datasets

# 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)     #传入训练集处理方式

# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)        #传入测试集处理方式

print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)

print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)

# 2023年6月29日17:30:34
# 类别和索引号 一一对应
# 各类别名称
class_names = train_dataset.classes
n_class = len(class_names)
class_names
# 映射关系：类别 到 索引号
train_dataset.class_to_idx
# 映射关系：索引号 到 类别
idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
idx_to_labels
# 保存为本地的 npy 文件
np.save('../idx_to_labels_eng.npy', idx_to_labels)
np.save('../labels_to_idx_eng.npy', train_dataset.class_to_idx)


# 定义数据加载器 DataLoader

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    BATCH_SIZE = 16

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,  # 是否随机打乱
                              num_workers=4  # 有几个CPU内核工作
                             )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4
                            )

    # DataLoader 是 python生成器，每次调用返回一个 batch 的数据
    images, labels = next(iter(train_loader))

    images.shape
    # torch.Size([16,3,224,224])
    # 意思是16张3通道 224*224的图像
    labels


    # 《可视化一个batch的图像和标注》
    # 演示单张图片使用

    # 将数据集中的Tensor张量转为numpy的array数据类型
    # images = images.numpy()
    # images[5].shape
    # plt.hist(images[5].flatten(), bins=50)
    # plt.show()
    #
    # # batch 中经过预处理的图像
    # idx = 2
    # plt.imshow(images[idx].transpose((1,2,0))) # 转为(224, 224, 3)
    # plt.title('label:'+str(labels[idx].item()))
    #
    # label = labels[idx].item()
    # label
    #
    # pred_classname = idx_to_labels[label]
    # pred_classname
    #
    # # 还原原始图像
    #
    # idx = 2
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # plt.imshow(np.clip(images[idx].transpose((1,2,0)) * std + mean, 0, 1))
    # plt.title('label:'+ pred_classname)
    # plt.show()


    # 导入训练需使用的工具包
    from torchvision import models
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from Model.densenet_with_attention import ModifiedDenseNet121

# 如何选择，取决于训练数据集与ImageNet数据集的差异

#选择一：只微调训练模型最后一层（全连接分类层）
    model = models.resnet18(pretrained=True) # 载入预训练模型

    # 修改全连接层，使得全连接层的输出与当前数据集类别数对应
    # 新建的层默认 requires_grad=True

    model.fc = nn.Linear(model.fc.in_features, n_class)
    model.fc
    # 只微调训练最后一层全连接层的参数，其它层冻结

    optimizer = optim.Adam(model.fc.parameters())  #有全连接层



#选择二：微调训练所有层

    # model = models.resnet50(pretrained=True) # 载入预训练模型       # “pretrained=True” 仍使用户部分权重，只是微调，赢在起跑线上
    #
    # model.fc = nn.Linear(model.fc.in_features, n_class)
    #
    # optimizer = optim.SGD(model.parameters(),lr=0.001)                    # ”model.parameters“表示所有层

    # 选择二 添加densenet121注意力机制代码
    # model = ModifiedDenseNet121(num_classes=12)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #选择三：随机初始化模型全部权重，从头训练所有层
    # 训练的数据集与ImageNet数据集完全不一样：大小、光照、量子显微镜、哈勃望远镜

    # model = models.resnet50(pretrained=False) # 只载入模型结构，不载入预训练权重参数

    # model.fc = nn.Linear(model.fc.in_features, n_class)

    # optimizer = optim.SGD(model.parameters(),lr=0.001)

#训练配置
    model = model.to(device)

    # 交叉熵损失函数
    # https://blog.csdn.net/m0_56654441/article/details/120572083 助于理解
    criterion = nn.CrossEntropyLoss()

    # 训练轮次 Epoch
    EPOCHS = 10

    # 学习率降低策略
    # “step_size=5, gamma=0.5” 每5个epoch，降低0.5
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 函数： 在训练集上测试
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score


    def train_one_batch(images, labels):
        '''运行一个 batch 的训练，返回当前 batch 的训练日志'''

        # 获得一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # 输入模型，执行前向预测
        loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

        # 优化更新权重
        # 反向传播“三部曲”
        optimizer.zero_grad()  # 清除梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 优化更新

        # 获取当前 batch 的标签类别和预测类别
        _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        log_train = {}
        log_train['Epoch'] = epoch
        log_train['Batch'] = batch_idx
        # 计算分类评估指标
        log_train['Train_Loss'] = loss
        log_train['Train_Accuracy'] = accuracy_score(labels, preds)
        # log_train['train_precision'] = precision_score(labels, preds, average='macro')
        # log_train['train_recall'] = recall_score(labels, preds, average='macro')
        # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

        return log_train

# 函数：在整个验证集上评估
    def evaluate_testset():
        '''在整个验证集上评估，返回分类评估指标日志'''

        loss_list = []
        labels_list = []
        preds_list = []

        with torch.no_grad():
            for images, labels in test_loader:  # 生成一个 batch 的数据和标注
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 输入模型，执行前向预测

                # 获取整个验证集的标签类别和预测类别
                _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
                preds = preds.cpu().numpy()
                loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
                loss = loss.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(preds)

        log_test = {}
        log_test['Epoch'] = epoch

        # 计算分类评估指标
        log_test['Test_Loss'] = np.mean(loss)
        log_test['Test_Accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['Test_Precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['Test_Recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['Test_F1-score'] = f1_score(labels_list, preds_list, average='macro')

        return log_test


    # 训练开始之前，记录日志
    epoch = 0
    batch_idx = 0
    best_test_accuracy = 0

    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    log_train = {}
    log_train['Epoch'] = 0
    log_train['Batch'] = 0
    images, labels = next(iter(train_loader))
    log_train.update(train_one_batch(images, labels))
    df_train_log = df_train_log.append(log_train, ignore_index=True)

    df_train_log

    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    log_test = {}
    log_test['Epoch'] = 0
    log_test.update(evaluate_testset())
    df_test_log = df_test_log.append(log_test, ignore_index=True)

    df_test_log

#创建wandb可视化项目
    import wandb
    # 添加个人账户
    # os.environ["WANDB_API_KEY"] = 'b94b0101e0579a3a7d3321ad7a8c47d5dc15e7fa'
    wandb.init(project='小麦干旱检测2023年10月', name=time.strftime('%Y-%m%d-%H%M'))

# 运行训练
    for epoch in range(1, EPOCHS + 1):

        print(f'Epoch {epoch}/{EPOCHS}')

        ## 训练阶段
        model.train()
        for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
            batch_idx += 1
            log_train = train_one_batch(images, labels)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
            wandb.log(log_train)

        # 渐变学习率
        # scheduler.step()

        ## 验证阶段
        model.eval()
        log_test = evaluate_testset()
        df_test_log = df_test_log.append(log_test, ignore_index=True)
        wandb.log(log_test)


        # 保存最新的最佳模型文件
        # if log_test['Test_Accuracy'] > best_test_accuracy:
        #     # 删除旧的最佳模型文件(如有)
        #     old_best_checkpoint_path = '../checkpoints/resnet50-best-{:.3f}.pth'.format(best_test_accuracy)
        #     if os.path.exists(old_best_checkpoint_path):
        #         os.remove(old_best_checkpoint_path)
        #     # 保存新的最佳模型文件
        #     new_best_checkpoint_path = '../checkpoints/resnet50-best-{:.3f}.pth'.format(log_test['Test_Accuracy'])
        #     torch.save(model, new_best_checkpoint_path)
        #     print('保存新的最佳模型', '../checkpoints/resnet50-best-{:.3f}.pth'.format(best_test_accuracy))
        #     best_test_accuracy = log_test['Test_Accuracy']

    df_train_log.to_csv('resnet50-训练日志-训练集.csv', index=False)
    df_test_log.to_csv('resnet50-训练日志-测试集.csv', index=False)


    # 在测试集上评价
    # 载入最佳模型作为当前模型
    # model = torch.load('../checkpoints/resnet50-best-{:.3f}.pth'.format(best_test_accuracy))
    # model.eval()
    # print(evaluate_testset())

    # 在测试集上评价
    # 载入最佳模型作为当前模型
    model = torch.load('../checkpoints/resnet50.pth')
    model.eval()
    print(evaluate_testset())
