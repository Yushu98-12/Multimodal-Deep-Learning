import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ModifiedDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedDenseNet121, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        self.densenet.fc = nn.Linear(1024, num_classes)  # Replace the classifier

        self.attention = SEBlock(1024)  # Add SEBlock for attention

    def forward(self, x):
        features = self.densenet.features(x)
        features = self.attention(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.fc(out)
        return out
