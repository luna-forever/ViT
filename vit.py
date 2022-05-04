import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import time
from torch.nn import functional as F
from math import floor, ceil
import math
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision.transforms import Compose, Resize, ToTensor

# import torchvision.transforms as transforms
device = torch.device('cpu')
print(device)

import random

# In[1] 设置超参数
num_epochs = 60
batch_size = 100
learning_rate = 0.001
global X_liner  # 定义特征收集器


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


# In[2] 金字塔池化
class SpatialPyramidPooling2d(nn.Module):
    def __init__(self, num_level, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type

    def forward(self, x):
        N, C, H, W = x.size()
        for i in range(self.num_level):
            level = i + 1
            kernel_size = (ceil(H / level), ceil(W / level))
            stride = (ceil(H / level), ceil(W / level))
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))
            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            if i == 0:
                res = tensor
            else:
                res = torch.cat((res, tensor), 1)
        return res


spp_layer = SpatialPyramidPooling2d(3)
# spp_layer.to(device)
# In[1] 加载数据
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./Fashion_MNIST', train=True, download=True,
                          transform=torchvision.transforms.Compose([
                              torchvision.transforms.RandomCrop(28, padding=2),
                              torchvision.transforms.RandomHorizontalFlip(p=0.5),
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                              RandomErasing(probability=0.1, mean=[0.4914]),
                          ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./Fashion_MNIST/', train=False, download=True,
                          transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=batch_size, shuffle=False)

# In[1] 加载模型

# In[1]我们需要用一个普通的线性层来投影它们
# 创建一个PatchEmbedding类来保持代码整洁
'''
注意：在检查了最初的实现之后，我发现为了提高性能，作者使用了Conv2d层而不是线性层,
这是通过使用与“patch_size”相等的kernel_size和stride 来获得的。
直观地说，卷积运算分别应用于每个切片。
因此，我们必须首先应用conv层，然后将生成的图像展平
'''


# In[1]下一步是添加cls标记和位置嵌入。cls标记只是每个序列（切片）中的一个数字。
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 用卷积层代替线性层->性能提升
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 先用大小为切片大小，步长为切片步长的卷积核来提取特征图，然后将特征图展平
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        # print(torch.randn((img_size // patch_size) **2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # print(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # 在输入前添加cls标记
        x = torch.cat([cls_tokens, x], dim=1)
        # print(self.positions.size())
        # 加位置嵌入
        # x += self.positions
        # print(x,x.size())
        return x


# print(PatchEmbedding()(x).shape)

# In[1]用nn.MultiHadAttention或实现我们自己的。为了完整起见，我将展示它的样子

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 将查询、键和值融合到一个矩阵中
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 分割num_heads中的键、查询和值
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # 最后一个轴上求和
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # 在第三个轴上求和
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# patches_embedded = PatchEmbedding()(x)
# print(MultiHeadAttention()(patches_embedded).shape)

# In[1]残差加法

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# In[1]反馈组件
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


# In[1]创建Transformer编码器块

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


# patches_embedded = PatchEmbedding()(x)
# print(TransformerEncoderBlock()(patches_embedded).shape)
# In[1]创建Transformer编码器

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


# In[1]分类器
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


# In[1]ViT架构
# 组成PatchEmbedding、TransformerEncoder和ClassificationHead来创建最终的ViT架构
class ViTNET(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 7,
                 emb_size: int = 768,
                 img_size: int = 28,
                 depth: int = 6,
                 n_classes: int = 10,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


model = ViTNET().to(device)
# In[1] 定义模型和损失函数
# [2,2,2]表示的是不同in_channels下的恒等映射数目

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[1] 设置一个通过优化器更新学习率的函数
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[1] 定义测试函数
def tes(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 清空特征收集器
            global X_liner
            X_liner = torch.empty((batch_size, 0), device=device)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# In[1] 训练模型更新学习率
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    in_epoch = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 清空特征收集器
        X_liner = torch.empty((batch_size, 0), device=device)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    tes(model, test_loader)
    out_epoch = time.time()
    print(f"use {(out_epoch - in_epoch) // 60}min{(out_epoch - in_epoch) % 60}s")
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

tes(model, train_loader)
torch.save(model.state_dict(), 'resnet.ckpt')
