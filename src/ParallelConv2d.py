import cv2 as cv
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

"""
其作用是将输入张量在通道维度上进行掩码处理，即将奇数通道位置上的元素乘以-1，偶数通道位置上的元素乘以1，
从而达到对通道维度上的信息进行交替处理的目的。
其输入为一个张量x，输出也是一个张量y，其中y的形状与x相同，但是在通道维度上经过了双向通道掩码处理。
"""


class Mask_1Chanle(nn.Module):  # 双向通道掩码
    def __init__(self, c):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones((1, c, 1, 1)))  # (1, c, 1, 1)
        self.register_parameter("p", self.p)

        for i in range(0, c, 2):  # 设置
            self.p.data[0, i] = -1

    def forward(self, x):
        y = self.p * x
        return y


"""
通道块的目的是为了通过通道之间的交互来增强特征的表达能力，
具体来说，通道块将输入张量 x 先分别经过两个1*1卷积核，得到输出张量a和b，
然后通过计算a*x + b 得到一个新的特征表示，再经过一个 1*1卷积核和一个实例归一化层得到最终的输出。

在通道块的计算过程中，双向通道掩码 Mask_1Chanle 起到了一个关键的作用。
具体来说，通过给定一个尺寸为 (1, c_out, 1, 1) 的参数张量p，
双向通道掩码将输入张量x与p逐元素相乘得到一个新的特征表示，
其中p的奇数通道位置上的元素为1，偶数通道位置上的元素为-1,这个特殊的参数设置可以促进通道之间的相互竞争，
从而更好地增强特征的表达能力。
"""

class ChanleBlock(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conva = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.convb = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.conv2 = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.mask_c = Mask_1Chanle(c_out)
        self.inN = nn.InstanceNorm2d(c_out)  # 通道归一化
        self.relu = nn.ReLU()

        # print(self)

    def forward(self, x):
        a = self.conva(x)
        b = self.convb(x)
        m = self.mask_c(self.conv2(a * x + b))
        y = self.relu(self.inN(m))  # ax+b 计算结构
        return y


class ParallelConv2D(nn.Module):
    def __init__(self, c_in, c_out,
                 kernel_size=3,
                 stride=1,
                 n=4,  # 平行卷积核数据数量
                 d=2,  # 多尺度空洞长度
                 DILATION=True
                 ):
        super().__init__()

        r = kernel_size // 2  # 计算卷积半径
        self.d = d  # 获取空洞卷积的基础空洞长度
        self.chanleblock = ChanleBlock(c_in, c_out)

        # 横向并行，x的n次方
        self.conv_as = nn.ModuleList([
            nn.Conv2d(
                c_out, c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=(d * i * r) if DILATION else r,
                dilation=(d * i) if DILATION else 1,  # 是否采用空洞多尺度
                groups=c_out,
                bias=False,
                padding_mode='reflect',
            )
            for i in range(1, n + 1)  # i=[1,....,n]
        ])

        self.conv_bs = nn.ModuleList([
            nn.Conv2d(
                c_out, c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=r if not DILATION else d * i * r,
                dilation=1 if not DILATION else d * i,  # 是否采用空洞多尺度
                groups=c_out,
                bias=False,
                padding_mode='reflect',
            )
            for i in range(1, n + 1)  # i=[1,....,n]
        ])

        self.inN = nn.InstanceNorm2d(c_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        y = self.chanleblock(x)  # (b, c_in, w, h) -> (b, c_out, w, h)

        pas = [conv(y) for conv in self.conv_as][::-1]  # 从高
        pbs = [conv(y) for conv in self.conv_bs][::-1]
        for a, b in zip(pas, pbs):
            y = self.lrelu(self.inN(a * y + b))  # ax + b 计算结构

        return y
"""
具体地说，pas 和 pbs 分别是 self.conv_as 和 self.conv_bs 中的所有卷积层对输入 x 的结果组成的列表，其长度为 n。
pas 和 pbs 中的元素从后往前遍历，表示卷积核的大小逐渐缩小。
接着，代码使用 zip 函数将 pas 和 pbs 中的对应元素按顺序配对，并在循环中对它们分别执行以下操作：
将 a * y + b 的结果传入 InstanceNorm2d 实例和 LeakyReLU 实例，分别进行实例归一化和激活函数操作。
将归一化和激活后的结果赋值给变量 y。
回到第一步，执行下一个配对元素的操作，直到所有的卷积操作都完成。
最终返回的 y 就是并行卷积的输出。
"""


def loadData():
    img = cv.imread('a.png')

    x = img.transpose(2, 0, 1)[np.newaxis,]
    x = torch.FloatTensor(x) / 256
    x = F.interpolate(x, (256, 256))

    return x


if __name__ == '__main__':
    block = ChanleBlock(2048, 1024)
    # print(test)
    img = torch.randn((2, 2048, 64, 64))
    out = block(img)
    print(out.shape)

