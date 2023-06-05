import torch
import torch.nn as nn
import torch.nn.functional as f

__all__ = ['Cadblock']


class ConvBNPReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            (kernel_size, kernel_size),
            stride=stride,
            padding=(padding, padding),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-03)
        self.prelu = nn.PReLU(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        output = self.bn(output)
        output = self.prelu(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-03)
        self.prelu = nn.PReLU(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
           input: input feature map
           return: normalized and preluivated feature map
        """
        output = self.bn(x)
        output = self.prelu(output)
        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        """
        通道卷积，通道卷积是指在卷积过程中，对输入的每个通道进行独立的卷积操作，输出通道数与输入通道数相同。
        Args:
            in_planes: number of input channels
            out_planes: number of output channels, default (in_planes == out_planes)
            kernel_size: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            (kernel_size, kernel_size),
            stride=stride,
            padding=(padding, padding),
            groups=in_planes,
            bias=False
        )

    def forward(self, x):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(x)
        return output


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, d=1):
        """
        通道分离空洞卷积将输入张量的每个通道独立处理，因此需要将 groups=in_planes 设置为输入通道数，这样可以让每个通道都有自己的卷积核。
        通过设置 dilation 参数可以实现空洞卷积，从而进一步增加感受野大小。
        args:
           in_planes: number of input channels
           out_planes: number of output channels, default (in_planes == out_planes)
           kernel_size: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d  # 计算卷积核在当前的 dilation rate 下需要的 padding 数量
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            (kernel_size, kernel_size),
            stride=stride,
            padding=(padding, padding),
            groups=in_planes,
            bias=False,
            dilation=d
        )

    def forward(self, x):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(x)
        return output


class AdAvgPooling(nn.Sequential):
    # 自适应平均池化模块
    def __init__(self) -> None:
        super(AdAvgPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.avg_pool(x)
        return f.interpolate(x, size=size, mode='bilinear', align_corners=False)


class AdMaxPooling(nn.Sequential):
    # 自适应最大池化模块
    def __init__(self) -> None:
        super(AdMaxPooling, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.max_pool(x)
        return f.interpolate(x, size=size, mode='bilinear', align_corners=False)


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
        self.lrelu = nn.LeakyReLU()

        # print(self)

    def forward(self, x):
        a = self.conva(x)
        b = self.convb(x)
        m = self.mask_c(self.conv2(a * x + b))
        y = self.lrelu(self.inN(m))  # ax+b 计算结构
        return y


class Cadblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        n = int(out_planes / 4)

        # 先通过一个1*1卷积
        # self.conv = ConvBNPReLU(in_planes, n, 1, 1)  # 采用 1x1 Conv 来减少计算量
        self.chanleblock = ChanleBlock(in_planes, n)
        # 分组卷积，通道数=分组数，单通道卷积
        self.ChannelConv = ChannelWiseConv(n, n, 3, 1)  # local feature 在一定的局部区域内提取的图像特征
        # 通道扩张卷积，收集上下文信息
        self.ChannelConvD = ChannelWiseDilatedConv(n, n, 3, 1, d=2)
        self.ChannelConvD1 = ChannelWiseDilatedConv(n, n, 3, 1, d=4)
        self.ChannelConvD2 = ChannelWiseDilatedConv(n, n, 3, 1, d=6)
        # # 自适应平均池化+双线性插值
        # self.avgpooling = AdAvgPooling()
        # # 自适应最大池化+双线性插值
        # self.maxpooling = AdMaxPooling()
        # BN+ReLu
        self.bn_prelu = BNPReLU(out_planes)

    def forward(self, x):
        # y = self.conv(x)
        y = self.chanleblock(x)
        print("y", y.shape)
        y1 = self.ChannelConv(y)
        print("y1", y1.shape)
        y2 = self.ChannelConvD(y)
        print("y2", y2.shape)
        y3 = self.ChannelConvD1(y)
        print("y3", y3.shape)
        y4 = self.ChannelConvD2(y)
        print("y4", y4.shape)

        output = torch.cat([y1, y2, y3, y4], 1)
        output = self.bn_prelu(output)

        return output


if __name__ == '__main__':
    test = Cadblock(in_planes=2048, out_planes=2048)
    print(test)
    img = torch.randn((2, 2048, 64, 64))
    out = test(img)
    print(out.shape)
