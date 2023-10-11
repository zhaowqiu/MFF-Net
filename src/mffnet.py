from collections import OrderedDict
from typing import Dict
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.backbone import resnet50
from src.cbmablock import cbam_block
from src.ParallelConv2d import ChanleBlock

"""
1 采取改进后的注意力机制监督*1
2 channnel
3 采用深度可分离卷积减少参数
4 损失函数，使用深监督                                                           
 """

class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    stride=1,
                                    kernel_size=kernel_size,
                                    padding=3,
                                    groups=in_ch,
                                    padding_mode='reflect')
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    padding_mode='reflect')

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class upConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = DepthWiseConv(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = F.interpolate(x, size=(2 * h, 2 * w), mode='bilinear', align_corners=False)
        x = self.relu(self.bn(self.conv(x)))
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__(in_ch, out_ch, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = DepthWiseConv(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


def upsample(src, tar):
    _, _, h, w = tar.shape
    # 把自己上采样和目标特征图一样
    src = F.interpolate(src, size=(h, w), mode='bilinear', align_corners=True)
    return src


class U1(nn.Module):
    def __init__(self, in_channels):
        super(U1, self).__init__()
        # in_channels:2048, out_channels:256
        self.conv = DownConvBNReLU(in_channels, in_channels // 2)
        self.conv1 = DownConvBNReLU(in_channels // 2, in_channels // 8)
        self.convup = upConvBNReLU(in_channels // 8, in_channels // 8)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x)
        x = self.convup(x)
        x = self.convup(x)

        return x


class U4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U4, self).__init__()
        # in_channels:256,out_channels:256
        # self.channle = ChanleBlock(in_channels, in_channels)
        # in_channels:256,out_channels:32
        self.conv = ConvBNReLU(in_channels, out_channels)
        # in_channels:256,out_channels:256
        self.convdown = DownConvBNReLU(out_channels, out_channels)
        # in_channels:256,out_channels:256
        self.convup = upConvBNReLU(out_channels, out_channels)

    def forward(self, x):
        # x = self.channle(x)
        x = self.conv(x)
        x = self.convdown(x)
        x = self.convdown(x)

        x = self.convup(x)
        x = self.convup(x)
        return x


class Mff_netx(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super(Mff_netx, self).__init__()

        self.backbone = backbone

        self.u1 = U1(in_channels=2048)
        self.u2 = U1(in_channels=1024)
        self.u3 = U1(in_channels=512)
        self.u4 = U4(in_channels=256, out_channels=32)

        self.cbam1 = cbam_block(channel=256)
        self.downchannel1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, padding=0)

        self.cbam2 = cbam_block(channel=128)
        self.downchannel2 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, padding=0)

        self.cbam3 = cbam_block(channel=64)
        self.downchannel3 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0)

        self.cbam4 = cbam_block(channel=32)
        self.downchannel4 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, padding=0)

        self.side1 = nn.Conv2d(256, num_classes, kernel_size=(3, 3), padding=1)
        self.side2 = nn.Conv2d(128, num_classes, kernel_size=(3, 3), padding=1)
        self.side3 = nn.Conv2d(64, num_classes, kernel_size=(3, 3), padding=1)
        self.side4 = nn.Conv2d(32, num_classes, kernel_size=(3, 3), padding=1)
        self.out_conv = nn.Conv2d(8, num_classes, kernel_size=(1, 1))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x
        # contract: features is a dict of tensors
        features = self.backbone(x)  # 通过backbone提取特征图
        # result = OrderedDict()
        x_layer1 = features["out1"]  # [128*128*256]
        x_layer2 = features["out2"]  # [64*64*512]
        x_layer3 = features["out3"]  # [64*64*1024]
        x_layer4 = features["out4"]  # [64*64*2048]

        xcbam1 = self.downchannel1(x_layer4)
        xcbam1 = self.cbam1(xcbam1)

        xcbam2 = self.downchannel2(x_layer3)
        xcbam2 = self.cbam2(xcbam2)

        xcbam3 = self.downchannel3(x_layer2)
        xcbam3 = self.cbam3(xcbam3)

        xcbam4 = self.downchannel4(x_layer1)
        xcbam4 = self.cbam4(xcbam4)

        xlyer1 = self.u1(x_layer4)  # [64*64*256]
        xlyer2 = self.u2(x_layer3)  # [128*128*128]
        xlyer3 = self.u3(x_layer2)  # [256*256*64]
        xlyer4 = self.u4(x_layer1)  # [256*256*32]

        # side output
        side_outputs = []


        # xcbam1 = upsample(xcbam1, input_shape)
        xsup1 = xcbam1 + xlyer1
        sup1 = self.side1(xsup1)
        sup1 = upsample(sup1, input_shape)
        side_outputs.insert(0, sup1)

        # xcbam2 = upsample(xcbam2, xlyer2)
        xsup2 = xcbam2 + xlyer2
        sup2 = self.side2(xsup2)
        sup2 = upsample(sup2, input_shape)
        side_outputs.insert(0, sup2)

        # xcbam3 = upsample(xcbam3, xlyer3)
        xsup3 = xcbam3 + xlyer3
        sup3 = self.side3(xsup3)
        sup3 = upsample(sup3, input_shape)
        side_outputs.insert(0, sup3)

        # xcbam4 = upsample(xcbam4, xlyer4)
        xsup4 = xcbam4 + xlyer4
        sup4 = self.side4(xsup4)
        sup4 = upsample(sup4, input_shape)
        side_outputs.insert(0, sup4)

        sup0 = self.out_conv(torch.cat((sup1, sup2, sup3, sup4), 1))
        side_outputs.insert(0, sup0)

        if self.training:
            # do not use torch.sigmoid for amp safe
            return side_outputs
        else:
            return sup0


def Mffnetx(pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    # 这里指定了返回 layer1, layer2, layer3, layer4 这几层的输出，其中 layer3 这一层的输出被命名为 "out3"。
    # 这些层的输出会被作为一个字典返回，字典的键是指定的层名，值是对应层的输出。
    return_layers = {'layer4': 'out4', 'layer1': 'out1', 'layer2': 'out2', 'layer3': 'out3'}

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = Mff_netx(backbone)

    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mffnetx().to(device).eval()
    # print(test)
    img = torch.randn(2, 3, 512, 512).to(device)
    out = model(img)[0]
    print(out.shape)

    from thop import clever_format, profile

    flops, params = profile(model, (img,), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))