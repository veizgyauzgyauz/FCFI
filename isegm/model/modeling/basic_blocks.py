import torch.nn as nn
import torch.nn.functional as F

from isegm.model import ops


class ConvHead(nn.Module):
    def __init__(self, out_channels, in_channels=32, num_layers=1,
                 kernel_size=3, padding=1,
                 norm_layer=nn.BatchNorm2d):
        super(ConvHead, self).__init__()
        convhead = []

        for i in range(num_layers):
            convhead.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding),
                nn.ReLU(),
                norm_layer(in_channels) if norm_layer is not None else nn.Identity()
            ])
        convhead.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))

        self.convhead = nn.Sequential(*convhead)

    def forward(self, *inputs):
        return self.convhead(inputs[0])


class SepConvHead(nn.Module):
    def __init__(self, num_outputs, in_channels, mid_channels, num_layers=1,
                 kernel_size=3, padding=1, dropout_ratio=0.0, dropout_indx=0,
                 norm_layer=nn.BatchNorm2d):
        super(SepConvHead, self).__init__()

        sepconvhead = []

        for i in range(num_layers):
            sepconvhead.append(
                SeparableConv2d(in_channels=in_channels if i == 0 else mid_channels,
                                out_channels=mid_channels,
                                dw_kernel=kernel_size, dw_padding=padding,
                                norm_layer=norm_layer, activation='relu')
            )
            if dropout_ratio > 0 and dropout_indx == i:
                sepconvhead.append(nn.Dropout(dropout_ratio))

        sepconvhead.append(
            nn.Conv2d(in_channels=mid_channels, out_channels=num_outputs, kernel_size=1, padding=0)
        )

        self.layers = nn.Sequential(*sepconvhead)

    def forward(self, *inputs):
        x = inputs[0]
        
        return self.layers(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_padding, dw_stride=1,
                 activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        _activation = ops.select_activation_function(activation)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel, stride=dw_stride,
                      padding=dw_padding, bias=use_bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            _activation()
        )

    def forward(self, x):
        return self.body(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class XConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d):
        super(XConvBNReLU, self).__init__()
        self.conv3x3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv1x1(self.conv3x3(x))))


def dilate(x, ksize=3):
    pad = (ksize - 1) // 2
    x = F.pad(x, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(x, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(x, ksize=3):
    out = 1 - dilate(1 - x, ksize)
    return out