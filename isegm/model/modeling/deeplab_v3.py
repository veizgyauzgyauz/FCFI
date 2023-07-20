from contextlib import ExitStack
from operator import inv
from pickletools import read_unicodestring4
from unittest import result
import numpy as np
import cv2
import math

import torch
from torch import nn
import torch.nn.functional as F

from .basic_blocks import SeparableConv2d
from .resnet import ResNetBackbone
from isegm.model import ops
from .basic_blocks import SepConvHead

from isegm.model.modeling.fcfi import FocusedCorrectionModule, CollaborativeFeedbackFusionModule


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone='resnet50',
                 norm_layer=nn.BatchNorm2d,
                 backbone_norm_layer=None,
                 ch=256,
                 mid_ch=64,
                 update_num=1,
                 expansion_ratio=2.0,
                 project_dropout=0.5,
                 inference_mode=False,
                 **kwargs):
        super(DeepLabV3Plus, self).__init__()
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer

        self.backbone_name = backbone
        self.norm_layer = norm_layer
        self.backbone_norm_layer = backbone_norm_layer
        self.inference_mode = False
        self.ch = ch
        self.aspp_in_channels = 2048
        self.skip_project_in_channels = 256  # layer 1 out_channels
        self.update_num = update_num
        self.expansion_ratio = expansion_ratio

        self._kwargs = kwargs
        if backbone == 'resnet34':
            self.aspp_in_channels = 512
            self.skip_project_in_channels = 64

        self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False,
                                       norm_layer=self.backbone_norm_layer, **kwargs)

        self.head = _DeepLabHead(in_channels=ch + 32, mid_channels=ch, out_channels=ch,
                                 norm_layer=self.norm_layer)
        self.skip_project = _SkipProject(self.skip_project_in_channels, 32, norm_layer=self.norm_layer)
        self.aspp = _ASPP(in_channels=self.aspp_in_channels,
                          atrous_rates=[12, 24, 36],
                          out_channels=ch,
                          project_dropout=project_dropout,
                          norm_layer=self.norm_layer)

        self.correction_module = FocusedCorrectionModule(
            in_ch=ch, mid_ch=mid_ch,
            norm_layer=self.norm_layer,
            expan_ratio=expansion_ratio,
        )
        self.feedback_fusion = CollaborativeFeedbackFusionModule(in_ch=ch, mid_ch=mid_ch, norm_layer=self.norm_layer)

        if inference_mode:
            self.set_prediction_mode()

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True,
                                    norm_layer=self.backbone_norm_layer, **self._kwargs)
        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = pretrained.state_dict()

        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict)

        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def set_prediction_mode(self):
        self.inference_mode = True
        self.eval()

    def forward(self, input, click_encoding, new_points, prev_output, gate, additional_features=None, is_training=True):
        """
        new_points (List): Each element in it is a list which contains three elements, i.e., y, x, and is_positive.
        """
        with ExitStack() as stack:
            if self.inference_mode:
                stack.enter_context(torch.no_grad())
            c1, _, c3, c4 = self.backbone(input, additional_features)
            c1 = self.skip_project(c1)

            x = self.aspp(c4)
            x = F.interpolate(x, c1.size()[2:], mode='bilinear', align_corners=True)
            
            updated_feedback1, valid_masks = self.correction_module(input, click_encoding, x, prev_output, new_points, is_training)
            x, updated_feedback2 = self.feedback_fusion(x, updated_feedback1, gate, self.update_num)
            
            x = torch.cat((x, c1), dim=1)
            x = self.head(x)
        return x, updated_feedback1, valid_masks, updated_feedback2


class _SkipProject(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_SkipProject, self).__init__()
        _activation = ops.select_activation_function("relu")

        self.skip_project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            _activation()
        )

    def forward(self, x):
        return self.skip_project(x)


class _DeepLabHead(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=256, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()

        self.block = nn.Sequential(
            SeparableConv2d(in_channels=in_channels, out_channels=mid_channels, dw_kernel=3,
                            dw_padding=1, activation='relu', norm_layer=norm_layer),
            SeparableConv2d(in_channels=mid_channels, out_channels=mid_channels, dw_kernel=3,
                            dw_padding=1, activation='relu', norm_layer=norm_layer),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256,
                 project_dropout=0.5, norm_layer=nn.BatchNorm2d):
        super(_ASPP, self).__init__()

        b0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU()
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        # b4 = _ASPPConv(in_channels, out_channels, rate4, norm_layer)
        b5 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.concurent = nn.ModuleList([b0, b1, b2, b3, b5])

        project = [
            nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels,
                      kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU()
        ]
        if project_dropout > 0:
            project.append(nn.Dropout(project_dropout))
        self.project = nn.Sequential(*project)

    def forward(self, x):
        x = torch.cat([block(x) for block in self.concurent], dim=1)

        return self.project(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        pool = self.gap(x)
        return F.interpolate(pool, x.size()[2:], mode='bilinear', align_corners=True)


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU()
    )

    return block