import torch
from torch import nn
import torch.nn.functional as F

from isegm.model.modeling.basic_blocks import ConvBNReLU, XConvBNReLU


############################# Residual Feedback Fusion ##############################
class CollaborativeFeedbackFusionModule(nn.Module):
    def __init__(self, in_ch, mid_ch, norm_layer=nn.BatchNorm2d):
        super(CollaborativeFeedbackFusionModule, self).__init__()
        self.feature_conv = ConvBNReLU(in_ch, mid_ch, kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
        self.fusion_encoder = nn.Sequential(
            ConvBNReLU(1 + mid_ch, mid_ch, norm_layer=norm_layer),
            XConvBNReLU(mid_ch, mid_ch, norm_layer),
            XConvBNReLU(mid_ch, mid_ch, norm_layer),
            nn.Conv2d(mid_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.residual_predictor = ConvBNReLU(in_ch+1, 1, kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)

    def forward(self, feature, feedback, gate, update_num=1):
        feedback = F.interpolate(feedback, feature.size()[2:], mode='bilinear', align_corners=True)
        
        for _ in range(update_num):
            feature_encoded = self.feature_conv(feature)
            fused_input = torch.cat((feedback, feature_encoded), dim=1)
            updated_feedback = self.fusion_encoder(fused_input)

            fused_feature = torch.cat((feature, updated_feedback), dim=1)
            residual = self.residual_predictor(fused_feature)

            feature = feature + gate * residual

        return feature, updated_feedback