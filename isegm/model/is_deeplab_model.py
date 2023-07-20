import torch
import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model_deeplab import ISModel
from .modeling.deeplab_v3 import DeepLabV3Plus
from .modeling.basic_blocks import SepConvHead
from isegm.model.modifiers import LRMult


class DeeplabModel(ISModel):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, mid_ch=64, aspp_dropout=0.5,
                 update_num=1, expansion_ratio=2.0,
                 backbone_norm_layer=None, backbone_lr_mult=0.1, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch, mid_ch=mid_ch,
                                               update_num=update_num, expansion_ratio=expansion_ratio,
                                               project_dropout=aspp_dropout, norm_layer=norm_layer, backbone_norm_layer=backbone_norm_layer)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))
        self.head = SepConvHead(1, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
                                num_layers=2, norm_layer=norm_layer)

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask
    
    def get_coord_features(self, image, points, designated_click_radius=-1):
        coord_features = self.dist_maps(image, points, designated_radius=designated_click_radius)

        return coord_features

    def backbone_forward(self, image, click_encoding, new_points, prev_mask, gate, coord_features=None, is_training=True):
        backbone_features = self.feature_extractor(image, click_encoding, new_points, prev_mask, gate, coord_features, is_training=is_training)

        if is_training:
            return {
                'instances': self.head(backbone_features[0]), 'updated_feedback1': backbone_features[1], 'updated_feedback2': backbone_features[3],
                'valid_masks': backbone_features[2]
            }
        else:
            return {
                'instances': self.head(backbone_features[0]), 'updated_feedback1': backbone_features[1], 'updated_feedback2': backbone_features[3],
            }


    def forward(self, image, points, new_points, gate, designated_click_radius=-1, is_training=True):
        image, prev_mask = self.prepare_input(image)
        gauss_dist_maps = self.get_coord_features(image, points, designated_click_radius)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, gauss_dist_maps), dim=1))
            outputs = self.backbone_forward(x, gauss_dist_maps, new_points, prev_mask, gate, is_training=is_training)
        else:
            coord_features = self.maps_transform(gauss_dist_maps)
            outputs = self.backbone_forward(image, gauss_dist_maps, new_points, prev_mask, gate, coord_features, is_training=is_training)

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)

        outputs['updated_feedback1'] = nn.functional.interpolate(outputs['updated_feedback1'], size=image.size()[2:],
                                                         mode='bilinear')
        if is_training:
            outputs['valid_masks'] = nn.functional.interpolate(outputs['valid_masks'], size=image.size()[2:],
                                                            mode='nearest')
        
        outputs['updated_feedback2'] = nn.functional.interpolate(outputs['updated_feedback2'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
                                                         
        if 'memories' in outputs.keys():
            outputs['memories'] = [nn.functional.interpolate(x, size=image.size()[2:],
                                                                mode='bilinear', align_corners=True) for x in outputs['memories']]
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        outputs['gauss_dist_maps'] = gauss_dist_maps
        return outputs

