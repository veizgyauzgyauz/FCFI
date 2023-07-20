import torch
import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model_hrnet import ISModel
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult


class HRNetModel(ISModel):
    @serialize
    def __init__(self, width=48, ocr_width=256, mid_ch=32,
                 update_num=1, expansion_ratio=2.0, 
                 small=False, num_classes=1, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = HighResolutionNet(width=width, ocr_width=ocr_width, aux_ch=3+1+self.coord_feature_ch, mid_ch=mid_ch, small=small,
                                                   update_num=update_num, expansion_ratio=expansion_ratio,
                                                   num_classes=num_classes, norm_layer=norm_layer)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        if ocr_width > 0:
            self.feature_extractor.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.conv3x3_ocr.apply(LRMult(1.0))


    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask
    
    def get_coord_features(self, image, prev_mask, points, designated_click_radius=-1):
        coord_features = self.dist_maps(image, points, designated_radius=designated_click_radius)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    def backbone_forward(self, image, click_encoding, new_points, prev_mask, gate, coord_features=None, is_training=True):
        net_outputs = self.feature_extractor(image, click_encoding, new_points, prev_mask, gate, coord_features, is_training=is_training)
        
        return {
            'instances': net_outputs[0], 'instances_aux': net_outputs[1], 'updated_feedback1': net_outputs[2],
            'valid_masks': net_outputs[3], 'updated_feedback2': net_outputs[4],
        }

    def forward(self, image, points, new_points, gate, designated_click_radius=-1, is_training=True):
        image, prev_mask = self.prepare_input(image)
        gauss_dist_maps = self.get_coord_features(image, prev_mask, points, designated_click_radius)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, gauss_dist_maps), dim=1))
            outputs = self.backbone_forward(x, gauss_dist_maps, new_points, prev_mask, gate, is_training=is_training)
        else:
            coord_features = self.maps_transform(gauss_dist_maps)
            outputs = self.backbone_forward(image, gauss_dist_maps, new_points, prev_mask, gate, coord_features, is_training=is_training)

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)

        for i in range(len(outputs['updated_feedback1'])):
            outputs['updated_feedback1'][i] = nn.functional.interpolate(outputs['updated_feedback1'][i], size=image.size()[2:],
                                                            mode='bilinear', align_corners=True)
            outputs['updated_feedback2'][i] = nn.functional.interpolate(outputs['updated_feedback2'][i], size=image.size()[2:],
                                                            mode='bilinear', align_corners=True)
        
        if is_training:
            for i in range(len(outputs['valid_masks'])):
                outputs['valid_masks'][i] = nn.functional.interpolate(outputs['valid_masks'][i], size=image.size()[2:],
                                                                mode='nearest')
        
                                                         
        if 'memories' in outputs.keys():
            outputs['memories'] = [nn.functional.interpolate(x, size=image.size()[2:],
                                                                mode='bilinear', align_corners=True) for x in outputs['memories']]
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        outputs['gauss_dist_maps'] = gauss_dist_maps
        return outputs
