import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

from isegm.model.modeling.basic_blocks import ConvBNReLU, XConvBNReLU


class FocusedCorrectionModule(nn.Module):
    def __init__(self, in_ch, mid_ch, aux_ch=6, norm_layer=nn.BatchNorm2d, expan_ratio=2.0, pred_thr=0.49):
        super(FocusedCorrectionModule, self).__init__()
        
        self.expan_ratio = expan_ratio
        self.adaptive_crop = False
        self.pred_thr = pred_thr

        self.info_encoder = nn.Sequential(
            ConvBNReLU(in_ch + aux_ch, mid_ch, norm_layer=norm_layer),
            XConvBNReLU(mid_ch, mid_ch, norm_layer),
            XConvBNReLU(mid_ch, mid_ch, norm_layer),
        )
    
    def get_crop_length(self, feedback):
        assert feedback.size(0) == 1 or feedback.size(0) == 2
        assert feedback.max() > 0
        mask = torch.where(feedback[0] > 0.49, torch.ones_like(feedback[0]), torch.zeros_like(feedback[0])).to(feedback.device)
        coord = masks_to_boxes(mask) # input size: (N, H, W), output size: (N, 4)
        coord = coord[0]
        w_length = int(coord[2] - coord[0])
        h_length = int(coord[3] - coord[1])
        return h_length, w_length
    
    def limit_crop_locations(self, crop_locations, h, w):
        zero_tensor = torch.zeros_like(crop_locations[:, 0]).to(crop_locations.device)
        one_tensor = torch.ones_like(crop_locations[:, 0]).to(crop_locations.device)

        crop_locations[:, 0] = torch.maximum(crop_locations[:, 0], zero_tensor)
        crop_locations[:, 1] = torch.maximum(crop_locations[:, 1], zero_tensor)
        crop_locations[:, 2] = torch.minimum(crop_locations[:, 2], h * one_tensor)
        crop_locations[:, 3] = torch.minimum(crop_locations[:, 3], w * one_tensor)

        return crop_locations

    def get_crop_locations(self, new_points, crop_sizes, h, w):
        crop_h, crop_w = crop_sizes
        crop_h, crop_w = int(crop_h / 2 + 0.5), int(crop_w / 2 + 0.5)
        crop_locations = torch.zeros(new_points.size(0), 4).to(new_points.device)

        crop_locations[:, 0] = new_points[:, 0] - crop_h
        crop_locations[:, 1] = new_points[:, 1] - crop_w
        crop_locations[:, 2] = new_points[:, 0] + crop_h
        crop_locations[:, 3] = new_points[:, 1] + crop_w
        crop_locations = self.limit_crop_locations(crop_locations, h, w)

        crop_locations = crop_locations.long()
        return crop_locations

    def get_masks_from_locations(self, input_h, input_w, locs, device):
        h = lambda x: 1. / (1. + torch.exp(-10. * x))
        
        unit_x = torch.stack([torch.arange(0, input_w)] * input_h).float()
        unit_y = torch.stack([torch.arange(0, input_h)] * input_w).float()
        x = torch.stack([unit_x])
        y = torch.stack([unit_y.t()])
        x, y = x.to(device), y.to(device)
        
        masks = []
        for i in range(len(locs)):
            y0, x0, y1, x1 = locs[i]            
            mask = (h(y - y0) - h(y - y1)) * (h(x - x0) - h(x - x1))
            masks.append(mask)
        
        masks = torch.stack(masks)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        
        return masks

    def resize_points(self, points, source_h, source_w, target_h, target_w):
        results = points.clone().detach()
        points = points.float()
        results[:, 0] = points[:, 0] / source_h * target_h
        results[:, 1] = points[:, 1] / source_w * target_w
        points = points.long()
        results = results.long()
        return results
    
    def justify_points(self, points, crop_locations):
        results = points.clone().detach()
        results[: ,0] = points[:, 0] - crop_locations[:, 0]
        results[: ,1] = points[:, 1] - crop_locations[:, 1]
        results = results.long()
        return results
        
    def resize_locations(self, locations, source_h, source_w, target_h, target_w):
        results = locations.clone().detach()
        locations = locations.float()

        results[:, 0] = torch.floor(locations[:, 0] / source_h * target_h)
        results[:, 1] = torch.floor(locations[:, 1] / source_w * target_w)
        results[:, 2] = torch.ceil(locations[:, 2] / source_h * target_h)
        results[:, 3] = torch.ceil(locations[:, 3] / source_w * target_w)
        results = self.limit_crop_locations(results, target_h, target_w)
        locations = locations.long()
        results = results.long()
        return results
    
    def update_single_feedback(self, features, feedback, new_points, masks=None):
        features = F.normalize(features, p=2, dim=1)
            
        b, c, ft_h, ft_w = features.size()
        coordinates_for_new_pt, labels_for_new_pt = torch.split(new_points, [2, 1], dim=1)
        with torch.no_grad():
            indices = torch.zeros(b, 1).to(features.device)
            indices[:, 0] = coordinates_for_new_pt[:, 0] * ft_w + coordinates_for_new_pt[:, 1]
            indices = indices.long().unsqueeze(-1)
        
        assert indices.min() > -1e-5
        assert indices.max() < ft_h * ft_w

        features = features.reshape(b, c, -1).permute(0, 2, 1) # (batch_size, h * w, c)
        features_for_new_pt = torch.gather(features, dim=1, index=indices.repeat(1, 1, c)).permute(0, 2, 1) # (b, c, 1)
        features = features.permute(0, 2, 1) # (batch_size, c, h * w)
        affinity = torch.einsum('ijk,ijl->ikl', features_for_new_pt, features).reshape(b, 1, ft_h, ft_w) # (b, 1, h * w)
        
        labels_for_new_pt = labels_for_new_pt[:, :, None, None] # (b, 1, 1, 1)
        if masks is None:
            corrected_feedback = (1 - affinity) * feedback + affinity * labels_for_new_pt
        else:
            corrected_feedback = (1 - masks) * feedback + masks * (affinity * labels_for_new_pt + (1 - affinity) * feedback)

        return corrected_feedback
    
    def update_feedback(self, features, feedback, new_points, masks=None):
        if isinstance(features, list):
            corrected_feedback = []
            for i in range(len(features)):
                sub_feature = features[i].unsqueeze(0)
                sub_feedback = feedback[i].unsqueeze(0)
                sub_corrected_feedback = self.update_single_feedback(sub_feature, sub_feedback, new_points[i].unsqueeze(0), masks)
                corrected_feedback.append(sub_corrected_feedback)
            return corrected_feedback
        else:
            return self.update_single_feedback(features, feedback, new_points, masks)

    def crop(self, x, crop_locations):
        b, _, h, w = x.size()
        y = []
        for i in range(b):
            y0, x0, y1, x1 = crop_locations[i, 0].item(), crop_locations[i, 1].item(), crop_locations[i, 2].item(), crop_locations[i, 3].item()
            cropped_x = x[i, :, y0 : y1, x0 : x1]
            y.append(cropped_x)
        
        return y
    
    def paste(self, patches, x, crop_locations):
        batch_size = x.size(0)
        for i in range(batch_size):
            y0, x0, y1, x1 = crop_locations[i, 0].item(), crop_locations[i, 1].item(), crop_locations[i, 2].item(), crop_locations[i, 3].item()
            x[i, :, y0 : y1, x0 : x1] = patches[i]
        return x

    def forward(self, images, click_encoding, features, feedback, new_points, is_training=True):
        _, _, ft_h, ft_w = features.size()
        _, _, fb_h, fb_w = feedback.size()
        
        if self.adaptive_crop and feedback.max() > 0:
            h_length, w_length = self.get_crop_length(feedback)
            crop_h = self.expan_ratio * h_length * 2.5
            crop_w = self.expan_ratio * w_length * 2.5
        else:
            expan_ratio = self.expan_ratio
        
            crop_h = expan_ratio * fb_h
            crop_w = expan_ratio * fb_w
        
        if is_training:
            new_points = new_points.unsqueeze(1)
        
        for i in range(new_points.size(1)):
            sub_new_points = new_points[:, i]
            crop_locations = self.get_crop_locations(sub_new_points, (crop_h, crop_w), fb_h, fb_w)
            new_points_rs = self.resize_points(sub_new_points, fb_h, fb_w, ft_h, ft_w)
            
            images = F.interpolate(images, size=(ft_h, ft_w), mode='bilinear', align_corners=True)
            click_encoding = F.interpolate(click_encoding, size=(ft_h, ft_w), mode='bilinear', align_corners=True)
            feedback_rs = F.interpolate(feedback, size=(ft_h, ft_w), mode='bilinear', align_corners=True)

            crop_locations_rs = self.resize_locations(crop_locations, fb_h, fb_w, ft_h, ft_w)
            
            concatenated_input = torch.cat((images, click_encoding, feedback_rs, features), dim=1)
            updated_features = self.info_encoder(concatenated_input)
            
            if is_training:
                valid_masks = self.get_masks_from_locations(ft_h, ft_w, crop_locations_rs, features.device)
                feedback = self.update_feedback(updated_features, feedback_rs, new_points_rs, valid_masks)

            else:
                cropped_feedback = self.crop(feedback_rs, crop_locations_rs)
                cropped_features = self.crop(updated_features, crop_locations_rs)
                new_points_rs_justified = self.justify_points(new_points_rs, crop_locations_rs)
                corrected_cropped_feedback = self.update_feedback(cropped_features, cropped_feedback, new_points_rs_justified, None)
                feedback = self.paste(corrected_cropped_feedback, feedback_rs, crop_locations_rs)
                
        if is_training:            
            return feedback, valid_masks
        else:
            return feedback, None