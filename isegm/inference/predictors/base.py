import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()
        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, clicker, prev_mask, gate, designated_click_radius=-1, new_click_num=1):
        clicks_list = clicker.get_clicks()

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        prev_mask = torch.from_numpy(prev_mask)[None, None, :, :].float().to(input_image.device)
        input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
        
        start_time = time.time()
        outputs = self._get_prediction(image_nd, clicks_lists, gate, is_image_changed, designated_click_radius, new_click_num)
        pred_logits = outputs['instances']
        updated_feedback1 = outputs['updated_feedback1']
        updated_feedback2 = outputs['updated_feedback2']

        if isinstance(updated_feedback1, list):
            updated_feedback1 = updated_feedback1[-1]
        if isinstance(updated_feedback2, list):
            updated_feedback2 = updated_feedback2[-1]
        
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        updated_feedback1 = F.interpolate(updated_feedback1, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        updated_feedback2 = F.interpolate(updated_feedback2, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)
            if t.__class__.__name__ != 'SigmoidForPred': 
                updated_feedback1 = t.inv_transform(updated_feedback1)
                updated_feedback2 = t.inv_transform(updated_feedback2)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            result = self.get_prediction(clicker)
            end_time = time.time()
            return result, end_time - start_time, updated_feedback1, updated_feedback2

        self.prev_prediction = prediction
        end_time = time.time()
        return prediction.cpu().numpy()[0, 0], end_time - start_time, updated_feedback1.cpu().numpy()[0, 0], updated_feedback2.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, gate, is_image_changed, designated_click_radius=-1, new_click_num=1):
        new_clicks = []
        for i in range(len(clicks_lists)):
            batch_new_clicks = []
            for j in range(new_click_num):
                ind = len(clicks_lists[i]) - j - 1
                new_click = torch.tensor((clicks_lists[i][ind].coords[0], clicks_lists[i][ind].coords[1], float(clicks_lists[i][ind].is_positive))).to(image_nd.device).unsqueeze(0).long()
                batch_new_clicks.append(new_click)
            batch_new_clicks = torch.cat(batch_new_clicks, dim=0).unsqueeze(0)
            new_clicks.append(batch_new_clicks)
        new_clicks = torch.cat(new_clicks, dim=0) # batch_size, new_click_num, 3
        points_nd = self.get_points_nd(clicks_lists)
        outputs = self.net(image_nd, points_nd, new_clicks, gate, designated_click_radius, is_training=False)
        return outputs

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
