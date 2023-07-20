from time import process_time, time

import numpy as np
import torch
import cv2

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious, all_bious, all_assds = [], [], []
    elapsed_time = 0

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, sample_bious, sample_assds, _ = evaluate_sample(sample.image_name, sample.image, sample.gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append(sample_ious)
        all_bious.append(sample_bious)
        all_assds.append(sample_assds)
        
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, all_bious, all_assds, elapsed_time


def evaluate_sample(image_name, image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, dilation_ratio=0.02,
                    vis_points=False, vis_bbox=False, vis_iou=False, expansion_ratio=1.0,):
    
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    iou_list, biou_list, assd_list = [], [], []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            if click_indx == 0:
                pred_probs, processed_time, updated_feedback1, updated_feedback2 = predictor.get_prediction(clicker, pred_mask, 0.0)
            else:
                pred_probs, processed_time, updated_feedback1, updated_feedback2 = predictor.get_prediction(clicker, pred_probs, 1.0)
            
            pred_mask = pred_probs > pred_thr
            
            iou = utils.get_iou(gt_mask, pred_mask)
            biou = utils.get_biou(gt_mask, pred_mask, dilation_ratio=dilation_ratio)
            assd = utils.get_assd(gt_mask, pred_mask)


            iou_list.append(iou)
            biou_list.append(biou)
            assd_list.append(assd)

            if callback is not None:
                callback(image_name, image, gt_mask, pred_probs, iou, sample_id, click_indx, clicker.clicks_list, vis_points=vis_points, vis_bbox=vis_bbox, vis_iou=vis_iou, expansion_ratio=expansion_ratio)

        return clicker.clicks_list, np.array(iou_list, dtype=np.float32), np.array(biou_list, dtype=np.float32), np.array(assd_list, dtype=np.float32), pred_probs

