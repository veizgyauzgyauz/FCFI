from functools import lru_cache

import copy
import cv2
import numpy as np
import torch


def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    
 
def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None,
                               mask_color=(-1, -1, -1), boundary_color=(-1, -1, -1),
                               pos_point_color=(0, 255, 0), neg_point_color=(255, 0, 0),
                               radius=4):
    result = img.copy()
    mask = mask.astype(np.uint8)
    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        if boundary_color[0] < 0 or boundary_color[1] < 0 or boundary_color[2] < 0:
            if mask_color[0] < 0 or mask_color[1] < 0 or mask_color[2] < 0:
                rgb_mask = palette[mask]
            else:
                rgb_mask = np.repeat(mask[:,:,None], repeats=3, axis=2)
                rgb_mask[:,:,0] = rgb_mask[:,:,0] * mask_color[0]
                rgb_mask[:,:,1] = rgb_mask[:,:,1] * mask_color[1]
                rgb_mask[:,:,2] = rgb_mask[:,:,2] * mask_color[2]

            mask_region = (mask > 0).astype(np.uint8)
            result = result * (1 - mask_region[:, :, np.newaxis]) + \
                (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
                alpha * rgb_mask
            result = result.astype(np.uint8)

        else:
            enrode_kernel = np.ones((3,3))
            eroded_pred = cv2.erode(mask, enrode_kernel, iterations=2)
            dilated_pred = cv2.dilate(mask, enrode_kernel, iterations=2)
            circle_pred = dilated_pred - eroded_pred

            if mask_color[0] < 0 or mask_color[1] < 0 or mask_color[2] < 0:
                eroded_pred_3d = palette[eroded_pred]
                eroded_pred_3d = eroded_pred_3d.astype(np.uint8)
            else:
                eroded_pred_3d = np.repeat(eroded_pred[:,:,None], repeats=3, axis=2)
                eroded_pred_3d[:,:,0] = eroded_pred_3d[:,:,0] * mask_color[0]
                eroded_pred_3d[:,:,1] = eroded_pred_3d[:,:,1] * mask_color[1]
                eroded_pred_3d[:,:,2] = eroded_pred_3d[:,:,2] * mask_color[2]

            circle_pred_3d = np.repeat(circle_pred[:,:,None], repeats=3, axis=2)
            circle_pred_3d[:,:,0] = circle_pred_3d[:,:,0] * boundary_color[0]
            circle_pred_3d[:,:,1] = circle_pred_3d[:,:,1] * boundary_color[1]
            circle_pred_3d[:,:,2] = circle_pred_3d[:,:,2] * boundary_color[2]
            
            result = cv2.addWeighted(img, alpha, eroded_pred_3d, 1 - alpha, 0)
            
            eroded_mask = eroded_pred[:,:,None]
            result = eroded_mask * result + (1 - eroded_mask) * img

            circle_mask = circle_pred[:,:,None]
            result = circle_mask * circle_pred_3d + (1 - circle_mask) * result

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_point_color, radius=radius)
        result = draw_points(result, neg_points, neg_point_color, radius=radius)

    return result


def visualize_clicks_train(img, points, radius=5, first_click_center=False, new_points=None):
    # img: (H, W), [0, 255]
    # points: (N, 3)
    
    result = (255 * img).cpu().numpy().astype(np.uint8).transpose((1, 2, 0))
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    num_points = points.shape[0] // 2

    points, points_order = torch.split(points, [2, 1], dim=1)
    valid_points = torch.max(points, dim=1, keepdim=False)[0] >= 0 # (N)
    
    mask = torch.zeros_like(valid_points, dtype=torch.bool) # (N)
    mask[:num_points] = 1
    mask = valid_points & mask # (N)
    pos_points = points[mask]

    mask = torch.zeros_like(valid_points, dtype=torch.bool) # (N)
    mask[num_points:] = 1
    mask = valid_points & mask # (N)
    neg_points = points[mask]

    if new_points is not None:
        y_new_pt = int(new_points[0])
        x_new_pt = int(new_points[1])
        cv2.circle(result, center=(x_new_pt,y_new_pt), radius=2*radius, color=(255, 0, 255), thickness=-1)
        cv2.circle(result, center=(x_new_pt,y_new_pt), radius=2*radius, color=(0, 255, 0), thickness=2)
    
    for i, point in enumerate(pos_points.tolist()):
        y = int(point[0])
        x = int(point[1])
        if i == 0 and first_click_center:
            cv2.circle(result, center=(x,y), radius=2*radius, color=(255, 0, 255), thickness=-1)
            cv2.circle(result, center=(x,y), radius=2*radius, color=(0, 255, 0), thickness=2)
        else:
            cv2.circle(result, center=(x,y), radius=radius, color=(0, 0, 255), thickness=-1)
    
    for point in neg_points.tolist():
        y = int(point[0])
        x = int(point[1])
        cv2.circle(result, center=(x,y), radius=radius, color=(255, 0, 0), thickness=-1)
    return result


def visualize_clicks_val(img, points, radius=5, enhance_first_click=False):
    # img: (H, W, 3), [0, 255]
    # points: list, the element is of Clicker class

    assert img.shape[2] == 3
    result = copy.deepcopy(img)
    for i, point in enumerate(points):
        (y, x) = point.coords
        if point.is_positive:
            if i == 0 and enhance_first_click:
                cv2.circle(result, center=(x,y), radius=2*radius, color=(255,0,255), thickness=-1)
                cv2.circle(result, center=(x,y), radius=2*radius, color=(0,255,0), thickness=2)
            else:
                cv2.circle(result, center=(x,y), radius=radius, color=(0,255,0), thickness=-1)
        else:
            cv2.circle(result, center=(x,y), radius=radius, color=(0,0,255), thickness=-1)

    return result