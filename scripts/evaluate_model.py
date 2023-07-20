import sys
import pickle
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.parse_args import parse_args_val
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks, visualize_clicks_val
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset


global noc90
noc90 = 10000


def main():
    args, cfg = parse_args_val()

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg)

        for checkpoint_path in checkpoints_list:
            model = utils.load_is_model(checkpoint_path, args.device)
            
            if args.expansion_ratio > 0:
                model.feature_extractor.correction_module.expan_ratio = args.expansion_ratio

            if args.adaptive_crop:
                model.feature_extractor.correction_module.adaptive_crop = True

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, cfg)
            predictor = get_predictor(model, args.mode, args.device,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      zoom_in_params=zoomin_params,
                                      with_flip=args.with_flip,
                                      )

            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis_preds else None

            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               callback=vis_callback,
                                               dilation_ratio=args.dilation_ratio,
                                               vis_points=args.vis_points,
                                               vis_bbox=args.vis_bbox,
                                               vis_iou=args.vis_iou,
                                               expansion_ratio=model.feature_extractor.expansion_ratio)

            row_name = args.mode if single_model_eval else checkpoint_path.stem
            if args.iou_analysis:
                save_iou_analysis_data(args, dataset_name, logs_path,
                                       logs_prefix, dataset_results,
                                       model_name=args.model_name)
            
            save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                         save_ious=single_model_eval and args.save_ious,
                         single_model_eval=single_model_eval,
                         print_header=print_header)
            print_header = False


def get_predictor_and_zoomin_params(args, cfg):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit
    if 'ZOOM_IN' not in cfg:
        zoom_in_params = None
    else:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'target_size': cfg.ZOOM_IN.CVPR.TARGET_SIZE if args.zoom_in_target_size <= 0 else args.zoom_in_target_size,
                'expansion_ratio': cfg.ZOOM_IN.CVPR.EXPANSION_RATIO if args.zoom_in_expansion_ratio <= 0 else args.zoom_in_expansion_ratio
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = int(args.eval_mode[5:])
            zoom_in_params = {
                'skip_clicks': cfg.ZOOM_IN.FIXED.SKIP_CLICKS,
                'target_size': (crop_size, crop_size)
            }
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''

    if len(str(args.resume_path)) > 0 and args.resume_path.exists():
        if args.save_path is None:
            raise ValueError("You should specify a \"save_path\"")
        logs_path = args.save_path
        
        checkpoints_list = [Path(args.resume_path)]
        return checkpoints_list, logs_path, ''
    
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(str(exp_path_prefix).split('/')[-1] + '*'))
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."
        
        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            
            logs_prefix = 'all_checkpoints'

        if args.save_path is None:
            suffix = str(exp_path.relative_to(cfg.EXPS_PATH)).replace('train_logs/', '', 1)
            if args.refinement_mode >= 0 and args.refinement_iters > 0:
                logs_path = args.logs_path / suffix / (args.resume_path.stem + '_refinement')
            else:
                logs_path = args.logs_path / suffix / args.resume_path.stem
        else:
            logs_path = args.save_path

        if args.vis_aux:
            logs_path = Path(str(logs_path) + '_aux')
    else:
        checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem

    return checkpoints_list, logs_path, logs_prefix


def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, all_bious, all_assds, elapsed_time = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)
    
    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)
    
    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.4f};'
                             for click_id in range(1, args.n_clicks + 1) if click_id <= min_num_clicks])
        table_row += '\n\n' + miou_str
        
        mean_bious = np.array([x[:min_num_clicks] for x in all_bious]).mean(axis=0)
        mbiou_str = ' '.join([f'mBIoU@{click_id}={mean_bious[click_id - 1]:.4f};'
                             for click_id in range(1, args.n_clicks + 1) if click_id <= min_num_clicks])
        table_row += '\n\n' + mbiou_str

        mean_assds = np.array([x[:min_num_clicks] for x in all_assds]).mean(axis=0)
        massds_str = ' '.join([f'ASSD@{click_id}={mean_assds[click_id - 1]:.4f};'
                             for click_id in range(1, args.n_clicks + 1) if click_id <= min_num_clicks])
        table_row += '\n\n' + massds_str

    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.txt'
    
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')
    
    
def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None):
    all_ious, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_{args.mode}',
            'all_ious': all_ious
        }, f)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
    save_path = logs_path / 'predictions' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(image_name, image, gt_mask, pred_probs, iou, sample_id, click_indx, clicks_list, vis_points=False, vis_bbox=False, vis_iou=False, expansion_ratio=1.0, save_in_same_file=False):
        if save_in_same_file:
            save_name = '%s_@%03d.png' % (image_name, click_indx)
            sample_path = save_path / save_name
        else:
            save_name = '%s_%02d_pred.png' % (image_name, len(clicks_list))
            sample_path = save_path / 'all'
            sample_path.mkdir(parents=True, exist_ok=True)
            sample_path = sample_path / save_name
        
        vis_pred = 255 * (pred_probs > prob_thresh).astype(np.uint8)
        if vis_points:            
            vis_pred = np.repeat(vis_pred[:,:,None], repeats=3, axis=2)
            vis_pred = visualize_clicks_val(vis_pred, clicks_list, radius = image.shape[0]//60)
        if vis_bbox:
            # Calculate the bounding box
            h, w = pred_probs.shape
            y0 = int(max(clicks_list[-1].coords[0].item() - expansion_ratio * h // 2, 0))
            x0 = int(max(clicks_list[-1].coords[1].item() - expansion_ratio * w // 2, 0))
            y1 = int(min(clicks_list[-1].coords[0].item() + expansion_ratio * h // 2, h - 1))
            x1 = int(min(clicks_list[-1].coords[1].item() + expansion_ratio * w // 2, w - 1))
            
            # Draw the bounding box
            y0 = int(max)
            line_color = (211, 107, 166)
            line_thickness = 3
            cv2.line(vis_pred, (x0, y0), (x1, y0), line_color, thickness=line_thickness)
            cv2.line(vis_pred, (x1, y0), (x1, y1), line_color, thickness=line_thickness)
            cv2.line(vis_pred, (x1, y1), (x0, y1), line_color, thickness=line_thickness)
            cv2.line(vis_pred, (x0, y1), (x0, y0), line_color, thickness=line_thickness)
        if vis_iou:
            cv2.putText(vis_pred, 'iou=%.4f' % iou, (vis_pred.shape[1] - 160, vis_pred.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imwrite(str(sample_path), vis_pred)

    return callback


if __name__ == '__main__':
    main()
