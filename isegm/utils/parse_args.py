import argparse
from pathlib import Path
import torch

from isegm.utils.exp import load_config_file


def parse_args_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('--config-path', type=str, default='./configs/train/config.yml',
                        help='The path to the config file.')

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')

    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs. '
                             'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
                             'You should use either this argument or "--gpus".')

    parser.add_argument('--gpus', type=str, default='', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='The number of training epochs.')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')
                        
    parser.add_argument('--first_click_center', action='store_true',
                        help='The first click is in the center of an object or not')
    parser.add_argument('--with_prev_mask', action='store_true',
                        help='Add previous mask to the network or not')

    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument('--num_classes', type=int, default=1,
                        help='The number of object regions that are partitioned into.')

    parser.add_argument("--max_num_next_clicks", type=int, default=0)
    
    parser.add_argument('--update_num', type=int, default=1,
                        help='The number of times to update feedback')
    parser.add_argument('--expansion_ratio', type=float, default=2.0,
                        help='Expansion ratio for the focused correction module')
    parser.add_argument("--mid_ch", type=int, default=64)
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='The number of training epochs.')
    args = parser.parse_args()
    return args


def parse_args_val():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='')

    group_checkpoints = parser.add_mutually_exclusive_group()
    group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='The path to the checkpoint. '
                                        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                                        'or an absolute path. The file extension can be omitted.')
    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='The relative path to the experiment with checkpoints.'
                                        '(relative to cfg.EXPS_PATH)')
                                        
    parser.add_argument('--resume-path', type=str, default='',
                                   help='The relative path to the checkpoint.')

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,SBD,PascalVOC',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC')
    
    parser.add_argument('--backbone', type=str, default='r101', choices=['r101', 'h18s'])
    
    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.90,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument('--eval-mode', type=str, default='cvpr',
                        help='Possible choices: cvpr, fixed<number> (e.g. fixed400, fixed600).')

    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    parser.add_argument('--vis-points', action='store_true', default=False)
    parser.add_argument('--vis-bbox', action='store_true', default=False)
    parser.add_argument('--vis-iou', action='store_true', default=False)
    parser.add_argument('--model-name', type=str, default=None,
                        help='The model name that is used for making plots.')
    # parser.add_argument('--config-path', type=str, default='./configs/val/config.yml',
    #                     help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')
    parser.add_argument('--save-path', type=str, default='',
                        help='The path to save the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')

    parser.add_argument('--vis-aux', action='store_true', default=False)
    parser.add_argument('--with-flip', action='store_true', default=False)

    parser.add_argument("--zoom_in_target_size", type=int, default=-1)
    parser.add_argument("--zoom_in_expansion_ratio", type=float, default=-1.0)
    parser.add_argument("--expansion_ratio", type=float, default=-1.0)
    parser.add_argument('--adaptive_crop', action='store_true', default=False)

    parser.add_argument("--dilation_ratio", type=float, default=0.02,
                        help='The dilation ratio for the BIoU calculation')
    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    args.config_path = './configs/val/config_{}_{}.yml'.format(args.datasets.lower(), args.backbone.lower())
    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    if args.save_path == '':
        args.save_path = None
    else:
        args.save_path = Path(args.save_path)

    if args.resume_path != '':
        args.resume_path = Path(args.resume_path)
    return args, cfg