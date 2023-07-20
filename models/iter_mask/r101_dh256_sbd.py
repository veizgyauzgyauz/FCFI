import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.data.datasets import SBDDataset
from isegm.model.losses import NormalizedFocalLossSigmoid, MaskedNormalizedFocalLossSigmoid, SigmoidBinaryCrossEntropyLoss
from isegm.data.transforms import *
from isegm.engine.trainer_deeplab import ISTrainer
from isegm.model.metrics import AdaptiveIoU
from isegm.data.points_sampler import MultiPointSampler
from isegm.model import initializer

from isegm.model.is_deeplab_model import DeeplabModel

MODEL_NAME = 'sbd_resnet101'

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (320, 480)
    model_cfg.num_max_points = 24

    model = DeeplabModel(backbone='resnet101', deeplab_ch=256, mid_ch=cfg.mid_ch, aspp_dropout=0.20,
                         update_num=cfg.update_num, expansion_ratio=cfg.expansion_ratio, use_leaky_relu=True,
                         use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=cfg.with_prev_mask)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights()
    
    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4
    loss_cfg.feedback1_loss = MaskedNormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.feedback1_loss_weight = 1.0
    loss_cfg.feedback2_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.feedback2_loss_weight = 1.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.8,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2,
                                       first_click_center=cfg.first_click_center)

    trainset = SBDDataset(
        cfg.SBD_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        samples_scores_path='./assets/sbd_samples_weights.pkl',
        samples_scores_gamma=1.25
    )

    valset = SBDDataset(
        cfg.SBD_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500
    )

    optimizer_params = {
        'lr': cfg.lr, 'betas': (0.9, 0.999), 'eps': 1e-8
    }
    
    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 100, 150], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=cfg.checkpoint_interval,
                        image_dump_interval=2000,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=cfg.max_num_next_clicks)
    trainer.run(num_epochs=cfg.epochs)

