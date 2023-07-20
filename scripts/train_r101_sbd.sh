CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
models/iter_mask/r101_dh256_sbd.py \
--gpus=0,1 \
--batch-size=24 \
--epochs 220 \
--lr=5e-4 \
--workers=16 \
--exp-name=r101_dh256_sbd_expanratio0.3_midch32 \
--with_prev_mask \
--max_num_next_clicks 3 \
--update_num 1 \
--expansion_ratio 0.3 \
--mid_ch 32 \
--checkpoint_interval 5 \
--first_click_center