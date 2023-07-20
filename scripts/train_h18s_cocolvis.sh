CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
models/iter_mask/hrnet18s_cocolvis.py \
--gpus=0,1 \
--batch-size=64 \
--epochs 220 \
--lr=5e-4 \
--workers=16 \
--exp-name=h18s_cocolvis_expanratio0.3_midch64 \
--with_prev_mask \
--max_num_next_clicks 3 \
--update_num 1 \
--expansion_ratio 0.3 \
--mid_ch 64 \
--checkpoint_interval 5 \
--first_click_center