CUDA_VISIBLE_DEVICES=2 \
python scripts/evaluate_model.py \
NoBRS \
--datasets=SBD \
--backbone=r101 \
--resume-path=./weights/model_r101.pth \
--save-path=./results/r101 \
--with-flip \
--print-ious
# --vis-preds \
# --vis-points
