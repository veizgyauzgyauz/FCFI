CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_model.py \
NoBRS \
--datasets=GrabCut \
--backbone=h18s \
--resume-path=./weights/model_h18s.pth \
--save-path=./results/h18s \
--with-flip \
--print-ious
# --vis-preds \
# --vis-points
