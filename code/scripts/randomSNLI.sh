python train.py \
    --exp_name random \
    --output_dir output \
    --seed 1 \
    --epochs 50 \
    --batch_size 4096 \
    --max_physical_batch_size 1024 \
    --learning_rate 1e-3 \
    --optimizer adamW \
    --dataset snli \
    --data_root datasets \
    --model bert \
    --al False \
    --labeling_budget 50000 \
    --epsilon 8.0 \
    --delta 2e-5 \
    --max_sample_grad_norm 1.0