OUTPUT_FLAGS="--output_dir output --data_root datasets"
SYS_FLAGS="--max_physical_batch_size 1024"
TRAINING_FLAGS="--epochs 50 --batch_size 4096 --learning_rate 1e-3 --optimizer adamW --dataset snli --model bert"
AL_FLAGS="--al False --labeling_budget 50000"
DP_FLAGS="--epsilon 8.0 --delta 2e-5 --max_sample_grad_norm 1.0"

uv run train.py --exp_name random --seed 1 $OUTPUT_FLAGS $SYS_FLAGS $TRAINING_FLAGS $AL_FLAGS $DP_FLAGS