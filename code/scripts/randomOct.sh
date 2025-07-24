OUTPUT_FLAGS="--output_dir output --data_root datasets"
SYS_FLAGS="--use_functorch True --max_physical_batch_size 256"
TRAINING_FLAGS="--epochs 100 --batch_size 4096 --learning_rate 1e-3 --optimizer nadam --dataset oct --model resnet9"
AL_FLAGS="--al False --labeling_budget 25000"
DP_FLAGS="--epsilon 8.0 --epsilon_al 0.0 --delta 4e-5 --max_sample_grad_norm 1.0"

uv run train.py --exp_name random $OUTPUT_FLAGS $SYS_FLAGS $TRAINING_FLAGS $AL_FLAGS $DP_FLAGS