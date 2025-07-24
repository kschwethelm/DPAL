OUTPUT_FLAGS="--output_dir output --data_root datasets"
SYS_FLAGS="--use_functorch True --max_physical_batch_size 512"
TRAINING_FLAGS="--epochs 90 --batch_size 512 --learning_rate 2e-3 --optimizer nadam --dataset blood --model resnet9"
AL_FLAGS="--al False --labeling_budget 2500"
DP_FLAGS="--epsilon 8.0 --delta 4e-4 --max_sample_grad_norm 2.0"

uv run train.py --exp_name random --seed 1 $OUTPUT_FLAGS $SYS_FLAGS $TRAINING_FLAGS $AL_FLAGS $DP_FLAGS