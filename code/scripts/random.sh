OUTPUT_PARAMS="--output_dir output --data_root datasets"
SYS_PARAMS="--use_functorch True --max_physical_batch_size 2048"
TRAINING_PARAMS="--epochs 100 --batch_size 4096 --learning_rate 1e-3 --optimizer nadam --dataset cifar10 --model resnet9"
AL_PARAMS="--al False --labeling_budget 25000"
DP_PARAMS="--epsilon 8.0 --epsilon_al 0.0 --delta 1e-5 --max_sample_grad_norm 1.0"

python train.py --exp_name random $OUTPUT_PARAMS $SYS_PARAMS $TRAINING_PARAMS $AL_PARAMS $DP_PARAMS