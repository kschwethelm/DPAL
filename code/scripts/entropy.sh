OUTPUT_PARAMS="--output_dir output --data_root datasets"
SYS_PARAMS="--use_functorch True --max_physical_batch_size 1900"
TRAINING_PARAMS="--epochs 30 --batch_size 4096 --learning_rate 1e-3 --optimizer nadam --dataset cifar10 --model resnet9"
AL_PARAMS="--al True --dp_al_mode step_ampl_prob_noise --query_strategy entropy --initial_dataset_size 10000 --query_sizes 10000,3000,1000,1000 --labeling_budget 25000 --al_batch_size 2048"
DP_PARAMS="--epsilon 8.0 --epsilon_al 2.0 --delta 1e-5 --max_sample_grad_norm 1.0"

python train.py --exp_name entropy $OUTPUT_PARAMS $SYS_PARAMS $TRAINING_PARAMS $AL_PARAMS $DP_PARAMS
