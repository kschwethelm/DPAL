OUTPUT_PARAMS="--output_dir output --data_root datasets"
SYS_PARAMS="--use_functorch True --max_physical_batch_size 600"
TRAINING_PARAMS="--epochs 21 --batch_size 512 --learning_rate 2e-3 --optimizer nadam --dataset blood --model resnet9"
AL_PARAMS="--al True --dp_al_mode step_ampl_prob_noise --query_strategy entropy --initial_dataset_size 1024 --query_sizes 1024,300,100,52 --labeling_budget 2500 --al_batch_size 2048"
DP_PARAMS="--epsilon 8.0 --epsilon_al 0.0 --delta 4e-4 --max_sample_grad_norm 1.0"

python train.py --exp_name entropy $OUTPUT_PARAMS $SYS_PARAMS $TRAINING_PARAMS $AL_PARAMS $DP_PARAMS
    