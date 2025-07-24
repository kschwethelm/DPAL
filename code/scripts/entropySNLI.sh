OUTPUT_FLAGS="--output_dir output --data_root datasets"
SYS_FLAGS="--max_physical_batch_size 1024"
TRAINING_FLAGS="--epochs 15 --batch_size 4096 --learning_rate 1e-3 --optimizer adamW --dataset snli --model bert"
AL_FLAGS="--al True --dp_al_mode step_ampl_prob_noise --query_strategy entropy --initial_dataset_size 20000 --query_sizes 20000,6000,2000,2000 --labeling_budget 50000 --al_batch_size 4096"
DP_FLAGS="--epsilon 8.0 --epsilon_al 2.0 --delta 2e-5 --max_sample_grad_norm 1.0"

uv run train.py --exp_name entropy --seed 1 $OUTPUT_FLAGS $SYS_FLAGS $TRAINING_FLAGS $AL_FLAGS $DP_FLAGS