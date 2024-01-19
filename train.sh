EXPERIMENT=360_v2
GROUP=counter
NAME=test_scale_3
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar \
    --wandb \
    --wandb_name=${GROUP}_${NAME}