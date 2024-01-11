EXPERIMENT=dtu_example
GROUP=example_group
NAME=example_name
# CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
CONFIG=projects/neuralangelo/configs/dtu.yaml
GPUS=1  # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar