EXPERIMENT=360_v2
GROUP=counter
NAME=test_scale_3
CHECKPOINT=logs/${GROUP}/${NAME}/epoch_00083_iteration_000010000_checkpoint.pt
OUTPUT_MESH=logs/${GROUP}/${NAME}/360_v2_counter.ply
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
RESOLUTION=2048
BLOCK_RES=128
GPUS=1  # use >1 for multi-GPU mesh extraction
torchrun --nproc_per_node=${GPUS} projects/neuralangelo/scripts/extract_mesh.py \
    --config=${CONFIG} \
    --checkpoint=${CHECKPOINT} \
    --output_file=${OUTPUT_MESH} \
    --resolution=${RESOLUTION} \
    --block_res=${BLOCK_RES} \
    --textured