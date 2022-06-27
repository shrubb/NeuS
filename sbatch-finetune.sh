#!/bin/bash

# Finetune a 'metamodel'.

#SBATCH --job-name 100_rank50_splitReplFirstHalf__ftTo130-3_camRegNoBug
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 2:40:0

#SBATCH -p gpu #gpu,htc,gpu_a100,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

set -e

# Set these!
EXPERIMENT_DIR="./logs-new/100_rank50_splitReplFirstHalf__ftTo130-3_camRegNoBug"
METAMODEL_CHECKPOINT="./logs-new/100_rank50_splitReplFirstHalf_/checkpoints/ckpt_0550000.pth"
DATASET_DIR="./datasets/Gonzalo/130/2021-07-22-11-55-13/portrait_reconstruction/"

CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

PORT=26100
NPROC=1
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path ${METAMODEL_CHECKPOINT} \
--conf $CONF1 --extra_config_args "general { base_exp_dir = ${EXPERIMENT_DIR} }, dataset { data_dirs = [\"${DATASET_DIR}\"], images_to_pick = [[0, [\"00552\", \"00460\", \"01072\"]]], images_to_pick_val = [[0, [\"00929\", \"01029\"]]] }, train { parts_to_freeze = [\"nerf_outside\"] }"
# --conf $CONF1 --extra_config_args "general { base_exp_dir = ${EXPERIMENT_DIR} }, dataset { data_dirs = [\"${DATASET_DIR}\"], images_to_pick = [[0, [\"00460\"]]], images_to_pick_val = [[0, [\"00929\", \"01029\"]]] }, train { parts_to_freeze = [\"nerf_outside\"] }"

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF2 --extra_config_args "general { base_exp_dir = ${EXPERIMENT_DIR} }"
