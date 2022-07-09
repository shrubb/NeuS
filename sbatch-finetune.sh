#!/bin/bash

# Finetune a 'metamodel'.

#SBATCH --job-name botticelli
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 2:30:0

#SBATCH -p gpu,htc #gpu,htc,gpu_a100,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

set -e

# Set these!
EXPERIMENT_DIR="./logs-paper/in_the_wild/100_rank1000_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr1.8e4/botticelli"
METAMODEL_CHECKPOINT="/gpfs/data/gpfs0/egor.burkov/Projects/NeuS/logs-new/100_rank1000_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr1.8e4/checkpoints/ckpt_0720000.pth"
DATASET_DIR="./datasets/in_the_wild_data/botticelli"
# Set these if you want to use only certain images from the dataset for val/train. Otherwise, comment out.
#IMAGES_TRAIN="[\"00552\", \"00460\", \"01072\"]"
#IMAGES_VAL="[\"00929\", \"01029\"]"

CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

PORT=26100 # Important: increment on every run
NPROC=1
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path ${METAMODEL_CHECKPOINT} \
--conf $CONF1 --extra_config_args "general { base_exp_dir = ${EXPERIMENT_DIR} }, dataset { data_dirs = [\"${DATASET_DIR}\"], images_to_pick = [[0, ${IMAGES_TRAIN:-\"default\"}]], images_to_pick_val = [[0, ${IMAGES_VAL:-\"default\"}]] }"

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF2 --extra_config_args "general { base_exp_dir = ${EXPERIMENT_DIR} }"

