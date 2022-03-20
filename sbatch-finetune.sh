#!/bin/bash

# Finetune a 'metamodel'.

#SBATCH --job-name 100_rank1000_smallLR_ftTo132-1
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0-4

#SBATCH -p gpu_a100,gpu,htc,res,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 5
#SBATCH --mem-per-gpu 12G

set -e

# Set these!
EXPERIMENT_DIR="./logs/100_rank1000_smallLR_ftTo132-1"
METAMODEL_CHECKPOINT="/gpfs/data/gpfs0/egor.burkov/Projects/NeuS/logs/100_rank1000_smallLR/checkpoints/ckpt_0850000.pth"

CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

PORT=26100
NPROC=1
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path ${METAMODEL_CHECKPOINT} \
--conf $CONF1 --extra_config_args "general { base_exp_dir = ${EXPERIMENT_DIR} }"

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF2 --extra_config_args "general { base_exp_dir = ${EXPERIMENT_DIR} }"
