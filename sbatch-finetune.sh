#!/bin/bash

# Finetune a 'metamodel'.

#SBATCH --job-name 100_rank30_smallLR_ftTo1b2a-1
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0-3

#SBATCH -p gpu,htc,res
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 5
#SBATCH --mem-per-gpu 13G

set -e

#: '
CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

PORT=26147
NPROC=1
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF1 --extra_config_args 'train { validate_resolution_level = 1 }'
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF2 --extra_config_args 'train { validate_resolution_level = 1 }'
# '
