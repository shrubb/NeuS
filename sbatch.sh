#!/bin/bash

# Train a 'metamodel' or a single-scene model. Or just run from arbitrary config.

#SBATCH --job-name 018-debug
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 1-0

#SBATCH -p gpu,htc,gpu_a100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 5
#SBATCH --mem-per-gpu 72G

##SBATCH --reservation egor.burkov_80

#: '
CONF=`mktemp`
cp confs/gonzalo_single.conf $CONF

PORT=25001
NPROC=1
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py \
--mode train \
--conf $CONF \
--extra_config_args 'dataset { batch_size = ${train.batch_size} }, train { batch_size = 512 }'
# '

