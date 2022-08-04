#!/bin/bash

# Train a 'metamodel' or a single-scene model. Or just run from arbitrary config.

#SBATCH --job-name 100_rank1000_splitAll_720k_ann100k_lr0.8e4
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 1-10

#SBATCH -p htc,gpu #,gpu_a100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 3
#SBATCH --mem-per-gpu 72G

##SBATCH --reservation egor.burkov_80

set -e

CONF=`mktemp`
cp confs/gonzalo_100.conf $CONF

PORT=25001
NPROC=1

# Using "if" to prevent bash from seeing read's exit code (1) and triggering "set -e"
if read -r -d '' EXTRA_ARGS; then :; fi << EndOfText
general {
    base_exp_dir = ./logs-new/100_rank1000_splitAll_720k_ann100k_lr0.8e4
}
model {
    scenewise_split_type = all
    scenewise_core_rank = 1000
    sdf_network {
        scenewise_split_type = \${model.scenewise_split_type}
        scenewise_core_rank = \${model.scenewise_core_rank}
    }
    rendering_network {
        scenewise_split_type = \${model.scenewise_split_type}
        scenewise_core_rank = \${model.scenewise_core_rank}
    }
}
dataset {
    batch_size = \${train.batch_size}
}
train {
    batch_size = 512
    learning_rate = 0.8e-4
}
EndOfText

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py \
--mode train \
--conf $CONF \
--extra_config_args "${EXTRA_ARGS}"

