#!/bin/bash

# Continue training from a checkpoint.
# Suitable for both meta-learning and fine-tuning.

#SBATCH --job-name color-fit
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0-2

#SBATCH -p gpu_a100 #gpu,htc,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
# Set these
CHECKPOINT=logs-paper-tmp/in_the_wild/100_rank1000_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr1.8e4_48/elon_14.00Xface/checkpoints/ckpt_0038000.pth
EXPERIMENT_DIR=logs-paper-tmp/in_the_wild/100_rank1000_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr1.8e4_48/elon_14.00Xface_colorLongFit_largerLR/
PORT=27100
NPROC=1

# Using "if" to prevent bash from seeing read's exit code (1) and triggering "set -e"
if read -r -d '' EXTRA_ARGS; then :; fi << EndOfText
dataset {
    batch_size = \${train.batch_size}
}
general {
    base_exp_dir = ${EXPERIMENT_DIR}
}
train {
    parts_to_freeze = ["nerf_outside", "sdf", "cameras", "deviation"]

    batch_size = 768

    learning_rate = 5e-5
    learning_rate_reduce_steps = [58000, 78000]
    learning_rate_reduce_factor = 0.4
    warm_up_end = 300
    end_iter = 88000
}
model {
    neus_renderer {
        n_samples = 64
        n_importance = 64
    }
}
EndOfText

torchrun \
    --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
    --checkpoint_path "${CHECKPOINT}" \
    --extra_config_args "${EXTRA_ARGS}"
