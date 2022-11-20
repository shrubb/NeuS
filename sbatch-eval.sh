#!/bin/bash

# Extract mesh or render a video (set `--mode`).
# Use the latest checkpoint in $DIR.
# Ideally, $PORT should be different at each run.

#SBATCH --job-name eval
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0:10:0

#SBATCH -p gpu_devel #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
# Set these
DIR=logs-links-tmp/10_rank50_splitReplFirstHalf_720k_lr1.8e4_camFtScw0.07_48/elon_14.00Xface
PORT=23101

# Don't set these
LATEST_CKPT=`ls $DIR/checkpoints | tail -1`
NPROC=1

# Using "if" to prevent bash from seeing read's exit code (1) and triggering "set -e"
if read -r -d '' EXTRA_ARGS; then :; fi << EndOfText
dataset {
    images_to_pick = [[0, "default"]]
}
train {
    parts_to_skip_loading = ["cameras"]
    load_optimizer = false
}
EndOfText

torchrun \
    --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py \
    --mode validate_mesh_0 \
    --checkpoint_path $DIR/checkpoints/$LATEST_CKPT \
    --extra_config_args ''
# '

# 018: interpolation_9_31
# 019: interpolation_8_28
# 132: interpolation_15_36
# 134: interpolation_25_9
# 130: 
# 200: 

# 036: interpolation_17_37

