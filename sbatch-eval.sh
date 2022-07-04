#!/bin/bash

# Extract mesh or render a video (set `--mode`).
# Use the latest checkpoint in $DIR.
# Ideally, $PORT should be different at each run.

#SBATCH --job-name eval
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0:10:0

#SBATCH -p gpu #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
# Set these
DIR=logs-new/100_rank50_splitReplFirstHalf_720k_noBkgdBugFix_ann100k
PORT=23101

# Don't set these
LATEST_CKPT=`ls $DIR/checkpoints | tail -1`
NPROC=1
torchrun \
       --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py \
       --mode validate_mesh_98 \
       --checkpoint_path $DIR/checkpoints/$LATEST_CKPT \
       --extra_config_args 'dataset { images_to_pick = [[0, "default"]] }, train { parts_to_skip_loading = ["cameras"], load_optimizer = false }'
# '

# 018: interpolation_9_31
# 019: interpolation_8_28
# 132: interpolation_15_36
# 134: interpolation_25_9
# 130: 
# 200: 

# 036: interpolation_17_37

