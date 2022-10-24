import subprocess
import pathlib
import sys

def run_subprocess(command, working_dir=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=working_dir)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)

def quoted(x):
    return f'"{x}"'

if __name__ == '__main__':
    DATASET_ROOT = pathlib.Path("/gpfs/data/gpfs0/egor.burkov/Projects/NeuS/datasets/in_the_wild/preprocessed")

    SCENES = [
        'elon_14.00Xface',
    ]

    MODEL_NAME = "100_rank50_splitReplFirstHalf_720k_noBkgdBugFix_ann100k"

    port = 29100

    for scene in SCENES:
        dataset_dir = DATASET_ROOT / scene
        assert dataset_dir.is_dir()

        experiment_name = f"{MODEL_NAME}_ftTo{scene}"
        exp_dir = pathlib.Path(f"./logs-paper/in_the_wild/{MODEL_NAME}_lowCamLR_newMaskLoss/{scene}")
        if exp_dir.is_dir():
            print(f"Already exists, skipping: {exp_dir}")
            port += 1
            continue
        exp_dir.mkdir(exist_ok=True, parents=True)

        script = \
f"""#!/bin/bash

# Finetune a 'metamodel'.

#SBATCH --job-name {experiment_name}
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 2:10:0

#SBATCH -p gpu,htc,gpu_a100 #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

##SBATCH --reservation egor.burkov_80

set -e

CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

LATEST_CKPT=`ls -t ./logs-new/{MODEL_NAME}/checkpoints | head -1`

PORT={port}
NPROC=1

# ======= Fit and fine-tune the model =======
# Using "if" to prevent bash from seeing read's exit code (1) and triggering "set -e"
if read -r -d '' EXTRA_ARGS; then :; fi << EndOfText
general {{
    base_exp_dir = {exp_dir}
}}
dataset {{
    data_dirs = ["{dataset_dir}"]
    images_to_pick = [[0, "default"]]
    images_to_pick_val = [[0, "default"]]
    batch_size = 512
}}
train {{
    batch_size = \\${{dataset.batch_size}}
    semantic_consistency_weight = 0.1
}}
EndOfText

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path ./logs-new/{MODEL_NAME}/checkpoints/$LATEST_CKPT \
--conf $CONF1 --extra_config_args "${{EXTRA_ARGS}}"

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF2 --extra_config_args 'general {{ base_exp_dir = {exp_dir} }}'

rm {exp_dir}/checkpoints/*
"""

        with open("tmp.sh", 'w') as f:
            f.write(script)

        run_subprocess(["sbatch", "tmp.sh"])

        port += 1
