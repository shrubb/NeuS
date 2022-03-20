import subprocess
import pathlib
import sys

def run_subprocess(command, working_dir=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=working_dir)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)

if __name__ == '__main__':
    VIEW_NAMES = ('left', 'frontal', 'right')
    SCENES = [
        ("1b2a8613401e42a8", ["0018", "0000", "0065"]),
        ("3b5a2eb92a501d54", ["0015", "0001", "0062"]),
        ("444ea0dc5e85ee0b", ["0012", "0000", "0059"]),
        ("5ae021f2805c0854", ["0007", "0001", "0054"]),
        ("5cd49557ea450c89", ["0014", "0000", "0065"]),
        ("609cc60fd416e187", ["0014", "0006", "0065"]),
        ("7dd427509fe84baa", ["0012", "0006", "0064"]),
        ("868765907f66fd85", ["0013", "0000", "0064"]),
        ("e98bae39fad2244e", ["0011", "0000", "0064"]),
        ("f7e930d8a9ff2091", ["0009", "0000", "0065"]),
    ]
    MODEL_NAME = "100_rank1000_smallLR"

    port = 26100

    for scene, images in SCENES:
        dataset_dir = pathlib.Path(f"./datasets/H3DS_processed/{scene}")
        assert dataset_dir.is_dir()

        for image, view_name in zip(images, VIEW_NAMES):
            experiment_name = f"{MODEL_NAME}_ftTo{scene[:4]}-1-{image}"
            exp_dir = pathlib.Path(f"./logs/paper/h3ds/{MODEL_NAME}/{view_name}/{scene}-1-{image}")
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
#SBATCH --time 0-4

##SBATCH -p gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#SBATCH --reservation egor.burkov_80

set -e

CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

LATEST_CKPT=`ls -t ./logs/{MODEL_NAME}/checkpoints | head -1`

PORT={port}
NPROC=1
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path ./logs/{MODEL_NAME}/checkpoints/$LATEST_CKPT \
--conf $CONF1 --extra_config_args 'general {{ base_exp_dir = {exp_dir} }}, dataset {{ data_dirs = ["{dataset_dir}"], images_to_pick = [[0, ["{image}"]]], images_to_pick_val = [[0, ["{images[0]}", "{images[-1]}"]]], }}, train {{ validate_resolution_level = 1 }}'

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF2 --extra_config_args 'general {{ base_exp_dir = {exp_dir} }}'
"""

            with open("tmp.sh", 'w') as f:
                f.write(script)

            run_subprocess(["sbatch", "tmp.sh"])

            port += 1
