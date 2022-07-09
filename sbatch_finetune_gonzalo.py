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
    DATASET_ROOT = pathlib.Path("/gpfs/data/gpfs0/egor.burkov/Datasets/Gonzalo")

    VIEWS = [
        ('left', slice(0, 1)),
        ('frontal', slice(1, 2)),
        ('right', slice(2, 3)),
        ('LR', slice(0, 3, 2)),
        ('LFR', slice(None)),
    ]
    # path to scene / train images / val images
    SCENES = [
        ("132/2021-07-22-12-28-42", ["00460", "00661", "00598"], ["00497", "00629"]), # Аня В
        ("134/2021-07-22-12-44-12", ["00689", "00479", "00556"], ["00731", "00584"]), # Женя У
        ("130/2021-07-22-11-55-13", ["00552", "00460", "01072"], ["00490", "00749"]), # Ренат
        ("149/2021-07-28-20-34-57", ["00958", "00721", "00812"], ["00686", "01070"]), # мужик в серой майке
    ]

    # MODEL_NAME = "100_rank50_splitReplFirstHalf_720k_noBkgdBugFix_ann100k"
    # MODEL_NAME = "100_rank150_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr3.5e4"
    # MODEL_NAME = "100_rank400_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr2.5e4"
    MODEL_NAME = "100_rank1000_splitReplFirstHalf_720k_noBkgdBugFix_ann100k_lr1.8e4"

    port = 28300

    for scene, train_images, val_images in SCENES:
        dataset_dir = DATASET_ROOT / scene / "portrait_reconstruction"
        assert dataset_dir.is_dir()

        for view_name, view_idxs in VIEWS:
            current_train_images = train_images[view_idxs]

            experiment_name = f"{MODEL_NAME}_ftTo{scene[:3]}-{view_name}"
            exp_dir = pathlib.Path(f"./logs-paper/gonzalo/{MODEL_NAME}_camFt_lowCamLR/{view_name}-{len(current_train_images)}/{scene[:3]}")
            if exp_dir.is_dir():
                print(f"Already exists, skipping: {exp_dir}")
                port += 1
                continue
            exp_dir.mkdir(exist_ok=True, parents=True)

            cameras_exp_dir = pathlib.Path(f"./logs-paper/gonzalo_val-cameras-opt/{MODEL_NAME}_camFt_lowCamLR/{view_name}-{len(current_train_images)}/{scene[:3]}")
            for iteration in '18000', '38000':
                (cameras_exp_dir / iteration).mkdir(exist_ok=True, parents=True)

            script = \
f"""#!/bin/bash

# Finetune a 'metamodel'.

#SBATCH --job-name {experiment_name}
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 3:0:0

#SBATCH -p gpu_a100,htc,gpu #,gpu_devel
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

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path ./logs-new/{MODEL_NAME}/checkpoints/$LATEST_CKPT \
--conf $CONF1 --extra_config_args 'general {{ base_exp_dir = {exp_dir} }}, dataset {{ data_dirs = ["{dataset_dir}"], images_to_pick = [[0, [{', '.join(map(quoted, current_train_images))}]]], images_to_pick_val = [[0, [{', '.join(map(quoted, val_images))}]]], batch_size = 512 }}, train {{ batch_size = ${{dataset.batch_size}}, cameras_optimizer_extra_args {{ base_learning_rate = 2.5e-5 }} }}'

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF2 --extra_config_args 'general {{ base_exp_dir = {exp_dir} }}, train {{ cameras_optimizer_extra_args {{ base_learning_rate = 2.5e-5 }} }}'

# ======= Optimize val cameras to measure PSNR =======

CONF3=`mktemp`
cp confs/gonzalo_optimize_val_cameras.conf $CONF3
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path {exp_dir}/checkpoints/ckpt_0018000.pth \
--conf $CONF3 --extra_config_args 'general {{ base_exp_dir = {cameras_exp_dir}/18000 }}, dataset {{ data_dirs = ["{dataset_dir}"], images_to_pick = [[0, [{', '.join(map(quoted, val_images))}]]], images_to_pick_val = [[0, [{', '.join(map(quoted, val_images))}]]], batch_size = 512 }}, train {{ restart_from_iter = 0, batch_size = ${{dataset.batch_size}} }}'
rm {cameras_exp_dir}/18000/checkpoints/*
rm {cameras_exp_dir}/18000/meshes/*

torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--checkpoint_path {exp_dir}/checkpoints/ckpt_0038000.pth \
--conf $CONF3 --extra_config_args 'general {{ base_exp_dir = {cameras_exp_dir}/38000 }}, dataset {{ data_dirs = ["{dataset_dir}"], images_to_pick = [[0, [{', '.join(map(quoted, val_images))}]]], images_to_pick_val = [[0, [{', '.join(map(quoted, val_images))}]]], batch_size = 512 }}, train {{ restart_from_iter = 0, batch_size = ${{dataset.batch_size}} }}'
rm {cameras_exp_dir}/38000/checkpoints/*
rm {cameras_exp_dir}/38000/meshes/*
"""

            with open("tmp.sh", 'w') as f:
                f.write(script)

            run_subprocess(["sbatch", "tmp.sh"])

            port += 1
