from typing import List

from models.dataset import Dataset, load_K_Rt_from_P
from models.fields import \
    RenderingNetwork, SDFNetwork, SingleVarianceNetwork, MultiSceneNeRF, TrainableCameraParams
from models.renderer import NeuSRenderer

import cv2
import numpy as np
import trimesh
import torch
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
import apex
from tqdm import tqdm
from pyhocon import ConfigFactory, ConfigTree
from pyhocon.converter import HOCONConverter
import clip

import random
import pathlib
import os
import time
import logging
import argparse
from shutil import copyfile
import contextlib

CLIP_INPUT_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)),
])
POSES = [
    [[ 9.13055420e-01, -1.76651618e-08,  4.07835543e-01,  1.66675574e-02], # front
     [-3.41649936e-03,  9.99964893e-01,  7.64884520e-03, -1.25894707e+00],
     [-4.07821238e-01, -8.37718882e-03,  9.13023353e-01,  5.17910280e+00],],
    [[ 6.21426702e-01, -3.38232020e-08,  7.83472300e-01, -2.61097016e+00], # front front left
     [-6.71150442e-03,  9.99963462e-01,  5.32340771e-03, -1.25458061e+00],
     [-7.83443570e-01, -8.56638607e-03,  6.21404052e-01,  5.57463642e+00],],
    [[ 2.94054359e-01, -4.15223020e-08,  9.55788732e-01, -4.26811061e+00], # front left
     [-7.95119256e-03,  9.99965549e-01,  2.44627800e-03, -1.24256250e+00],
     [-9.55755651e-01, -8.31899885e-03,  2.94044346e-01,  7.18465868e+00],],
    [[ 1.28013372e-01, -4.25833129e-08,  9.91772473e-01, -4.84210039e+00], # front left left
     [-8.31875671e-03,  9.99964833e-01,  1.07378943e-03, -1.23630512e+00],
     [-9.91737545e-01, -8.38777330e-03,  1.28008887e-01,  7.87177358e+00],],
    [[-2.31622174e-01, -5.49429409e-08,  9.72805917e-01, -5.59038472e+00], # left
     [ 1.55864701e-01,  9.87081170e-01,  3.71109620e-02, -9.68393748e-01],
     [-9.60238278e-01,  1.60221800e-01, -2.28629708e-01,  9.78317243e+00],],
    [[-6.86852098e-01, -3.18955671e-08,  7.26797283e-01, -5.37014711e+00], # back left left
     [-6.54776953e-03,  9.99959528e-01, -6.18785666e-03, -1.19452542e+00],
     [-7.26767778e-01, -9.00904275e-03, -6.86824143e-01,  1.19669037e+01],],
    [[-7.40376890e-01, -2.83647132e-08,  6.72191978e-01, -5.18546782e+00], # back left
     [-9.44109727e-03,  9.99901354e-01, -1.03987278e-02, -1.16405409e+00],
     [-6.72125697e-01, -1.40452068e-02, -7.40303814e-01,  1.40300920e+01],],
    [[-9.78950083e-01, -1.11002176e-08,  2.04100594e-01, -3.29386437e+00], # back back left
     [ 4.33199406e-02,  9.77216005e-01,  2.07780212e-01, -2.15182202e+00],
     [-1.99450329e-01,  2.12248057e-01, -9.56645370e-01,  1.55593423e+01],],
    [[-9.28415358e-01,  2.00680539e-08, -3.71544033e-01, -2.40627455e-01], # back
     [-4.21732068e-02,  9.93537128e-01,  1.05382591e-01, -1.84820227e+00],
     [ 3.69142771e-01,  1.13508016e-01, -9.22415078e-01,  1.61927992e+01],],
    [[-6.65150940e-01,  3.31999388e-08, -7.46709049e-01,  2.31837376e+00], # back back right
     [-8.94868746e-02,  9.92793143e-01,  7.97128454e-02, -1.82465179e+00],
     [ 7.41327524e-01,  1.19841725e-01, -6.60357058e-01,  1.53759080e+01],],
    [[-3.49171966e-01,  4.10960190e-08, -9.37058687e-01,  4.04240439e+00], # back right
     [-1.05092347e-01,  9.93691146e-01,  3.91601324e-02, -1.65195251e+00],
     [ 9.31146860e-01,  1.12151310e-01, -3.46969098e-01,  1.51685644e+01],],
    [[ 7.33379871e-02,  3.51380116e-08, -9.97307181e-01,  5.34065222e+00], # back right right
     [-1.22628272e-01,  9.92411792e-01, -9.01756436e-03, -1.44159609e+00],
     [ 9.89739299e-01,  1.22959383e-01,  7.27815703e-02,  1.24946116e+01],],
    [[ 4.42092955e-01,  3.95985751e-08, -8.96969259e-01,  5.68106786e+00], # right
     [-9.17920992e-02,  9.94749963e-01, -4.52419110e-02, -1.18470202e+00],
     [ 8.92260075e-01,  1.02335818e-01,  4.39771980e-01,  1.02253422e+01],],
    [[ 7.13936865e-01,  3.16013669e-08, -7.00210094e-01,  5.29553680e+00], # front right right
     [-2.64307968e-02,  9.99287367e-01, -2.69488972e-02, -1.13249517e+00],
     [ 6.99711084e-01,  3.77469212e-02,  7.13428080e-01,  8.28680028e+00],],
    [[ 8.96319091e-01,  1.94137417e-08, -4.43409622e-01,  4.39001923e+00], # front right
     [ 4.03162930e-03,  9.99958873e-01,  8.14967882e-03, -1.24415518e+00],
     [ 4.43391293e-01, -9.09237657e-03,  8.96282256e-01,  6.39939936e+00],],
    [[ 9.98546600e-01,  2.34306574e-09, -5.38949072e-02,  2.60953448e+00], # front front right
     [ 4.56380600e-04,  9.99964237e-01,  8.45570955e-03, -1.25409103e+00],
     [ 5.38929738e-02, -8.46801698e-03,  9.98510897e-01,  5.69713397e+00],],
]

def psnr(color_fine, true_rgb, mask):
    assert mask.shape[:-1] == color_fine.shape[:-1] and mask.shape[-1] == 1
    return 20.0 * torch.log10(
        1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask.sum() * 3.0 + 1e-5)).sqrt())


class Runner:
    def __init__(self, args):
        self.rank = args.rank
        self.world_size = args.world_size

        conf_path = args.conf
        checkpoint_path = args.checkpoint_path
        self.device = args.device

        assert conf_path or checkpoint_path or args.extra_config_args, \
            "Specify at least config, checkpoint or extra_config_args"

        def update_config_tree(target: ConfigTree, source: ConfigTree, current_prefix: str = ''):
            """
            Recursively update values in `target` with those in `source`.

            current_prefix:
                str
                No effect, only used for logging.
            """
            for key in source.keys():
                if key not in target:
                    target[key] = source[key]
                else:
                    assert type(source[key]) == type(target[key]), \
                        f"Types differ in ConfigTrees: asked to update '{type(target[key])}' " \
                        f"with '{type(source[key])}' at key '{current_prefix}{key}'"

                    if type(source[key]) is ConfigTree:
                        update_config_tree(target[key], source[key], f'{current_prefix}{key}.')
                    else:
                        if target[key] != source[key] and self.rank == 0:
                            logging.info(
                                f"Updating config value at '{current_prefix}{key}'. "
                                f"Old: '{target[key]}', new: '{source[key]}'")

                        target[key] = source[key]

        # The eventual configuration, gradually filled from various sources
        # Config params resolution order: cmdline -> file -> checkpoint
        self.conf = ConfigFactory.parse_string("")
        if conf_path is not None:
            if self.rank == 0: logging.info(f"Using config '{conf_path}'")
            update_config_tree(self.conf, ConfigFactory.parse_file(conf_path))
        if args.extra_config_args is not None:
            update_config_tree(self.conf, ConfigFactory.parse_string(args.extra_config_args))

        # Now we know where to look for checkpoint
        if checkpoint_path is None:
            # If not specified as cmdline argument, get from config
            checkpoint_path = self.conf.get_string('train.checkpoint_path', default=None)
        if checkpoint_path is None:
            # If not specified anywhere, try looking in 'base_exp_dir'
            base_exp_dir = self.conf.get_string('general.base_exp_dir', default=None)
            if base_exp_dir is not None:
                checkpoints_dir = pathlib.Path(base_exp_dir) / "checkpoints"
                if checkpoints_dir.is_dir():
                    checkpoints = sorted(checkpoints_dir.iterdir())
                else:
                    checkpoints = []

                if checkpoints:
                    checkpoint_path = checkpoints[-1]
        if checkpoint_path is None:
            if self.rank == 0: logging.info(f"Not loading any checkpoint")
        else:
            # Load the checkpoint, for now just to extract config from there
            if self.rank == 0: logging.info(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            self.conf = ConfigFactory.parse_string("")

            if 'config' in checkpoint:
                # Temporary dynamic defaults for backward compatibility. TODO: remove
                if 'dataset.original_num_scenes' not in checkpoint['config']:
                    checkpoint['config']['dataset']['original_num_scenes'] = \
                        len(checkpoint['config']['dataset.data_dirs'])

                update_config_tree(self.conf, checkpoint['config'])
            if conf_path is not None:
                update_config_tree(self.conf, ConfigFactory.parse_file(conf_path))
            if args.extra_config_args is not None:
                update_config_tree(self.conf, ConfigFactory.parse_string(args.extra_config_args))

        self.base_exp_dir = self.conf.get_string('general.base_exp_dir', default=None)
        assert self.base_exp_dir is not None, "'base_exp_dir' not defined anywhere"

        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.optimize_cameras = not 'cameras' in self.conf.get_list(
            'train.parts_to_freeze', default=['cameras'])
        logging.info(f"optimize_cameras is {self.optimize_cameras}")

        self.dataset = Dataset(self.conf['dataset'], kind='train',
            return_cameras_only=self.optimize_cameras)
        self.dataset_val = Dataset(self.conf['dataset'], kind='val',
            return_cameras_only=self.optimize_cameras)

        self.apply_camera_correction_to_val = self.conf.get_bool(
            'train.apply_camera_correction_to_val', default=False)
        if self.apply_camera_correction_to_val:
            assert self.conf.get_list('dataset.images_to_pick') == \
                self.conf.get_list('dataset.images_to_pick_val'), \
                "You asked to apply camera params corrections to val dataset. For that, " \
                "train and val datasets must be identical! Or know what you're doing"

        logging.info(f"Experiment dir: {self.base_exp_dir}")

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_sphere_tracing_freq = self.conf.get_int('train.val_sphere_tracing_freq')
        # List of (scene_idx, image_idx) pairs. Example: [[0, 4], [1, 2]].
        # -1 for random. Examples: [-1] or [[0, 4], -1]
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.base_learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate_reduce_steps = \
            [int(x) for x in self.conf.get_list('train.learning_rate_reduce_steps')]
        self.learning_rate_reduce_factor = self.conf.get_float('train.learning_rate_reduce_factor')
        self.scenewise_layers_optimizer_extra_args = \
            dict(self.conf.get('train.scenewise_layers_optimizer_extra_args', default={}))
        self.cameras_optimizer_extra_args = \
            dict(self.conf.get('train.cameras_optimizer_extra_args', default={}))
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.restart_from_iter = self.conf.get_int('train.restart_from_iter', default=None)

        self.use_fp16 = self.conf.get_bool('train.use_fp16', default=False)

        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        if 'train.restart_from_iter' in self.conf:
            del self.conf['train']['restart_from_iter']

        self.finetune = self.conf.get_bool('train.finetune', default=False)

        PARTS_TO_TRAIN_ALL = set(['nerf_outside', 'sdf', 'deviation', 'color', 'cameras'])
        PARTS_TO_TRAIN_DEFAULT = PARTS_TO_TRAIN_ALL - set(['cameras'])

        parts_to_train = PARTS_TO_TRAIN_DEFAULT
        if 'train.parts_to_train' in self.conf:
            logging.warning(f"'train.parts_to_train' is deprecated, please migrate to 'train.parts_to_freeze'")
            parts_to_train = set(self.conf.get_list('train.parts_to_train'))
            for x in parts_to_train:
                assert x in PARTS_TO_TRAIN_ALL, f"Invalid entry in 'train.parts_to_train': {x}"
        if 'train.parts_to_freeze' in self.conf:
            parts_to_freeze = set(self.conf.get_list('train.parts_to_freeze', default=['cameras']))
            logging.info(f"Freezing these parts: {parts_to_freeze}")
            parts_to_train = PARTS_TO_TRAIN_ALL - parts_to_freeze
        logging.info(f"Will optimize only these parts: {parts_to_train}")

        load_optimizer = \
            self.conf.get_bool('train.load_optimizer', default=not self.finetune)
        parts_to_skip_loading = \
            self.conf.get_list('train.parts_to_skip_loading', default=[])
        # 'pick' or 'average'
        finetuning_init_algorithm = \
            self.conf.get_string('train.finetuning_init_algorithm', default='average')

        # For proper checkpoint auto-restarts
        for key in 'load_optimizer', 'restart_from_iter', 'parts_to_skip_loading':
            if f'train.{key}' in self.conf:
                del self.conf['train'][key]

        if self.finetune:
            assert self.dataset.num_scenes == 1, "Can only finetune to one scene"
            assert self.dataset_val.num_scenes == 1, "Can only finetune to one scene"

        # Loss weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight', default=0.0)
        self.radiance_grad_weight = self.conf.get_float('train.radiance_grad_weight', default=0.0)
        self.focal_fix_weight = self.conf.get_float('train.focal_fix_weight', default=0.1)
        self.semantic_consistency_weight = self.conf.get_float('train.semantic_consistency_weight',
            default=0.0)

        self.semantic_consistency_every_k_iterations = self.conf.get_int(
            'train.semantic_consistency_every_k_iterations', default=15)
        self.clip_resolution = self.conf.get_int('train.clip_resolution', default=224)
        self.semantic_loss_rendering_method = \
            self.conf.get_string('train.semantic_loss_rendering_method', default='sphere_tracing')
        self.semantic_grad_resolution_level = \
            self.conf.get_int('train.semantic_grad_resolution_level',
                default='semantic_grad_resolution_level')

        self.mode = args.mode
        self.model_list = []
        self.writer = None

        # Trainable networks and modules
        current_num_scenes = self.conf.get_int('dataset.original_num_scenes', default=self.dataset.num_scenes)
        self.nerf_outside = MultiSceneNeRF(**self.conf['model.nerf'], n_scenes=current_num_scenes).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], n_scenes=current_num_scenes).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'], n_scenes=current_num_scenes).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.trainable_camera_params = TrainableCameraParams([len(images) for images in self.dataset.images]).to(self.device)

        def get_optimizer(parts_to_train):
            """
            parts_to_train:
                iterable of str
                Valid strings: 'nerf_outside', 'sdf', 'deviation', 'color', 'cameras'.
            """
            class ScenePickingAdam(apex.optimizers.FusedAdam):
                @contextlib.contextmanager
                def limit_to_parameters_of_one_scene(self, scene_idx: int = None):
                    # Only retain SHARED and SCENEWISE-`scene_idx` param groups, remove the rest.
                    # If `scene_idx == None`, retain only SHARED parameters.
                    def should_pick_param_group(param_group):
                        name = param_group['group_name'].split('-')
                        if name[1] == 'SHARED':
                            return True
                        elif name[1] == 'SCENEWISE':
                            return int(name[2]) == scene_idx
                        else:
                            raise ValueError(f"Wrong group name: {name}")

                    all_param_groups = self.param_groups
                    picked_param_groups = \
                        [x for x in self.param_groups if should_pick_param_group(x)]

                    try:
                        self.param_groups = picked_param_groups
                        yield
                    finally:
                        self.param_groups = all_param_groups

            if self.scenewise_layers_optimizer_extra_args:
                logging.warning(
                    f"There are 'scenewise_layers_optimizer_extra_args'. " \
                    f"Will not apply them to bkgd NeRF and cameras: " \
                    f"{self.scenewise_layers_optimizer_extra_args}")

            def get_module_to_train(part_name):
                if part_name == 'nerf_outside':
                    return self.nerf_outside
                elif part_name == 'sdf':
                    return self.sdf_network
                elif part_name == 'deviation':
                    return self.deviation_network
                elif part_name == 'color':
                    return self.color_network
                elif part_name == 'cameras':
                    return self.trainable_camera_params
                else:
                    raise ValueError(f"Unknown 'parts_to_train': {part_name}")

            parameter_groups = []

            # Get optimizer groups for parameters that are SHARED between scenes
            total_tensors, total_parameters = 0, 0
            for part_name in sorted(parts_to_train):
                module_to_train = get_module_to_train(part_name)
                tensors_to_train = list(module_to_train.parameters('shared'))

                total_tensors += len(tensors_to_train)
                total_parameters += sum(x.numel() for x in tensors_to_train)

                parameter_group_settings = {
                    'group_name': f'{part_name}-SHARED',
                    'params': tensors_to_train,
                    'base_learning_rate': self.base_learning_rate}

                parameter_groups.append(parameter_group_settings)

            logging.info(
                f"Got {total_tensors} trainable SHARED tensors " \
                f" ({total_parameters} parameters total)")

            # Get optimizer groups for parameters that correspond to only ONE scene
            total_tensors, total_parameters = 0, 0
            for part_name in sorted(parts_to_train):
                module_to_train = get_module_to_train(part_name)

                num_scenes_in_model = len(self.nerf_outside)
                for scene_idx in range(num_scenes_in_model):
                    tensors_to_train = list(module_to_train.parameters('scenewise', scene_idx))

                    total_tensors += len(tensors_to_train)
                    total_parameters += sum(x.numel() for x in tensors_to_train)

                    parameter_group_settings = {
                        'group_name': f'{part_name}-SCENEWISE-{scene_idx}',
                        'params': tensors_to_train,
                        'base_learning_rate': self.base_learning_rate}

                    if part_name not in ('nerf_outside', 'cameras'):
                        parameter_group_settings.update(self.scenewise_layers_optimizer_extra_args)

                    parameter_groups.append(parameter_group_settings)

            logging.info(
                f"Got {total_tensors} trainable SCENEWISE tensors " \
                f" ({total_parameters} parameters total)")

            if self.cameras_optimizer_extra_args:
                logging.info(f"Will optimize cameras with: {self.cameras_optimizer_extra_args}")

                for parameter_group_settings in parameter_groups:
                    if parameter_group_settings['group_name'].startswith('cameras-'):
                        parameter_group_settings.update(self.cameras_optimizer_extra_args)

            return ScenePickingAdam(parameter_groups)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        if load_optimizer:
            self.optimizer = get_optimizer(parts_to_train)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint, parts_to_skip_loading, load_optimizer)

        if self.restart_from_iter is None:
            self.restart_from_iter = 0
        self.iter_step = self.restart_from_iter

        if self.finetune:
            with torch.no_grad():
                if finetuning_init_algorithm.startswith('pick'):
                    scene_idx = int(finetuning_init_algorithm.split('_')[1])
                    finetuning_init_algorithm = 'pick'
                else:
                    scene_idx = None

                self.sdf_network.switch_to_finetuning(finetuning_init_algorithm, scene_idx)
                self.color_network.switch_to_finetuning(finetuning_init_algorithm, scene_idx)
                self.deviation_network.switch_to_finetuning(finetuning_init_algorithm, scene_idx)
                self.nerf_outside.switch_to_finetuning(finetuning_init_algorithm, scene_idx)

        if not load_optimizer:
            self.optimizer = get_optimizer(parts_to_train)

        # Synchronize parameters (process #0 sends weights to all other processes)
        for network in self.sdf_network, self.color_network, \
                       self.nerf_outside, self.deviation_network:
            apex.parallel.Reducer(network)
        # Initialize `self._shared_parameters` for `self.average_shared_gradients_between_GPUs()`
        with self.optimizer.limit_to_parameters_of_one_scene(scene_idx=None):
            self._shared_parameters = sum([x['params'] for x in self.optimizer.param_groups], [])
        logging.info(
            f"There are {len(self._shared_parameters)} 'shared' parameters to average over GPUs " \
            f"at each step")

        # In case of finetuning
        self.conf['dataset']['original_num_scenes'] = self.dataset.num_scenes

        # Backup codes and configs for debug
        if self.mode[:5] == 'train' and self.rank == 0:
            self.file_backup()

        if self.semantic_consistency_weight > 0:
            logging.info(f"Loading CLIP")

            # Initialize CLIP model
            self.clip_model, _ = clip.load("ViT-B/32", device='cpu') # cpu because we need fp32
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.requires_grad_(False)

    # This is to average gradients of shared parameters after each iteration
    def average_shared_gradients_between_GPUs(self):
        if self.world_size == 1:
            return

        gradients = [x.grad for x in self._shared_parameters if x.grad is not None]
        with torch.no_grad():
            apex.parallel.distributed.flat_dist_call(gradients, torch.distributed.all_reduce)

    # This is to average model parameters between GPUs
    def average_parameters_between_GPUs(self):
        if self.world_size == 1:
            return

        parameters = sum([x['params'] for x in self.optimizer.param_groups], [])
        with torch.no_grad():
            apex.parallel.distributed.flat_dist_call(parameters, torch.distributed.all_reduce)

    def train(self):
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=self.base_exp_dir)

        res_step = self.end_iter - self.iter_step

        data_loader = iter(self.dataset.get_dataloader())
        logging.info("Starting training")

        for iter_i in tqdm(range(res_step), ncols=0):
            start_time = time.time()

            is_semantic_consistency_step = self.semantic_consistency_weight > 0 and \
                self.iter_step % self.semantic_consistency_every_k_iterations == 0

            # Special semantic consistency update iteration
            if is_semantic_consistency_step:
                scene_idx = 0

                # Compute reference CLIP embedding
                ref_embeddings = []

                for ref_image_idx in range(len(self.dataset.images[scene_idx])):
                    # Load image from dataset, cropped to head
                    crop = self.dataset.object_bboxes[scene_idx][ref_image_idx]
                    ref_image, ref_mask = self.dataset.get_image_and_mask(
                        scene_idx, ref_image_idx, crop=crop)

                    # Apply random background using dataset's mask
                    background_color = torch.rand(3)
                    ref_image = \
                        ref_mask * ref_image + (1.0 - ref_mask) * background_color[None, None]

                    # Compute and save CLIP embedding
                    ref_image = ref_image.permute(2, 0, 1) # HWC -> CHW
                    # TODO works only with ~square images, improve
                    assert 0.93 < ref_image.shape[1] / ref_image.shape[2] < 1.07
                    ref_image = torchvision.transforms.Resize(self.clip_resolution,
                        interpolation=torchvision.transforms.InterpolationMode.NEAREST)(ref_image)
                    ref_image = CLIP_INPUT_TRANSFORM(ref_image)
                    ref_image = ref_image.to(self.device)
                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(ref_image[None])[0].float()
                    embedding /= embedding.norm()
                    ref_embeddings.append(embedding)

                ref_embedding = torch.stack(ref_embeddings).mean(0)
                ref_embedding /= ref_embedding.norm()
                reference_semantic_embedding = ref_embedding

                # Compute CLIP embedding of the rendered shape
                INTRINSICS = torch.tensor([
                    [  3.53791131 * self.clip_resolution,   0.                 ,  0.49 * self.clip_resolution   ],
                    [  0.                 ,   3.53791131 * self.clip_resolution,  0.49 * self.clip_resolution   ],
                    [  0.                 ,   0.                 ,        1.        ]])
                POSE = torch.tensor(random.choice(POSES))

                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    rendered_rgb = self.render_view_by_pose(
                        0, self.clip_resolution, self.clip_resolution, POSE, INTRINSICS,
                        background_color[None], method=self.semantic_loss_rendering_method,
                        grad_resolution_level=self.semantic_grad_resolution_level)
                rendered_rgb = rendered_rgb.permute(2, 0, 1) # HWC -> CHW
                rendered_rgb = CLIP_INPUT_TRANSFORM(rendered_rgb)

                embedding_of_render = self.clip_model.encode_image(rendered_rgb[None])[0].float()
                embedding_of_render = embedding_of_render / embedding_of_render.norm()

                semantic_consistency_loss = \
                    ((reference_semantic_embedding - embedding_of_render) ** 2).mean()

                loss = self.semantic_consistency_weight * semantic_consistency_loss

            # Otherwise, normal NeuS update iteration
            else:
                if self.optimize_cameras:
                    scene_idx, (image_idx, camera_intrinsics, camera_extrinsics, \
                        pixels, true_rgb, mask) = next(data_loader)

                    pixels = pixels.to(self.device)
                    camera_extrinsics = camera_extrinsics.to(self.device) # 4, 4
                    camera_intrinsics = camera_intrinsics.to(self.device) # 4, 4

                    camera_intrinsics, camera_extrinsics = \
                        self.trainable_camera_params.apply_params_correction(
                            scene_idx, image_idx, camera_intrinsics, camera_extrinsics)
                    camera_intrinsics_inv = torch.inverse(camera_intrinsics)

                    rays_o, rays_d = self.dataset.gen_rays(
                        pixels, camera_extrinsics[:3, :4], camera_intrinsics_inv[:3, :3],
                        self.dataset.H, self.dataset.W)
                    near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
                else:
                    scene_idx, (rays_o, rays_d, true_rgb, mask, near, far) = next(data_loader)

                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)
                true_rgb = true_rgb.to(self.device)
                mask = mask.to(self.device)
                near = near.to(self.device)
                far = far.to(self.device)
                # ZERO = torch.zeros(1, 1).to(self.device)

                mask_sum = mask.sum() + 1e-5

                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])

                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    render_out = self.renderer.render(
                        rays_o, rays_d, near, far, scene_idx,
                        background_rgb=background_rgb,
                        cos_anneal_ratio=self.get_cos_anneal_ratio(),
                        compute_eikonal_loss=self.igr_weight > 0,
                        compute_radiance_grad_loss=self.radiance_grad_weight > 0)

                    color_fine = render_out['color_fine']
                    s_val = render_out['s_val']
                    cdf_fine = render_out['cdf_fine']
                    gradients = render_out['gradients']
                    weight_max = render_out['weight_max']
                    weight_sum = render_out['weight_sum']

                    loss = 0

                    # Image loss: L1
                    target_rgb = true_rgb
                    if self.mask_weight > 0.0:
                        color_fine_loss = ((color_fine - target_rgb) * mask).abs().mean()
                    else:
                        target_rgb = true_rgb
                        color_fine_loss = (color_fine - target_rgb).abs().mean()
                    loss += color_fine_loss

                    psnr_train = psnr(color_fine, true_rgb, mask)

                    # Mask loss: BCE
                    if self.mask_weight > 0.0:
                        mask_loss = torch.nn.functional.binary_cross_entropy(
                            weight_sum.clip(1e-5, 1.0 - 1e-5), mask)
                        loss += mask_loss * self.mask_weight

                    # SDF loss: eikonal
                    if self.igr_weight > 0:
                        # boolean mask, 1 if point is inside 1-sphere
                        relax_inside_sphere = render_out['relax_inside_sphere']

                        gradients_eikonal = render_out['gradients_eikonal']
                        gradient_error = (torch.linalg.norm(gradients_eikonal, ord=2, dim=-1) - 1.0) ** 2
                        eikonal_loss = \
                            (relax_inside_sphere * gradient_error).sum() / \
                            (relax_inside_sphere.sum() + 1e-5)
                        loss += eikonal_loss * self.igr_weight

                    # Radiance loss: gradient orthogonality
                    if self.radiance_grad_weight > 0:
                        # dim 0: gradient of r/g/b
                        # dim 1: point number
                        # dim 2: gradient over x/y/z
                        gradients_radiance = render_out['gradients_radiance'] # 3, K, 3
                        gradients_eikonal_ = gradients_eikonal[None] # 1, K, 3

                        # We want these gradients to be orthogonal, so force dot product to zero
                        grads_dot_product = (gradients_eikonal * gradients_radiance).sum(-1) # 3, K
                        radiance_grad_loss = (grads_dot_product ** 2).mean()
                        # radiance_grad_loss = torch.nn.HuberLoss(delta=0.0122)(
                        #     grads_dot_product, ZERO.expand_as(grads_dot_product))
                        loss += radiance_grad_loss * self.radiance_grad_weight

                    if self.optimize_cameras and self.focal_fix_weight > 0:
                        focal_fix_loss = \
                            (self.trainable_camera_params.log_focal_dist_delta[scene_idx] ** 2).mean()
                        loss += focal_fix_loss * self.focal_fix_weight

            # The return values are only needed for logging
            learning_rate_shared, learning_rate_scenewise = self.update_learning_rate()
            self.optimizer.zero_grad()
            self.gradient_scaler.scale(loss).backward()
            self.average_shared_gradients_between_GPUs()
            # TODO: create a separate scaler for every scene and for shared parameters
            with self.optimizer.limit_to_parameters_of_one_scene(scene_idx=scene_idx):
                self.gradient_scaler.step(self.optimizer)

            self.gradient_scaler.update()

            if iter_i % 100 == 0:
                self.average_parameters_between_GPUs()

            step_time = time.time() - start_time

            with torch.no_grad():
                self.iter_step += 1

                if self.rank == 0:
                    self.writer.add_scalar('Loss/Total', loss, self.iter_step)
                    if is_semantic_consistency_step:
                        self.writer.add_scalar('Loss/Semantic', semantic_consistency_loss, self.iter_step)
                    else:
                        self.writer.add_scalar('Loss/L1', color_fine_loss, self.iter_step)
                        if self.radiance_grad_weight > 0:
                            self.writer.add_scalar('Loss/Eikonal', eikonal_loss, self.iter_step)
                            self.writer.add_scalar('Loss/<dRGB,dSDF>', radiance_grad_loss, self.iter_step)
                        self.writer.add_scalar('Loss/PSNR (train)', psnr_train, self.iter_step)
                        self.writer.add_scalar('Statistics/s_val', s_val.item(), self.iter_step)
                        self.writer.add_scalar('Statistics/Learning rate', learning_rate_shared, self.iter_step)
                        self.writer.add_scalar('Statistics/Learning rate (scenewise)', learning_rate_scenewise, self.iter_step)
                        self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                        self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                        if self.optimize_cameras:
                            self.writer.add_scalar('Statistics/Focal distance correction 0',
                                self.trainable_camera_params.log_focal_dist_delta[0].exp(), self.iter_step)
                            self.writer.add_scalar('Statistics/Camera se(3) correction (abs max)',
                                self.trainable_camera_params.pose_se3_delta[0].abs().max(), self.iter_step)

                    self.writer.add_scalar(
                        'Statistics/Step time' + ', semantic' * is_semantic_consistency_step,
                        step_time, self.iter_step)

                    if self.iter_step % self.report_freq == 0:
                        print(self.base_exp_dir)
                        print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                    if self.iter_step % self.save_freq == 0 or self.iter_step == self.end_iter:
                        self.save_checkpoint()

                    if self.iter_step % self.val_sphere_tracing_freq == 0 or self.iter_step == self.end_iter or self.iter_step == 1:
                        self.validate_images_sphere_tracing()

                    if self.iter_step % self.val_freq == 0 or self.iter_step == self.end_iter or self.iter_step == 1:
                        self.validate_images()

                        if self.optimize_cameras:
                            # Display camera positions
                            camera_intrinsics = self.dataset.intrinsics_all[0][0].to(self.device)
                            camera_extrinsics_all = []

                            for camera_extrinsics in self.dataset.pose_all[0].to(self.device):
                                current_intrinsics, current_extrinsics = \
                                    self.trainable_camera_params.apply_params_correction(
                                        0, 0, camera_intrinsics, camera_extrinsics)
                                camera_extrinsics_all.append(current_extrinsics)

                            camera_extrinsics_all = torch.stack(camera_extrinsics_all)
                            # self.writer.add_mesh(
                            #     'Statistics/Cameras', camera_extrinsics_all[None, :3, 3],
                            #     config_dict={
                            #         'material': {
                            #             'cls': 'PointsMaterial',
                            #             'size': 0.1
                            #     }},
                            #     global_step=self.iter_step)
                            cameras_save_dir = os.path.join(self.base_exp_dir, "cameras")
                            os.makedirs(cameras_save_dir, exist_ok=True)
                            torch.save({
                                'intrinsics': current_intrinsics,
                                'extrinsics': camera_extrinsics_all,
                            }, os.path.join(cameras_save_dir, f"{self.iter_step:07}.pth"))

                    if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == self.end_iter:
                        self.validate_mesh(
                            world_space=True,
                            resolution=512 if self.iter_step == self.end_iter else 64)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        # These return values are for logging only
        lr_shared, lr_scenewise = np.nan, np.nan

        learning_rate_factor = 1.0

        if self.iter_step - self.restart_from_iter < self.warm_up_end:
            learning_rate_factor *= (self.iter_step - self.restart_from_iter) / self.warm_up_end

        alpha = self.learning_rate_alpha
        progress = self.iter_step / self.end_iter
        learning_rate_factor *= (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for reduce_step in self.learning_rate_reduce_steps:
            if self.iter_step >= reduce_step:
                learning_rate_factor *= self.learning_rate_reduce_factor

        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate_factor * g['base_learning_rate']

            # TEMPORARY experiment
            if g['group_name'].startswith('cameras-'):
                def cameras_lr_factor(progress):
                    LEFT = 0.7
                    RIGHT = 0.71
                    return 1.0 # max(0, min(1.0, (progress - LEFT) / (RIGHT - LEFT)))

                g['lr'] *= cameras_lr_factor(progress)
            else:
                # For logging only
                if np.isnan(lr_shared) and 'SHARED' in g['group_name']:
                    lr_shared = g['lr']
                if np.isnan(lr_scenewise) and 'SCENEWISE' in g['group_name']:
                    lr_scenewise = g['lr']

        return lr_shared, lr_scenewise

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        with open(os.path.join(self.base_exp_dir, 'recording', 'config.conf'), 'w') as f:
            f.write(HOCONConverter().to_hocon(self.conf))

    def load_checkpoint(self,
        checkpoint: dict, parts_to_skip_loading: List[str] = [], load_optimizer: bool = True):

        def load_weights(module, state_dict):
            # backward compatibility:
            # replace 'lin(0-9)' (old convention) with 'linear_layers.' (new convention)
            for tensor_name in list(state_dict.keys()):
                if tensor_name.startswith('lin') and tensor_name[3].isdigit():
                    new_tensor_name = f'linear_layers.{tensor_name[3:]}'
                    state_dict[new_tensor_name] = state_dict[tensor_name]
                    del state_dict[tensor_name]

            module.load_state_dict(state_dict)
            # missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

            # if missing_keys:
            #     raise RuntimeError(
            #         f"Missing keys in checkpoint: {missing_keys}\n" \
            #         f"Unexpected keys: {unexpected_keys}")
            # if unexpected_keys:
            #     logging.warning(f"Ignoring unexpected keys in checkpoint: {unexpected_keys}")

        if parts_to_skip_loading != []:
            logging.warning(f"Not loading weights for {parts_to_skip_loading}")

        if 'nerf_outside' not in parts_to_skip_loading:
            load_weights(self.nerf_outside, checkpoint['nerf'])
        if 'sdf' not in parts_to_skip_loading:
            load_weights(self.sdf_network, checkpoint['sdf_network_fine'])
        if 'deviation' not in parts_to_skip_loading:
            load_weights(self.deviation_network, checkpoint['variance_network_fine'])
        if 'color' not in parts_to_skip_loading:
            load_weights(self.color_network, checkpoint['color_network_fine'])
        if 'cameras' not in parts_to_skip_loading:
            if 'trainable_camera_params' in checkpoint:
                load_weights(self.trainable_camera_params, checkpoint['trainable_camera_params'])

        if load_optimizer:
            # Don't let overwrite custom keys
            param_groups_ckpt = \
                {g['group_name']: g for g in checkpoint['optimizer']['param_groups']}

            for param_group in self.optimizer.param_groups:
                for hyperparameter in 'base_learning_rate', 'weight_decay':
                    param_groups_ckpt[param_group['group_name']][hyperparameter] = \
                        param_group[hyperparameter]

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.use_fp16:
                if 'gradient_scaler' in checkpoint:
                    self.gradient_scaler.load_state_dict(checkpoint['gradient_scaler'])
                else:
                    logging.warning(f"`use_fp16 == True`, but no `gradient_scaler` in checkpoint!")

        if self.restart_from_iter is None:
            self.restart_from_iter = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'trainable_camera_params': self.trainable_camera_params.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'config': self.conf,
        }
        if self.use_fp16:
            checkpoint['gradient_scaler'] = self.gradient_scaler.state_dict()

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', f'ckpt_{self.iter_step:07d}.pth'))

    def validate_images(self, resolution_level=None):
        logging.info('Validation. Iter: {}'.format(self.iter_step))

        if resolution_level is None:
            resolution_level = self.validate_resolution_level

        render_images = []
        normal_images = []
        psnr_val = []

        for val_scene_idx in range(self.dataset_val.num_scenes):
            num_images_per_scene = len(self.dataset_val.images[val_scene_idx])
            for val_image_idx in range(num_images_per_scene):
                rays_o, rays_d, true_rgb, mask, near, far = self.dataset_val.gen_rays_at(
                    val_scene_idx, val_image_idx, resolution_level=resolution_level,
                    camera_params_correction=self.trainable_camera_params \
                    if self.apply_camera_correction_to_val else None)

                H, W, _ = rays_o.shape
                rays_o = rays_o.to(self.device).reshape(-1, 3).split(self.batch_size)
                rays_d = rays_d.to(self.device).reshape(-1, 3).split(self.batch_size)
                near = near.to(self.device).reshape(-1, 1).split(self.batch_size)
                far = far.to(self.device).reshape(-1, 1).split(self.batch_size)

                out_rgb_fine = []
                out_normal_fine = []

                for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
                    background_rgb = \
                            torch.ones([1, 3], device=rays_o.device) if self.use_white_bkgd else None

                    render_out = self.renderer.render(rays_o_batch,
                                                      rays_d_batch,
                                                      near_batch,
                                                      far_batch,
                                                      val_scene_idx,
                                                      cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                      background_rgb=background_rgb,
                                                      compute_eikonal_loss=False,
                                                      compute_radiance_grad_loss=False)

                    def feasible(key): return (key in render_out) and (render_out[key] is not None)

                    if feasible('color_fine'):
                        out_rgb_fine.append(render_out['color_fine'].cpu())
                    if feasible('gradients') and feasible('weights'):
                        n_samples = self.renderer.n_samples + self.renderer.n_importance
                        normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                        if feasible('inside_sphere'):
                            normals = normals * render_out['inside_sphere'][..., None]
                        normals = normals.sum(dim=1).cpu()
                        out_normal_fine.append(normals)
                    del render_out

                img_fine = torch.cat(out_rgb_fine).reshape(H, W, 3).clamp(0.0, 1.0)

                normal_img = torch.cat(out_normal_fine)
                rot = torch.inverse(self.dataset_val.pose_all[val_scene_idx][val_image_idx, :3, :3].cpu())
                normal_img = ((rot[None, :, :] @ normal_img[:, :, None]).reshape(H, W, 3) * 0.5 + 0.5)
                normal_img = normal_img.clamp(0.0, 1.0)

                # os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
                # os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

                psnr_val.append(psnr(img_fine, true_rgb, mask))

                render_image = torch.cat((img_fine, true_rgb), dim=1)
                normal_image = normal_img

                render_images.append(render_image)
                normal_images.append(normal_image)

        def pad(image, width):
            return torch.nn.functional.pad(image, (0, 0, 0, width - image.shape[1]))

        max_W = max(image.shape[1] for image in normal_images)
        render_images = [pad(image, max_W * 2) for image in render_images]
        normal_images = [pad(image, max_W)     for image in normal_images]

        render_images = cv2.cvtColor(torch.cat(render_images).numpy(), cv2.COLOR_BGR2RGB)
        normal_images = cv2.cvtColor(torch.cat(normal_images).numpy(), cv2.COLOR_BGR2RGB)

        if self.rank == 0:
            self.writer.add_image(
                'Image/Render (val)', render_images, self.iter_step, dataformats='HWC')
            self.writer.add_image(
                'Image/Normals (val)', normal_images, self.iter_step, dataformats='HWC')
            self.writer.add_scalar('Loss/PSNR (val)', np.mean(psnr_val), self.iter_step)

    def validate_images_sphere_tracing(self):
        # Render side view by sphere tracing
        LIGHT_PURPLE = torch.tensor([[255., 192., 203.]]) / 255
        H, W = 224, 224
        INTRINSICS_VAL = torch.tensor([
            [  3.53791131 * H,   0.                 ,  0.49 * H   ],
            [  0.                 ,   3.53791131 * W,  0.49 * W   ],
            [  0.                 ,   0.                 ,        1.        ]])
        POSE = torch.tensor(POSES[4]) # left view

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            val_view_sphere_tracing = self.render_view_by_pose(
                0, H, W, POSE, INTRINSICS_VAL, LIGHT_PURPLE)

        val_view_sphere_tracing = cv2.cvtColor(
            val_view_sphere_tracing.float().cpu().numpy(), cv2.COLOR_BGR2RGB)

        if self.rank == 0:
            self.writer.add_image(
                'Image/Render (val sph)', val_view_sphere_tracing,
                self.iter_step, dataformats='HWC')

    def render_view_by_pose(self, scene_idx, h, w, pose, intrinsics, background_color=None,
        method='sphere_tracing', grad_resolution_level=1):
        """
        Pose can optionally be obtained from Blender. Open any mesh obtained by running
        this code with `--mode validate_mesh_XX` and export the pose with this plugin:
        https://github.com/Cartucho/vision_blender/. The plugin will output two files,
        'XXXX.npz' (denoted `npz` below) and 'camera_info.json'.

        scene_idx
            int
            Refers to a scene in `self.dataset`.
        h, w
            int
            `npz['normal_map'].shape[:2]`, or get it from 'camera_info.json'.
        pose_blender
            torch.tensor, float32, shape == (3, 4)
            `npz['extrinsic_mat']`.
        intrinsics
            torch.tensor, float32, shape == (3, 3)
            `npz['intrinsic_mat']`.
        background_color
            torch.tensor, float32, shape == (1, 3)
            RGB, 0..1
        method
            str
            'volume_rendering' or 'sphere_tracing'
        grad_resolution_level
            int
            1(default) = require grad for every pixel of output image,
            2 = for every other row/col etc.
        """
        background_rgb = torch.zeros(1, 3) # black
        if background_color is not None:
            background_rgb.copy_(background_color)
        background_rgb = background_rgb.to(self.device)

        tx = torch.linspace(0, w - 1, w)
        ty = torch.linspace(0, h - 1, h)
        pixels = torch.stack(torch.meshgrid(tx, ty, indexing='xy'), dim=-1) # h, w, 2

        intrinsics_inv = intrinsics.inverse()

        def augment_to_4x4(m):
            if m.shape[1] == 3:
                m = torch.cat([m, torch.tensor([[0, 0, 0]]).T], dim=1)
            if m.shape[0] == 3:
                m = torch.cat([m, torch.tensor([[0, 0, 0, 1]])])
            return m

        # Apply the "scale mat" transform to the given pose.
        # This transform moves 'reference landmarks' to the origin and scales them.
        proj_matrix = augment_to_4x4(intrinsics) @ augment_to_4x4(pose)
        proj_matrix @= torch.tensor(self.dataset.scale_mats_np[scene_idx][0])

        _, pose = load_K_Rt_from_P(None, proj_matrix[:3].numpy())
        pose = torch.tensor(pose)[:3]

        rays_o, rays_d = self.dataset.gen_rays(pixels, pose, intrinsics_inv, h, w) # h, w, 3
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)
        near = near.to(self.device)
        far = far.to(self.device)

        if method == 'volume_rendering':
            img_rendered = torch.empty(h, w, 3, device=self.device)

            grad_mask_ = torch.zeros(h, w, dtype=torch.bool, device=self.device)
            grad_mask_[::grad_resolution_level, ::grad_resolution_level] = True

            for require_grad in (True, False):
                grad_mask = grad_mask_ if require_grad else ~grad_mask_

                rays_o_ = rays_o[grad_mask].reshape(-1, 3).split(self.batch_size)
                rays_d_ = rays_d[grad_mask].reshape(-1, 3).split(self.batch_size)
                near_ = near[grad_mask].reshape(-1, 1).split(self.batch_size)
                far_ = far[grad_mask].reshape(-1, 1).split(self.batch_size)

                with contextlib.nullcontext() if require_grad else torch.no_grad():
                    rgb_all = []

                    total_batches_computed = 0
                    for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o_, rays_d_, near_, far_):
                        # gpu_total = torch.cuda.get_device_properties(0).total_memory // 1024**2
                        # gpu_reserved = torch.cuda.memory_reserved(0) // 1024**2
                        # gpu_allocated = torch.cuda.memory_allocated(0) // 1024**2
                        # print(f"Computing batch {total_batches_computed} / {len(rays_o_)}, "
                        #       f"GPU alloc/reserved/total {gpu_allocated}/{gpu_reserved}/{gpu_total} MB"); total_batches_computed += 1

                        render_out = self.renderer.render(rays_o_batch,
                                                          rays_d_batch,
                                                          near_batch,
                                                          far_batch,
                                                          scene_idx,
                                                          cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                          background_rgb=background_rgb,
                                                          compute_eikonal_loss=False,
                                                          compute_radiance_grad_loss=False)

                        rgb_all.append(render_out['color_fine'])
                        del render_out

                    rgb_all = torch.cat(rgb_all) # float32, 0..1

                img_rendered.masked_scatter_(grad_mask[..., None], rgb_all)

        elif method == 'sphere_tracing':
            if grad_resolution_level != 1:
                raise NotImplementedError(
                    f"grad_resolution_level is {grad_resolution_level}, but only 1 " \
                    f"is supported for method == 'sphere_tracing'")

            batch_size = 1024 * 32 # A larger batch size thanks to smaller tasks
            rays_o = rays_o.reshape(-1, 3).split(batch_size)
            rays_d = rays_d.reshape(-1, 3).split(batch_size)
            near = near.reshape(-1).split(batch_size)
            far = far.reshape(-1).split(batch_size)

            img_rendered = []

            for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
                depth = near_batch.clone()
                valid_mask = torch.ones_like(depth, dtype=torch.bool, device=self.device)

                with torch.no_grad():
                    N_ITERS = 20
                    for _ in range(N_ITERS):
                        points = rays_o_batch + rays_d_batch * depth[..., None]
                        sdf = self.sdf_network.sdf(points, scene_idx) # TODO predict only for valid pts
                        depth[valid_mask] += sdf[valid_mask][..., 0]
                        valid_mask[(depth > far_batch) | (depth < 0)] = False

                points = rays_o_batch + rays_d_batch * depth[..., :, None] # batch_size, 3
                gradients, sdf, feature_vectors = self.sdf_network.gradient(points, scene_idx)
                radiance = self.color_network(
                    points, gradients, rays_d_batch, feature_vectors, scene_idx)

                # PyTorch didn't like the inplace operation, so...
                radiance = radiance * valid_mask[..., None] + background_rgb * (~valid_mask)[..., None]
                # radiance_with_bkgd = torch.empty_like(radiance)
                # radiance_with_bkgd[valid_mask] = radiance[valid_mask]
                # radiance_with_bkgd[~valid_mask] = background_rgb

                img_rendered.append(radiance)

            img_rendered = torch.cat(img_rendered).reshape(h, w, 3)

        else:
            raise ValueError(f"Unknown rendering method: '{method}'")

        return img_rendered

    def render_interpolated_view(self, scene_idx, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d, near, far, pose = self.dataset.gen_rays_between(
            scene_idx, idx_0, idx_1, ratio, resolution_level=resolution_level)

        H, W, _ = rays_o.shape
        rays_o = rays_o.to(self.device).reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.to(self.device).reshape(-1, 3).split(self.batch_size)
        near = near.to(self.device).reshape(-1, 1).split(self.batch_size)
        far = far.to(self.device).reshape(-1, 1).split(self.batch_size)

        rgb_all = []
        normals_all = []

        for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near_batch,
                                              far_batch,
                                              scene_idx,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,
                                              compute_eikonal_loss=False,
                                              compute_radiance_grad_loss=False)

            rgb_all.append(render_out['color_fine'].detach().cpu().numpy())

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            n_samples = self.renderer.n_samples + self.renderer.n_importance
            normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
            if feasible('inside_sphere'):
                normals = normals * render_out['inside_sphere'][..., None]
            normals = normals.sum(dim=1).cpu().numpy()
            normals_all.append(normals)

            del render_out

        img_fine = (np.concatenate(rgb_all, axis=0).reshape([H, W, 3]) * 255).clip(0, 255).astype(np.uint8)

        normal_img = np.concatenate(normals_all)
        rot = np.asarray(pose[:3, :3])
        normal_img = ((rot[None, :, :] @ normal_img[:, :, None]).reshape(H, W, 3) * 0.5 + 0.5)
        normal_img = (normal_img.clip(0.0, 1.0) * 255).astype(np.uint8)

        return img_fine, normal_img

    def validate_mesh(self, scene_idx=0, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = self.renderer.extract_geometry(
            scene_idx, bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        vertex_colors = []
        k = 10000
        with torch.no_grad():
            for i in range(vertices.shape[0] // k + 1):
                if i * k >= vertices.shape[0]:
                    break
                pts = torch.tensor(vertices[i * k: (i + 1) * k], dtype=torch.float32, device=self.device)
                sdf_nn_output = self.sdf_network(pts, scene_idx)
                feature_vector = sdf_nn_output[:, 1:]
                gradients, _, _ = self.sdf_network.gradient(pts, scene_idx)
                gradients = gradients.squeeze()
                dirs = gradients
                vertex_colors.append(self.color_network(pts, gradients, dirs, feature_vector, scene_idx).cpu().numpy())
        vertex_colors = np.concatenate(vertex_colors, axis=0)
        vertex_colors = np.uint8(np.round(vertex_colors * 255))
        vertex_colors = vertex_colors[:, ::-1]  # bgr to rgb

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[scene_idx][0][0, 0] + \
                self.dataset.scale_mats_np[scene_idx][0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)

        # if world_space and self.dataset.reg_mats_np is not None:
        #     mesh.apply_transform(self.dataset.reg_mats_np[scene_idx][0])

        mesh.export(os.path.join(
            self.base_exp_dir, 'meshes', '{}_{:0>8d}.ply'.format(scene_idx, self.iter_step)))

        logging.info('End')

    def interpolate_view(self, scene_idx, img_idx_0, img_idx_1, n_frames=20):
        # image, normals = self.render_interpolated_view(
        #     scene_idx, None, None, 0.0, resolution_level=2)
        # frame = np.concatenate([image, normals], axis=0)
        # cv2.imwrite(
        #     os.path.join(
        #         self.base_exp_dir,
        #         'render',
        #         f"{self.iter_step:0>8d}_scene{scene_idx:03d}_blender9simplify.png"),
        #     frame)
        # return

        assert n_frames > 1
        images = []
        for i in tqdm(range(n_frames), ncols=0):
            image, normals = self.render_interpolated_view(
                scene_idx, img_idx_0, img_idx_1,
                np.sin(((i / (n_frames - 1)) - 0.5) * np.pi) * 0.5 + 0.5, resolution_level=2)
            video_frame = np.concatenate([image, normals], axis=0)

            images.append(video_frame)
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv2.VideoWriter(
            os.path.join(
                video_dir,
                f"{self.iter_step:0>8d}_scene{scene_idx:03d}_{img_idx_0}_{img_idx_1}.mp4"),
            fourcc, 15, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

def seed_RNGs(seed):
    logging.info(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    print('Hello Wooden')

    FORMAT = "PID %(process)d - %(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=pathlib.Path, default=None)
    parser.add_argument('--checkpoint_path', type=pathlib.Path, default=None)
    parser.add_argument('--extra_config_args', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)

    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.rank = int(os.environ.get('RANK', 0))
    args.num_gpus = args.world_size = int(os.environ.get('WORLD_SIZE', 1))

    if args.rank > 0 and args.mode != 'train':
        logger.warning(f"`--mode` != 'train', shutting down all processes but one")

    # Multi-GPU training
    torch.cuda.set_device(args.local_rank)
    args.device = f'cuda:{args.local_rank}'
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    seed_RNGs(args.rank)

    runner = Runner(args)

    if args.mode == 'train':
        runner.train()
    elif args.mode.startswith('validate_mesh'):
        try:
            scene_idx = int(args.mode.split('_')[2])
        except IndexError:
            scene_idx = 0

        with torch.no_grad():
            runner.validate_mesh(
                scene_idx=scene_idx, world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate_'):
        # Interpolate views, given [optional: scene index and] two image indices
        arguments = args.mode.split('_')[1:]
        if len(arguments) == 2:
            scene_idx, (img_idx_0, img_idx_1) = 0, map(int, arguments)
        elif len(arguments) == 3:
            scene_idx, img_idx_0, img_idx_1 = map(int, arguments)
        else:
            raise ValueError(f"Wrong number of '_' arguments (must be 3 or 4): {args.mode}")

        with torch.no_grad():
            runner.interpolate_view(scene_idx, img_idx_0, img_idx_1)
    else:
        raise ValueError(f"Wrong '--mode': {args.mode}")
