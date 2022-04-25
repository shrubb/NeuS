from typing import List

from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, MultiSceneNeRF
from models.renderer import NeuSRenderer

import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import apex
from tqdm import tqdm
from pyhocon import ConfigFactory, ConfigTree
from pyhocon.converter import HOCONConverter

import random
import pathlib
import os
import time
import logging
import argparse
from shutil import copyfile
import contextlib

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
        self.dataset = Dataset(self.conf['dataset'], kind='train')
        self.dataset_val = Dataset(self.conf['dataset'], kind='val')

        logging.info(f"Experiment dir: {self.base_exp_dir}")

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
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
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.restart_from_iter = self.conf.get_int('train.restart_from_iter', default=None)

        self.use_fp16 = self.conf.get_bool('train.use_fp16', default=False)

        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        if 'train.restart_from_iter' in self.conf:
            del self.conf['train']['restart_from_iter']

        self.finetune = self.conf.get_bool('train.finetune', default=False)
        parts_to_train = \
            self.conf.get_list('train.parts_to_train', default=[])
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

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight', default=0.0)
        self.radiance_grad_weight = self.conf.get_float('train.radiance_grad_weight', default=0.0)

        self.mode = args.mode
        self.model_list = []
        self.writer = None

        # Networks
        current_num_scenes = self.conf.get_int('dataset.original_num_scenes', default=self.dataset.num_scenes)
        self.nerf_outside = MultiSceneNeRF(**self.conf['model.nerf'], n_scenes=current_num_scenes).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], n_scenes=current_num_scenes).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'], n_scenes=current_num_scenes).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)

        def get_optimizer(parts_to_train):
            """
            parts_to_train:
                list of str
                If [], train all parts: 'nerf_outside', 'sdf', 'deviation', 'color'.
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

            if parts_to_train == []:
                parts_to_train = ['nerf_outside', 'sdf', 'deviation', 'color']
            else:
                logging.warning(f"Will optimize only these parts: {parts_to_train}")

            if self.scenewise_layers_optimizer_extra_args:
                logging.warning(
                    f"There are 'scenewise_layers_optimizer_extra_args'. " \
                    f"Will not apply them to bkgd NeRF: " \
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
                else:
                    raise ValueError(f"Unknown 'parts_to_train': {part_name}")

            parameter_groups = []

            # Get optimizer groups for parameters that are SHARED between scenes
            total_tensors, total_parameters = 0, 0
            for part_name in parts_to_train:
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
            for part_name in parts_to_train:
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

                    if part_name != 'nerf_outside':
                        parameter_group_settings.update(self.scenewise_layers_optimizer_extra_args)

                    parameter_groups.append(parameter_group_settings)

            logging.info(
                f"Got {total_tensors} trainable SCENEWISE tensors " \
                f" ({total_parameters} parameters total)")

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
                self.sdf_network.switch_to_finetuning(finetuning_init_algorithm)
                self.color_network.switch_to_finetuning(finetuning_init_algorithm)
                self.deviation_network.switch_to_finetuning(finetuning_init_algorithm)
                self.nerf_outside.switch_to_finetuning(finetuning_init_algorithm)

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

        for iter_i in tqdm(range(res_step)):
            start_time = time.time()

            scene_idx, (rays_o, rays_d, true_rgb, mask, near, far) = next(data_loader)

            rays_o = rays_o.cuda()
            rays_d = rays_d.cuda()
            true_rgb = true_rgb.cuda()
            mask = mask.cuda()
            near = near.cuda()
            far = far.cuda()
            # ZERO = torch.zeros(1, 1).cuda()

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

                # Image loss
                color_error = color_fine - true_rgb
                if self.mask_weight > 0.0:
                    color_error *= mask

                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum')
                if self.mask_weight > 0.0:
                    color_fine_loss /= mask_sum
                else:
                    color_fine_loss /= color_error.numel()
                loss += color_fine_loss

                psnr_train = psnr(color_fine, true_rgb, mask)

                # Mask loss
                if self.mask_weight > 0.0:
                    mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
                    loss += mask_loss * self.mask_weight

                # Eikonal loss
                if self.igr_weight > 0:
                    # boolean mask, 1 if point is inside 1-sphere
                    relax_inside_sphere = render_out['relax_inside_sphere']

                    gradients_eikonal = render_out['gradients_eikonal']
                    gradient_error = (torch.linalg.norm(gradients_eikonal, ord=2, dim=-1) - 1.0) ** 2
                    eikonal_loss = \
                        (relax_inside_sphere * gradient_error).sum() / \
                        (relax_inside_sphere.sum() + 1e-5)
                    loss += eikonal_loss * self.igr_weight

                # Radiance gradient loss
                if self.radiance_grad_weight > 0:
                    # dim 0: gradient of r/g/b
                    # dim 1: point number
                    # dim 2: gradient over x/y/z
                    gradients_radiance = render_out['gradients_radiance'] # 3, K, 3
                    gradients_eikonal_ = gradients_eikonal.detach()[None] # 1, K, 3

                    # We want these gradients to be orthogonal, so force dot product to zero
                    grads_dot_product = (gradients_eikonal * gradients_radiance).sum(-1) # 3, K
                    radiance_grad_loss = (grads_dot_product ** 2).mean()
                    # radiance_grad_loss = torch.nn.HuberLoss(delta=0.0122)(
                    #     grads_dot_product, ZERO.expand_as(grads_dot_product))
                    loss += radiance_grad_loss * self.radiance_grad_weight

            # These values are only needed for logging
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
                    self.writer.add_scalar('Statistics/Step time', step_time, self.iter_step)

                    if self.iter_step % self.report_freq == 0:
                        print(self.base_exp_dir)
                        print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                    if self.iter_step % self.save_freq == 0 or self.iter_step == self.end_iter:
                        self.save_checkpoint()

                    if self.iter_step % self.val_freq == 0 or self.iter_step == self.end_iter or self.iter_step == 1:
                        self.validate_images()

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
                    val_scene_idx, val_image_idx, resolution_level=resolution_level)

                H, W, _ = rays_o.shape
                rays_o = rays_o.cuda().reshape(-1, 3).split(self.batch_size)
                rays_d = rays_d.cuda().reshape(-1, 3).split(self.batch_size)
                near = near.cuda().reshape(-1, 1).split(self.batch_size)
                far = far.cuda().reshape(-1, 1).split(self.batch_size)

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

    def render_novel_image(self, scene_idx, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d, near, far, pose = self.dataset.gen_rays_between(
            scene_idx, idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.cuda().reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.cuda().reshape(-1, 3).split(self.batch_size)
        near = near.cuda().reshape(-1, 1).split(self.batch_size)
        far = far.cuda().reshape(-1, 1).split(self.batch_size)

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
        rot = pose[:3, :3]
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

        if world_space and self.dataset.reg_mats_np is not None:
            mesh.apply_transform(self.dataset.reg_mats_np[scene_idx][0])

        mesh.export(os.path.join(
            self.base_exp_dir, 'meshes', '{}_{:0>8d}.ply'.format(scene_idx, self.iter_step)))

        logging.info('End')

    def interpolate_view(self, scene_idx, img_idx_0, img_idx_1, n_frames=30):
        assert n_frames > 1
        images = []
        for i in tqdm(range(n_frames)):
            image, normals = self.render_novel_image(
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
                scene_idx=scene_idx, world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate_'):
        # Interpolate views given [optional: scene index and] two image indices
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
