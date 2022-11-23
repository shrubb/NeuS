import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.embedder import get_embedder
import utils.camera

import logging

class LowRankMultiLinear(nn.Module):
    """N linear layers whose weights are linearly regressed at forward pass from smaller
    matrices, leading to "lower rank" (lower DoF) parametrization.
    """
    def __init__(self, n_scenes, in_dim, out_dim, rank, weight_norm=False, use_bias=False):
        """
        rank:
            How many instances of weights (`P`) are learned.
            The weights of all `n_scenes` layers will be computed as their linear combinations.
        """
        super().__init__()
        assert rank > 0
        self.use_bias = use_bias
        self.use_weight_norm = weight_norm

        self.linear_layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_scenes)])

        basis_weights = {}
        for parameter_name, parameter in self.linear_layers[0].named_parameters():
            basis_weights[parameter_name] = nn.Parameter(
                torch.empty(parameter.shape + (rank + use_bias,)))

        self.basis_weights = nn.ParameterDict(basis_weights) # 'weight': out_dim x in_dim x rank[+1]
        self.combination_coeffs = nn.ParameterList(
            [nn.Parameter(torch.empty(rank)) for _ in range(n_scenes)]) # n_scenes x rank

        self.reset_parameters_()
        self.finetuning = False

        if self.use_weight_norm:
            with torch.no_grad():
                # Create one learnable tensor for norm ("g").
                # Initialize it with approximate existing norm:
                weight = self.get_parameter('weight', 0)

                self.weight_norm_g = torch.nn.Parameter(
                    torch.norm_except_dim(weight, 2, dim=0))

        # A an application of PyTorch 'parametrization' functionality
        class LowRankWeight(nn.Module):
            def __init__(self,
                multi_linear_module: LowRankMultiLinear, parameter_name: str, scene_idx: int):
                super().__init__()

                # don't "register as submodule", just store a reference
                self.__dict__['multi_linear_module'] = multi_linear_module

                self.parameter_name = parameter_name
                self.scene_idx = scene_idx

            def right_inverse(self, *args):
                """By returning an empty tuple, this tells `register_parametrization()` that
                we'll never need any of the original `Linear`'s parameters ('weight', 'bias').
                """
                return ()

            def forward(self):
                """No input args (module's parameters to be 'reparametrized') because we source
                parameters only from external tensors (namely, from `self.multi_linear_module`).
                This is ensured by `right_inverse()` having empty output.
                """
                param = self.multi_linear_module.get_parameter(self.parameter_name, self.scene_idx)

                if parameter_name == 'weight' and self.multi_linear_module.use_weight_norm:
                    param = torch._weight_norm(param, self.multi_linear_module.weight_norm_g, 0)

                return param

        for parameter_name, _ in list(self.linear_layers[0].named_parameters()):
            for scene_idx, layer in enumerate(self.linear_layers):
                nn.utils.parametrize.register_parametrization(
                    layer, parameter_name, LowRankWeight(self, parameter_name, scene_idx))

    def get_parameter(self, parameter_name, scene_idx):
        """
        Compute the linear combination and return the final parameter value (before weight norm).
        Used by `LowRankWeight` parametrization.
        """
        # (rank)
        combination_coeffs = self.combination_coeffs[scene_idx]
        # (out_dim x in_dim x rank[+1]) - in case `parameter_name` is 'weight'
        parameter_basis = self.basis_weights[parameter_name]

        if self.use_bias:
            return parameter_basis[..., :-1] @ combination_coeffs + parameter_basis[..., -1]
        else:
            # (out_dim x in_dim)
            return parameter_basis @ combination_coeffs

    def __getitem__(self, scene_idx):
        return self.linear_layers[scene_idx]

    def reset_parameters_(self):
        _, in_dim, rank = self.basis_weights['weight'].shape
        if self.use_bias:
            rank -= 1
        assert rank + self.use_bias == self.basis_weights['weight'].shape[-1]

        # Initialize weight
        for i in range(rank):
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            nn.init.kaiming_uniform_(self.basis_weights['weight'][..., i], a=np.sqrt(5))
        if self.use_bias:
            with torch.no_grad():
                self.basis_weights['weight'][..., -1].fill_(0)

        # Initialize bias
        bound = 1 / np.sqrt(in_dim)
        nn.init.uniform_(self.basis_weights['bias'], -bound, bound)
        if self.use_bias:
            with torch.no_grad():
                self.basis_weights['bias'][..., -1].fill_(0)

        # Initialize linear combination coefficients
        for x in self.combination_coeffs:
            nn.init.kaiming_uniform_(x[None], nonlinearity='linear')

    def switch_to_finetuning(self, algorithm='pick', scene_idx=0):
        """
        Like `SDFNetwork.switch_to_finetuning()`, but just for this layer.

        algorithm
            str
            One of:
            - pick (take the 0th scene's linear combination coefficients)
            - average (average coefficients over all scenes)
        """
        if algorithm == 'pick':
            if scene_idx == -1:
                new_combination_coeffs = self.combination_coeffs[0] * 0
            else:
                new_combination_coeffs = self.combination_coeffs[scene_idx]
        elif algorithm == 'average':
            new_combination_coeffs = torch.stack(list(self.combination_coeffs)).mean(0)
        else:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")

        self.combination_coeffs = nn.ParameterList([nn.Parameter(new_combination_coeffs)])
        self.linear_layers = self.linear_layers[:1]
        self.finetuning = True

    def parameters(self, which_layers='all', scene_idx=None):
        """which_layers: 'all'/'scenewise'/'shared'
        """
        if self.finetuning:
            assert scene_idx is None or scene_idx == 0 and len(self.combination_coeffs) == 1

        if which_layers == 'all':
            return super().parameters()
        elif which_layers == 'scenewise':
            return list(self.combination_coeffs) if scene_idx is None \
                else [self.combination_coeffs[scene_idx]]
        elif which_layers == 'shared':
            return list(self.basis_weights.values()) + [self.weight_norm_g]
        else:
            raise ValueError(f"Wrong 'which_layers': {which_layers}")

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 n_scenes,
                 scenewise_split_type='interleave',
                 scenewise_core_rank=None,
                 scenewise_bias=False,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        """
        n_scenes
            int
            Spawn `n_scenes` copies of every other layer, each trained independently. During the
            forward pass, take a scene index `i` and use the `i`th copy at each such layer.

        scenewise_split_type
            str
            One of:
            - 'interleave'
            - 'interleave3'
            - 'interleave_with_skips'
            - 'append_half'
            - 'prepend_half'
            - 'replace_last_half'
            - 'replace_first_half'
            - 'radiance_only'
            - 'sdf_only'
            - 'replace_first_half_sdf_only'
            - 'replace_first_half_radiance_only'
            - 'replace_first_2'
            - 'replace_first_3'
            - 'all'
        """
        super().__init__()

        self.scenewise_split_type = scenewise_split_type
        if scenewise_split_type in ('append_half', 'prepend_half'):
            num_scene_specific_layers = (n_layers + 1) // 2
            n_layers += num_scene_specific_layers
        elif scenewise_split_type in ('replace_last_half', 'replace_first_half', 'replace_first_half_sdf_only'):
            num_scene_specific_layers = (n_layers + 1) // 2
        elif scenewise_split_type in ('replace_first_3',):
            num_scene_specific_layers = 3
        elif scenewise_split_type in ('replace_first_2',):
            num_scene_specific_layers = 2

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims) - 1
        self.skip_in = skip_in
        self.scale = scale

        self.linear_layers = nn.ModuleList()
        total_scene_specific_layers = 0

        for l in range(self.num_layers):
            in_dim = dims[l]
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if scenewise_split_type == 'interleave':
                layer_is_scene_specific = l % 2 == 1
            elif scenewise_split_type == 'interleave3':
                layer_is_scene_specific = l % 3 == 1
            elif scenewise_split_type == 'interleave_with_skips':
                layer_is_scene_specific = l % 2 == 1 and in_dim == out_dim
            elif scenewise_split_type == 'interleave_with_skips_and_last':
                layer_is_scene_specific = (self.num_layers - 1 - l) % 2 == 0
            elif scenewise_split_type in ('append_half', 'replace_last_half'):
                layer_is_scene_specific = self.num_layers - num_scene_specific_layers <= l < self.num_layers - 1
            elif scenewise_split_type in ('prepend_half', 'replace_first_half', 'replace_first_half_sdf_only', 'replace_first_2', 'replace_first_3'):
                layer_is_scene_specific = l < num_scene_specific_layers
            elif scenewise_split_type in ('sdf_only', 'all'):
                layer_is_scene_specific = l < self.num_layers - 1
            elif scenewise_split_type in ('radiance_only', 'replace_first_half_radiance_only'):
                layer_is_scene_specific = False
            else:
                raise ValueError(
                    f"Wrong value for `scenewise_split_type`: '{scenewise_split_type}'")

            def geometric_init_(weight, bias_):
                if l == self.num_layers - 1:
                    if not inside_outside:
                        torch.nn.init.normal_(weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(bias_, -bias)
                    else:
                        torch.nn.init.normal_(weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(bias_, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(bias_, 0.0)
                    torch.nn.init.constant_(weight[:, 3:], 0.0)
                    torch.nn.init.normal_(weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(bias_, 0.0)
                    torch.nn.init.normal_(weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(bias_, 0.0)
                    torch.nn.init.normal_(weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            def create_linear_layer():
                layer = nn.Linear(in_dim, out_dim)
                if geometric_init:
                    geometric_init_(layer.weight, layer.bias)
                if weight_norm:
                    layer = nn.utils.weight_norm(layer)
                return layer

            if layer_is_scene_specific:
                logging.info(f"SDF: layer {l} is scene-specific")
                if scenewise_core_rank in (None, 0):
                    lin = nn.ModuleList([create_linear_layer() for _ in range(n_scenes)])
                else:
                    lin = LowRankMultiLinear(
                        n_scenes, in_dim, out_dim, scenewise_core_rank, weight_norm, scenewise_bias)

                    if geometric_init:
                        for i in range(scenewise_core_rank):
                            geometric_init_(
                                lin.basis_weights['weight'][..., i],
                                lin.basis_weights['bias'][..., i])
                        if scenewise_bias:
                            with torch.no_grad():
                                lin.basis_weights['weight'][..., -1].fill_(0)
                                lin.basis_weights['bias'][..., -1].fill_(0)

                total_scene_specific_layers += 1
            else:
                lin = create_linear_layer()

            self.linear_layers.append(lin)

        logging.info(
            f"SDF network got {total_scene_specific_layers} (out of " \
            f"{self.num_layers}) scene-specific layers")

        self.activation = nn.Softplus(beta=100)
        self.dims = dims

        # TODO restructure `parameters()` in all custom classes to get rid of this dirty hack
        if len(list(super().parameters())) != len(list(self.parameters('all'))):
            raise NotImplementedError(
                "There's an extra parameter that's not yet handled by `self.parameters()`. " \
                "Please address this.")

    def forward(self, inputs, scene_idx):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(self.num_layers):
            lin = self.linear_layers[l]
            layer_is_scene_specific = type(lin) is not torch.nn.Linear

            skip_connection = None
            if layer_is_scene_specific:
                lin = lin[scene_idx]
                if self.scenewise_split_type == 'interleave_with_skips':
                    skip_connection = x

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            if layer_is_scene_specific and self.scenewise_split_type == 'interleave_with_skips_and_last' \
                and self.dims[l] == self.dims[l + 1]:
                skip_connection = x

            x = lin(x)

            if skip_connection is not None:
                x += skip_connection

            if l < self.num_layers - 1:
                x = self.activation(x)

        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, scene_idx):
        return self(x, scene_idx)[:, :1]

    def gradient(self, x, scene_idx):
        with torch.enable_grad():
            x.requires_grad_(True)
            forward = self(x, scene_idx)
            sdf, feature_vector = forward[:, :1], forward[:, 1:]
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            return gradients, sdf, feature_vector

    def switch_to_finetuning(self, algorithm='pick', scene_idx=0):
        """
        Switch the network trained on multiple scenes to the 'finetuning mode',
        to finetune it to some new (one) scene.

        algorithm
            str
            One of:
            - pick (take the `scene_idx`-th scene's 'subnetwork')
            - average (average weight tensors over all scenes)
        """
        for i in range(self.num_layers):
            layer_type = type(self.linear_layers[i])

            if layer_type is nn.ModuleList:
                if algorithm == 'pick':
                    new_layer = self.linear_layers[i][scene_idx]
                elif algorithm == 'average':
                    new_layer = self.linear_layers[i][0]
                    for param_name, _ in new_layer.named_parameters():
                        averaged_param = torch.stack(
                            [m.get_parameter(param_name) for m in self.linear_layers[i]]).mean(0)
                        new_layer.get_parameter(param_name).copy_(averaged_param)
                else:
                    raise ValueError(f"Unknown algorithm: '{algorithm}'")

                self.linear_layers[i] = nn.ModuleList([new_layer])

            elif layer_type is LowRankMultiLinear:
                self.linear_layers[i].switch_to_finetuning(algorithm, scene_idx)

    def parameters(self, which_layers='all', scene_idx=None):
        """which_layers: 'all'/'scenewise'/'shared'
        """
        if which_layers == 'all':
            assert scene_idx is None, "which_layers='all' isn't supported with scene_idx != None"
            return list(super().parameters())
        elif which_layers == 'scenewise':
            retval = []
            for module in self.linear_layers:
                if type(module) is nn.Linear:
                    pass # Only shared layers can be `Linear`, so don't add
                elif type(module) is nn.ModuleList:
                    module_to_add = module if scene_idx is None else module[scene_idx]
                    retval += list((module_to_add).parameters())
                elif type(module) is LowRankMultiLinear:
                    # Let `LowRankMultiLinear` decide
                    retval += list(module.parameters(which_layers, scene_idx))
                else:
                    raise RuntimeError(f"Unexpected module type: {module}")
            return retval
        elif which_layers == 'shared':
            retval = []
            for module in self.linear_layers:
                if type(module) is nn.Linear:
                    retval += list(module.parameters())
                elif type(module) is nn.ModuleList:
                    pass # Only scenewise layers can be `ModuleList`, so don't add
                elif type(module) is LowRankMultiLinear:
                    # Let `LowRankMultiLinear` decide
                    retval += list(module.parameters(which_layers, scene_idx))
                else:
                    raise RuntimeError(f"Unexpected module type: {module}")
            return retval
        else:
            raise ValueError(f"Wrong 'which_layers': {which_layers}")


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
# TODO: remove repetitive code from SDFNetwork
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 n_scenes,
                 scenewise_split_type='interleave',
                 scenewise_core_rank=None,
                 scenewise_bias=False,
                 weight_norm=True,
                 multires=0,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out

        self.scenewise_split_type = scenewise_split_type
        if scenewise_split_type in ('append_half', 'prepend_half'):
            num_scene_specific_layers = (n_layers + 1) // 2
            n_layers += num_scene_specific_layers
        elif scenewise_split_type in ('replace_last_half', 'replace_first_half', 'replace_first_half_radiance_only', 'replace_first_3', 'replace_first_2'):
            num_scene_specific_layers = (n_layers + 1) // 2

        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_ch = get_embedder(multires, input_dims=3)
            dims[0] += input_ch - 3

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view, input_dims=3)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims) - 1

        maybe_weight_norm = nn.utils.weight_norm if weight_norm else lambda x: x

        self.linear_layers = nn.ModuleList()
        total_scene_specific_layers = 0

        for l in range(0, self.num_layers):
            in_dim, out_dim = dims[l], dims[l + 1]

            if scenewise_split_type == 'interleave':
                layer_is_scene_specific = l % 2 == 1
            elif scenewise_split_type == 'interleave3':
                layer_is_scene_specific = l % 3 == 1
            elif scenewise_split_type == 'interleave_with_skips':
                layer_is_scene_specific = l % 2 == 1 and in_dim == out_dim
            elif scenewise_split_type == 'interleave_with_skips_and_last':
                layer_is_scene_specific = (self.num_layers - 1 - l) % 2 == 0
            elif scenewise_split_type in ('append_half', 'replace_last_half'):
                layer_is_scene_specific = l >= self.num_layers - num_scene_specific_layers
            elif scenewise_split_type in ('prepend_half', 'replace_first_half', 'replace_first_half_radiance_only', 'replace_first_2', 'replace_first_3'):
                layer_is_scene_specific = l < num_scene_specific_layers
            elif scenewise_split_type in ('sdf_only', 'replace_first_half_sdf_only'):
                layer_is_scene_specific = False
            elif scenewise_split_type in ('radiance_only', 'all'):
                layer_is_scene_specific = True
            else:
                raise ValueError(
                    f"Wrong value for `scenewise_split_type`: '{scenewise_split_type}'")

            if layer_is_scene_specific:
                logging.info(f"Radiance: layer {l} is scene-specific")
                if scenewise_core_rank in (None, 0):
                    lin = nn.ModuleList(
                        [maybe_weight_norm(nn.Linear(in_dim, out_dim)) for _ in range(n_scenes)])
                else:
                    lin = LowRankMultiLinear(
                        n_scenes, in_dim, out_dim, scenewise_core_rank, weight_norm, scenewise_bias)
                total_scene_specific_layers += 1
            else:
                lin = maybe_weight_norm(nn.Linear(in_dim, out_dim))

            self.linear_layers.append(lin)

        logging.info(
            f"Rendering network got {total_scene_specific_layers} (out of " \
            f"{self.num_layers}) scene-specific layers")

        self.relu = nn.ReLU()
        self.dims = dims

        # TODO restructure `parameters()` in all custom classes to get rid of this dirty hack
        if len(list(super().parameters())) != len(list(self.parameters('all'))):
            raise NotImplementedError(
                "There's an extra parameter that's not yet handled by `self.parameters()`. " \
                "Please address this.")

    def forward(self, points, normals, view_dirs, feature_vectors, scene_idx):
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'points_grads_only':
            rendering_input = torch.cat([points, normals], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers):
            lin = self.linear_layers[l]
            layer_is_scene_specific = type(lin) is not torch.nn.Linear

            skip_connection = None
            if layer_is_scene_specific:
                lin = lin[scene_idx]
                if self.scenewise_split_type == 'interleave_with_skips':
                    skip_connection = x
                elif self.scenewise_split_type == 'interleave_with_skips_and_last' \
                    and self.dims[l] == self.dims[l + 1]:
                    skip_connection = x

            x = lin(x)

            if skip_connection is not None:
                x += skip_connection

            if l < self.num_layers - 1:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x

    def gradient(self, points, normals, view_dirs, feature_vectors, scene_idx):
        with torch.enable_grad():
            points.requires_grad_(True)
            radiance = self(points, normals, view_dirs, feature_vectors, scene_idx)

            gradients = []
            for i in range(3): # R,G,B
                d_output = torch.zeros_like(radiance, requires_grad=False, device=radiance.device)
                d_output[..., i] = 1

                gradients.append(
                    torch.autograd.grad(
                        outputs=radiance,
                        inputs=points,
                        grad_outputs=d_output,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0])

            # dim 0: gradient of r/g/b
            # dim 1: point number
            # dim 2: gradient over x/y/z
            return torch.stack(gradients) # 3, K, 3

    def switch_to_finetuning(self, algorithm='pick', scene_idx=0):
        """
        Switch the network trained on multiple scenes to the 'finetuning mode',
        to finetune it to some new (one) scene.

        algorithm
            str
            One of:
            - pick (take the `scene_idx`-th scene's 'subnetwork')
            - average (average weight tensors over all scenes)
        """
        for i in range(self.num_layers):
            layer_type = type(self.linear_layers[i])

            if layer_type is nn.ModuleList:
                if algorithm == 'pick':
                    new_layer = self.linear_layers[i][scene_idx]
                elif algorithm == 'average':
                    new_layer = self.linear_layers[i][0]
                    for param_name, _ in new_layer.named_parameters():
                        averaged_param = torch.stack(
                            [m.get_parameter(param_name) for m in self.linear_layers[i]]).mean(0)
                        new_layer.get_parameter(param_name).copy_(averaged_param)
                else:
                    raise ValueError(f"Unknown algorithm: '{algorithm}'")

                self.linear_layers[i] = nn.ModuleList([new_layer])

            elif layer_type is LowRankMultiLinear:
                self.linear_layers[i].switch_to_finetuning(algorithm, scene_idx)

                # If we wanted to compute weights and revert to a regular `Linear` layer:
                # if type(self.linear_layers[i]) is LowRankMultiLinear:
                #     for p_name in 'weight', 'bias':
                #         nn.utils.parametrize.remove_parametrizations(layer_to_pick, p_name)

    def parameters(self, which_layers='all', scene_idx=None):
        """which_layers: 'all'/'scenewise'/'shared'
        """
        if which_layers == 'all':
            assert scene_idx is None, "which_layers='all' isn't supported with scene_idx != None"
            return list(super().parameters())
        elif which_layers == 'scenewise':
            retval = []
            for module in self.linear_layers:
                if type(module) is nn.Linear:
                    pass # Only shared layers can be `Linear`, so don't add
                elif type(module) is nn.ModuleList:
                    module_to_add = module if scene_idx is None else module[scene_idx]
                    retval += list((module_to_add).parameters())
                elif type(module) is LowRankMultiLinear:
                    # Let `LowRankMultiLinear` decide
                    retval += list(module.parameters(which_layers, scene_idx))
                else:
                    raise RuntimeError(f"Unexpected module type: {module}")
            return retval
        elif which_layers == 'shared':
            retval = []
            for module in self.linear_layers:
                if type(module) is nn.Linear:
                    retval += list(module.parameters())
                elif type(module) is nn.ModuleList:
                    pass # Only scenewise layers can be `ModuleList`, so don't add
                elif type(module) is LowRankMultiLinear:
                    # Let `LowRankMultiLinear` decide
                    retval += list(module.parameters(which_layers, scene_idx))
                else:
                    raise RuntimeError(f"Unexpected module type: {module}")
            return retval
        else:
            raise ValueError(f"Wrong 'which_layers': {which_layers}")


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

class MultiSceneNeRF(nn.ModuleList):
    def __init__(self, n_scenes, *args, **kwargs):
        super().__init__([NeRF(*args, **kwargs) for _ in range(n_scenes)])

    def switch_to_finetuning(self, algorithm='pick', scene_idx=0):
        """
        Switch the network trained on multiple scenes to the 'finetuning mode',
        to finetune it to some new (one) scene.

        algorithm
            str
            One of:
            - pick (take the `scene_idx`-th scene's 'subnetwork')
            - average (same as 'pick' because we don't use background in finetuning anyway)

            Arguments have no effect: always works like it's algorithm=='pick' and scene_idx==0.
        """
        if algorithm in ('pick', 'average'):
            super().__init__([self[0]])
        else:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")

    def parameters(self, which_layers='all', scene_idx=None):
        """which_layers: 'all'/'scenewise'/'shared'
        """
        if which_layers == 'all':
            assert scene_idx is None, "which_layers='all' isn't supported with scene_idx != None"
            return list(super().parameters())
        elif which_layers == 'shared':
            return []
        elif which_layers == 'scenewise':
            if scene_idx is None:
                return super().parameters()
            else:
                return list(self[scene_idx].parameters())
        else:
            raise ValueError(f"Wrong 'which_layers': {which_layers}")


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.empty([])))
        self.reset_parameters_(init_val)

    def forward(self, size):
        return torch.exp(self.variance * 10.0).view(1, 1).expand(size, 1)

    def parameters(self, which_layers='all', scene_idx=None):
        """which_layers: 'all'/'scenewise'/'shared'
        """
        if which_layers in ('all', 'shared'):
            assert scene_idx is None, "which_layers='all' isn't supported with scene_idx != None"
            return list(super().parameters())
        elif which_layers == 'scenewise':
            return []
        else:
            raise ValueError(f"Wrong 'which_layers': {which_layers}")

    def reset_parameters_(self, init_val):
        with torch.no_grad():
            self.variance.fill_(init_val)

    def switch_to_finetuning(self, algorithm=None, scene_idx=None):
        """
        Switch the network trained on multiple scenes to the 'finetuning mode',
        to finetune it to some new (one) scene.

        algorithm
            any type
            No effect. For compatibility only.
        """
        pass # self.reset_parameters_(0.365)

class TrainableCameraParams(nn.Module):
    """
    Trainable correction for several sets of camera parameters for several scenes.
    Intrinsics are shared between cameras in a scene; optionally, extrinsics can be shared too.
    1 trainable parameter for intrinsics (focal distance multiplicative delta)
    and 6 for extrinsics (se(3) correction = 3 translation amounts + 3 rotation angles).
    """
    def __init__(self, num_cameras_per_scene, share_extrinsics=False):
        """
        num_cameras_per_scene:
            list of int
            length = number of scenes in the dataset
        share_extrinsics:
            bool
            If True, learn only one extrinsic matrix per scene (and discard the
            contents of `num_cameras_per_scene`).
        """
        super().__init__()

        if share_extrinsics:
            num_cameras_per_scene = [1] * len(num_cameras_per_scene)
        self.share_extrinsics = share_extrinsics

        self.log_focal_dist_delta = nn.ParameterList(
            [nn.Parameter(torch.zeros([])) for _ in range(len(num_cameras_per_scene))])
        self.pose_se3_delta = nn.ParameterList([
            nn.Parameter(torch.zeros(num_cameras, 6)) for num_cameras in num_cameras_per_scene])

    def apply_params_correction(self, scene_idx, camera_idx, intrinsics_init, pose_init):
        """
        scene_idx
            int
        camera_idx
            int
            No effect if was constructed with `share_extrinsics=True`.
        intrinsics_init
            torch.FloatTensor, shape == (4, 4)
        pose_init
            torch.FloatTensor, shape == (4, 4)

        return:
        intrinsics_new
        pose_new
            torch.FloatTensor, shape == (4, 4)
            The above set of camera parameters but with trainable corrections applied.
        """
        if self.share_extrinsics:
            camera_idx = 0

        intrinsics_new = intrinsics_init.clone()
        focal_dist_delta = self.log_focal_dist_delta[scene_idx].exp()
        intrinsics_new[0, 0] *= focal_dist_delta
        intrinsics_new[1, 1] *= focal_dist_delta

        pose_SE3_delta = utils.camera.se3_to_SE3(self.pose_se3_delta[scene_idx][camera_idx])
        pose_new = utils.camera.compose([pose_SE3_delta, pose_init[:3, :4]])

        return intrinsics_new, pose_new

    def parameters(self, which_layers='all', scene_idx=None):
        """which_layers: no effect
        """
        if which_layers in ('all', 'scenewise'):
            if scene_idx is None:
                return list(super().parameters())
            else:
                return [self.pose_se3_delta[scene_idx], self.log_focal_dist_delta[scene_idx]]
        elif which_layers == 'shared':
            return []
        else:
            raise ValueError(f"Wrong 'which_layers': {which_layers}")
