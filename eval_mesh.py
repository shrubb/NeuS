import argparse
import hashlib
import os
from time import perf_counter
from typing import Dict, Union, Tuple, Optional

import matplotlib.cm
import matplotlib.colors
import numpy as np
import pytorch3d
import torch
import torch.nn as nn
import torch.utils.data
import trimesh
from h3ds.dataset import H3DS
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardGouraudShader,
    Textures
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.structures import Meshes, Pointclouds
from torchvision.utils import save_image

from utils.comm import synchronize, is_main_process, get_world_size
from utils.compute_iou import compute_iou
from utils.inside_mesh import inside_mesh
from utils.mesh import Mesh
from utils.numeric import perform_icp, transform_mesh
from utils.point_to_face import point_mesh_face_distances
from utils.uni_cd import unidirectional_chamfer_distance


class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None, **kwargs):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image


def get_reg_transform(scene_id):
    camera_path = f'/gpfs/data/gpfs0/egor.burkov/Datasets/H3DS_preprocessed/{scene_id}/cameras_sphere.npz'
    camera_dict = np.load(camera_path)
    return camera_dict['reg_mat_0']


def error_to_color(errors, clipping_error=None):
    if clipping_error is not None:
        errors_norm = np.clip(errors / float(clipping_error), 0., 1.)
    else:
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min())

    hsv = np.ones((errors.shape[-1], 3))
    hsv[:, 0] = (1. - errors_norm) / 3.

    return matplotlib.colors.hsv_to_rgb(hsv)


class MeshEvaluator:
    def __init__(self, path_gt: str, num_samples: int = 100000, device: torch.device = torch.device('cuda'),
                 h3ds_scene_id: Optional[str] = None, h3ds_region_id: Optional[str] = None):
        self.h3ds_scene_id = h3ds_scene_id
        self.h3ds_region_id = h3ds_region_id
        if self.h3ds_scene_id is not None:
            self.h3ds_dataset = H3DS(path='/gpfs/data/gpfs0/egor.burkov/Datasets/H3DS')
        self.path_gt: str = path_gt
        self.num_samples: int = num_samples
        self.device: torch.device = device
        self.mesh_gt: Optional[Meshes] = None
        self._uncutted_mesh_gt = None
        if path_gt is not None:
            self.mesh_gt = self.read_mesh(path_gt, gt=True)
            self.bbox_gt: torch.Tensor = self.mesh_gt.get_bounding_boxes()  # (1, 3, 2)
            self.samples_gt, self.normals_gt = self.get_gt_samples()

    def get_gt_samples(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.h3ds_scene_id is not None:
            return self.mesh_gt.verts_list()[0][None], None  # self.mesh_gt.verts_normals_list()[0][None]

        cache_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path: str = os.path.join(cache_dir, f"{md5(self.path_gt)}_{self.num_samples}.pt")

        if get_world_size() > 1 and not os.path.exists(cache_path) and not is_main_process():
            synchronize()

        if os.path.exists(cache_path):
            samples_gt, normals_gt = torch.load(cache_path, map_location=self.device).split(1)
        else:
            assert is_main_process()
            samples_gt, normals_gt = sample_points_from_meshes(self.mesh_gt, num_samples=self.num_samples,
                                                               return_normals=True)  # (1, n, 3), (1, n, 3)
            torch.save(torch.cat([samples_gt, normals_gt], dim=0).cpu(), cache_path)
            if get_world_size() > 1:
                synchronize()

        return samples_gt, normals_gt

    @property
    def max_side(self) -> float:
        if getattr(self, "_max_side", None) is None:
            sides = self.bbox_gt[:, :, 1] - self.bbox_gt[:, :, 0]  # (1, 3)
            self._max_side = sides.max().item()
        return self._max_side

    def read_mesh(self, path: str, read_texture: bool = True, gt=False, cut_pred=False) -> Meshes:
        mesh = Mesh().load(path, read_texture)

        region_gt = None
        if self.h3ds_scene_id is not None:
            region_gt = self.h3ds_dataset.load_region(self.h3ds_scene_id, self.h3ds_region_id or 'face')

        if gt:
            self._uncutted_mesh_gt = mesh
            if self.h3ds_region_id:
                mesh = mesh.cut(region_gt)
            self._mesh_gt = mesh
        else:
            if self.h3ds_scene_id is not None:
                # костыль, так как не сохранено в экспериментах изначально по-нормальному
                mesh = transform_mesh(mesh, get_reg_transform(self.h3ds_scene_id))

            if self._uncutted_mesh_gt is not None:
                _, t_icp = perform_icp(self._uncutted_mesh_gt, mesh, region_gt)
                mesh = transform_mesh(mesh, np.linalg.inv(t_icp))

            if self.h3ds_scene_id is not None and self.h3ds_region_id == 'face_sphere' and cut_pred:
                # when metric is calculated the mesh is uncutted
                # for visualization otherwise
                landmarks_true = self.h3ds_dataset.load_landmarks(self.h3ds_scene_id)
                mask_sphere = np.where(
                    np.linalg.norm(mesh.vertices -
                                   self._uncutted_mesh_gt.vertices[landmarks_true['nose_tip']],
                                   axis=-1) < 95)
                mesh = mesh.cut(mask_sphere)

        # mesh.compute_normals()

        return Meshes(
            verts=[torch.from_numpy(mesh.vertices).float()],
            # verts_normals=[torch.from_numpy(mesh.vertex_normals).float()],
            faces=[torch.from_numpy(mesh.faces).long()],
            textures=Textures(verts_rgb=torch.from_numpy(mesh.vertices_color).float()[None]),
        ).to(self.device)

    def compute_chamfer(self, samples_pred, normals_pred) -> Dict:
        chamfer_distance, normals_loss = \
            pytorch3d.loss.chamfer_distance(x=samples_pred, y=self.samples_gt,
                                            x_normals=normals_pred, y_normals=self.normals_gt)
        normal_consistency = 1 - normals_loss

        # Like Fan et al. [17] we use 1/10 times the maximal edge length
        # of the current object’s bounding box as unit 1
        unit = self.max_side / 10.0
        unit2 = unit ** 2
        chamfer_distance /= unit2

        return {
            'chamfer-l2': chamfer_distance.item(),
            'normals': normal_consistency.item()
        }

    def compute_iou(self, mesh_pred: Meshes) -> Dict:
        # valid only for watertight meshes
        # todo currently leads to oom when the num_samples is big

        bbox_gt = self.bbox_gt  # (1, 3, 2)
        bbox_pred = mesh_pred.get_bounding_boxes()  # (1, 3, 2)

        bbox_cat = torch.cat([bbox_pred, bbox_gt], dim=0)
        bound_min = bbox_cat.min(dim=0).values[None, :, 0]  # (1, 3)
        bound_max = bbox_cat.max(dim=0).values[None, :, 1]  # (1, 3)

        samples_range = (bound_max - bound_min)
        vol_samples = bound_min + torch.rand(self.num_samples, 3, device=bound_min.device) * samples_range  # (n, 3)
        occ_gt = inside_mesh(vol_samples, self.mesh_gt.verts_list()[0], self.mesh_gt.faces_list()[0])
        occ_pred = inside_mesh(vol_samples, mesh_pred.verts_list()[0], mesh_pred.faces_list()[0])
        iou = compute_iou(occ_pred, occ_gt)

        return {'iou': iou}

    def compute_fscore(self, samples_pred: torch.Tensor, mesh_pred: Meshes, qs=(0.01, 0.02, 0.04)) -> Dict:
        p_dists2 = point_mesh_face_distances(self.mesh_gt, Pointclouds(samples_pred))
        r_dists2 = point_mesh_face_distances(mesh_pred, Pointclouds(self.samples_gt))
        fs = {}
        for q in qs:
            d = q * self.max_side
            d2 = d ** 2
            precision = (p_dists2 <= d2).sum() / p_dists2.shape[0]
            recall = (r_dists2 <= d2).sum() / r_dists2.shape[0]
            fscore = 2.0 * precision * recall / (precision + recall)
            fs[f'f-score-{q}'] = fscore.item()
        return fs

    # def sample_points_from_pred_meshes(self, mesh, num_samples):
    #     if num_samples

    @torch.no_grad()
    def compute_metrics(self, mesh_pred: Union[str, Meshes]) -> Dict:
        # the volumetric IoU and a normal consistency score
        # are defined in https://arxiv.org/pdf/1812.03828.pdf
        # f-score is defined in https://arxiv.org/pdf/1905.03678.pdf
        # code reference: https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/im2mesh/eval.py

        if isinstance(mesh_pred, str):
            mesh_pred = self.read_mesh(mesh_pred)

        samples_pred, normals_pred = sample_points_from_meshes(mesh_pred, num_samples=self.num_samples,
                                                               return_normals=True)
        metrics = {
            # **self.compute_iou(mesh_pred),
            **self.compute_fscore(samples_pred, mesh_pred),
            **self.compute_chamfer(samples_pred, normals_pred)
        }

        return metrics

    @torch.no_grad()
    def visualize_mesh(self, mesh_pred: Union[str, Meshes], shader, read_texture=True) -> torch.Tensor:
        if isinstance(mesh_pred, str):
            mesh_pred = self.read_mesh(mesh_pred, read_texture, cut_pred=True)

        # intrinsics
        K = torch.tensor([
            [670, 0, 256, 0],
            [0, 670, 256, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], device=self.device, dtype=torch.float32)[None]
        image_size = (512, 512)

        # extrinsics
        pose = torch.tensor([
            # [9.9659306e-01, 7.6131098e-02, 3.1722244e-02, -1.6078455e+01],
            # [-7.9632021e-02, 9.8833752e-01, 1.2979856e-01, -6.3538975e+01],
            # [-2.1470578e-02, -1.3188246e-01, 9.9103284e-01, -6.1367126e+02],
            # [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]

            [8.7044e-01, 1.6037e-01, -4.6542e-01, 1.3151e+00],
            [-1.1271e-01, 9.8526e-01, 1.2871e-01, -5.2989e-01],
            [4.7920e-01, -5.9573e-02, 8.7568e-01, -2.6827e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]

            # [0.6131477, 0.23778298, -0.7533321, 2.9539409],
            # [-0.16528392, 0.97113144, 0.17200263, -0.3341688],
            # [0.77248377, 0.01905067, 0.6347487, 0.3226022],
            # [0., 0., 0., 1., ]
        ], device=self.device, dtype=torch.float32)
        RtT = torch.inverse(pose)
        RtT[:2, :] *= -1
        R = RtT[:3, :3].T[None]
        T = RtT[:3, 3][None]

        cameras = PerspectiveCameras(R=R, T=T, K=K,
                                     device=self.device, in_ndc=False,
                                     image_size=torch.tensor(image_size).view(1, 2))
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-6,
            faces_per_pixel=1,
            clip_barycentric_coords=True,
            cull_backfaces=True,
            perspective_correct=True,
        )
        lights = PointLights(device=self.device, location=cameras.get_camera_center())
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            # shader=HardGouraudShader(device=self.device, cameras=cameras, lights=lights),
            # shader=SimpleShader()
            shader=shader(device=self.device, cameras=cameras, lights=lights),
        )
        image = renderer(mesh_pred)[0, ..., :3].cpu()
        return image

    @torch.no_grad()
    def visualize_errors(self, mesh_pred: Union[str, Meshes]) -> torch.Tensor:
        if isinstance(mesh_pred, str):
            mesh_pred = self.read_mesh(mesh_pred, cut_pred=True)

        dist2_pred, dist2_gt = unidirectional_chamfer_distance(mesh_pred.verts_list()[0][None],
                                                               self.samples_gt)  # (1, n)
        dist_pred = torch.sqrt(dist2_pred)

        # dist_gt = torch.sqrt(dist2_gt)
        # print("dist_gt.mean(): ", dist_gt.mean(), dist_gt.shape)

        # vmax = 20.0
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        # rgb = matplotlib.cm.get_cmap("RdYlGn_r")(norm(dist_pred.cpu().numpy()))[..., :3]
        # rgb = torch.from_numpy(rgb).float().to(self.device)

        rgb = error_to_color(dist_pred.cpu().numpy(), clipping_error=5)
        rgb = torch.from_numpy(rgb).float().to(self.device)[None]  # (1, n, 3)

        mesh_error_colored = Meshes(
            verts=mesh_pred.verts_list(),
            faces=mesh_pred.faces_list(),
            textures=Textures(verts_rgb=rgb),
        )

        # return self.visualize_mesh(mesh_error_colored, SimpleShader)
        return self.visualize_mesh(mesh_error_colored, HardGouraudShader)


def md5(path):
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()


if __name__ == '__main__':
    start_time = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default=None)
    parser.add_argument('--gt_mesh', type=str, default=None)
    parser.add_argument('--scene_id', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    if args.out_dir is not None:
        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)

    if args.scene_id is not None:
        h3ds_dataset = H3DS(path='/gpfs/data/gpfs0/egor.burkov/Datasets/H3DS')

        mesh_pred = trimesh.load_mesh(args.mesh)
        mesh_pred.apply_transform(get_reg_transform(args.scene_id))
        head_chamfer_gt, _, _, _ = h3ds_dataset.evaluate_scene(args.scene_id, mesh_pred)
        print(f' > Chamfer distance full head (mm): {head_chamfer_gt.mean()}, {head_chamfer_gt.shape}')
        face_chamfer_gt, _, _, _ = h3ds_dataset.evaluate_scene(args.scene_id, mesh_pred, region_id='face')
        print(f' > Chamfer distance face (mm): {face_chamfer_gt.mean()}, {face_chamfer_gt.shape}')
        face_sphere_chamfer_gt, _, _, _ = h3ds_dataset.evaluate_scene(args.scene_id, mesh_pred, region_id='face_sphere')
        print(f' > Chamfer distance face_sphere (mm): {face_sphere_chamfer_gt.mean()}, {face_sphere_chamfer_gt.shape}')

        # metrics_path = os.path.join("outputs2", "metrics.csv")
        # write_head = not os.path.exists(metrics_path)
        # f = open(metrics_path, "a")
        # if write_head:
        #     f.write("scene_id,head,face,face_sphere,mesh\n")
        # f.write(f"{args.scene_id},{head_chamfer_gt.mean()},{face_chamfer_gt.mean()},{face_sphere_chamfer_gt.mean()},{args.mesh}\n")
        # f.close()

    if args.out_dir is not None:
        region_ids = [None, 'face_sphere'] if args.scene_id is not None else [None]
        for region_id in region_ids:
            region_suf = 'full' if region_id is None else region_id
            evaluator = MeshEvaluator(args.gt_mesh, h3ds_scene_id=args.scene_id, h3ds_region_id=region_id)
            texture = evaluator.visualize_mesh(args.mesh, SimpleShader)
            save_image(texture.permute(2, 0, 1), os.path.join(out_dir, f'texture_{region_suf}.png'))
            geometry = evaluator.visualize_mesh(args.mesh, HardGouraudShader, read_texture=False)
            save_image(geometry.permute(2, 0, 1), os.path.join(out_dir, f'geometry_{region_suf}.png'))
            if evaluator.mesh_gt is not None:
                errors = evaluator.visualize_errors(args.mesh)
                save_image(errors.permute(2, 0, 1), os.path.join(out_dir, f'errors_{region_suf}.png'))
                gt_geometry = evaluator.visualize_mesh(evaluator.mesh_gt, HardGouraudShader, read_texture=False)
                save_image(gt_geometry.permute(2, 0, 1), os.path.join(out_dir, f'gt_geometry_{region_suf}.png'))

    end_time = perf_counter()
    print(f"Finished {args.mesh} in {end_time - start_time} seconds.")
