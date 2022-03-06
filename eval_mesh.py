import argparse
import hashlib
import os
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


class MeshEvaluator:
    def __init__(self, path_gt: str, num_samples: int = 100000, device: torch.device = torch.device('cuda'),
                 h3ds_scene_id: Optional[str] = None):
        self.h3ds_scene_id = h3ds_scene_id
        if self.h3ds_scene_id is not None:
            self.h3ds_dataset = H3DS(path='/gpfs/data/gpfs0/egor.burkov/Datasets/H3DS')
        self.path_gt: str = path_gt
        self.num_samples: int = num_samples
        self.device: torch.device = device
        self.mesh_gt: Meshes = self.read_mesh(path_gt, gt=True)
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

    def read_mesh(self, path: str, read_texture: bool = True, gt=False) -> Meshes:
        mesh = Mesh().load(path, read_texture)

        if not gt:
            region_gt = None
            if self.h3ds_scene_id is not None:
                region_id = None  # face | face_sphere | nose
                region_gt = self.h3ds_dataset.load_region(self.h3ds_scene_id, region_id or 'face')
            _, t_icp = perform_icp(self._mesh_gt, mesh, region_gt)
            mesh = transform_mesh(mesh, np.linalg.inv(t_icp))

        # mesh.compute_normals()

        if gt:
            self._mesh_gt = mesh

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
            mesh_pred = self.read_mesh(mesh_pred, read_texture)

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
            [9.9659306e-01, 7.6131098e-02, 3.1722244e-02, -1.6078455e+01],
            [-7.9632021e-02, 9.8833752e-01, 1.2979856e-01, -6.3538975e+01],
            [-2.1470578e-02, -1.3188246e-01, 9.9103284e-01, -6.1367126e+02],
            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
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
            mesh_pred = self.read_mesh(mesh_pred)

        dist2_pred, dist2_gt = unidirectional_chamfer_distance(mesh_pred.verts_list()[0][None],
                                                               self.samples_gt)  # (1, n)
        dist_pred = torch.sqrt(dist2_pred)
        dist_gt = torch.sqrt(dist2_gt)
        # print(dist_pred.shape, dist2_pred.mean(), dist_pred.mean())
        # print(dist_gt.shape, dist_gt.mean())

        vmax = torch.quantile(dist_pred, 0.5).item()
        print("VMAX", vmax)
        # vmax = 3.0
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        rgb = matplotlib.cm.get_cmap("RdYlGn_r")(norm(dist_pred.cpu().numpy()))[..., :3]
        rgb = torch.from_numpy(rgb).float().to(self.device)

        mesh_error_colored = Meshes(
            verts=mesh_pred.verts_list(),
            faces=mesh_pred.faces_list(),
            textures=Textures(verts_rgb=rgb),
        )

        return self.visualize_mesh(mesh_error_colored, SimpleShader)


def md5(path):
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default=None)
    parser.add_argument('--gt_mesh', type=str, default=None)
    parser.add_argument('--scene_id', type=str, default=None)
    args = parser.parse_args()

    evaluator = MeshEvaluator(args.gt_mesh, h3ds_scene_id=args.scene_id)
    # metrics = evaluator.compute_metrics(args.mesh)
    # print(metrics)
    if args.scene_id is not None:
        h3ds_dataset = H3DS(path='/gpfs/data/gpfs0/egor.burkov/Datasets/H3DS')
        mesh_pred = trimesh.load_mesh(args.mesh)
        chamfer_gt, chamfer_pred, _, _ = h3ds_dataset.evaluate_scene(args.scene_id, mesh_pred)
        print("h3ds chamfer gt: ", chamfer_gt.shape, chamfer_gt.mean())
        print("h3ds chamfer pred: ", chamfer_pred.shape, chamfer_pred.mean())
    texture = evaluator.visualize_mesh(args.mesh, SimpleShader)
    save_image(texture.permute(2, 0, 1), 'texture.png')
    geometry = evaluator.visualize_mesh(args.mesh, HardGouraudShader, read_texture=False)
    save_image(geometry.permute(2, 0, 1), 'geometry.png')
    errors = evaluator.visualize_errors(args.mesh)
    save_image(errors.permute(2, 0, 1), 'errors.png')
