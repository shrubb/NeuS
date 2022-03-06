import json

import numpy as np
import trimesh

from utils.mesh import Mesh


def perform_icp(mesh_source: Mesh,
                mesh_target: Mesh,
                mask_source: np.ndarray = None,
                mask_target: np.ndarray = None,
                **icp_args) -> tuple:
    points_source = mesh_source.vertices if mask_source is None else mesh_source.vertices[
        mask_source]
    points_target = mesh_target.vertices if mask_target is None else mesh_source.vertices[
        mask_target]
    transform, _, _ = trimesh.registration.icp(points_source, points_target,
                                               **icp_args)

    return transform_mesh(mesh_source, transform), transform


def transform_mesh(mesh: Mesh, transform: np.ndarray):
    mesh_t = mesh.copy()
    mesh_t.vertices = AffineTransform(matrix=transform).transform(
        mesh_t.vertices)
    return mesh_t


class AffineTransform:

    def __init__(self, dim: int = 3, matrix: np.ndarray = np.eye(4)):
        self.dim = dim
        self.matrix = np.copy(matrix)

    def load(self, filename: str):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)

        self.dim = int(data['dimension'])
        self.matrix = np.eye(self.dim + 1)
        for el in data['transform']:
            self.matrix[int(el['row']),
                        int(el['column'])] = float(el['element'])
        return self

    def save(self, filename: str):
        data = {"dimension": str(self.dim), 'transform': []}
        for r in range(self.dim + 1):
            for c in range(self.dim + 1):
                data['transform'].append({
                    'row': str(r),
                    'column': str(c),
                    'element': str(self.matrix[r, c])
                })

        with open(filename, 'w') as f:
            json.dump(data, f)

    def transform(self, points: np.ndarray):
        # Assuming ( n_points, dim)
        n_points, dim = points.shape
        assert dim == self.dim

        # Project points
        points_h = np.concatenate((points.T, np.ones((1, n_points))), axis=0)
        return np.dot(self.matrix, points_h)[:self.dim, :].T

    def inverse(self):
        t = AffineTransform(dim=self.dim)
        t.matrix = np.linalg.inv(self.matrix)
        return t
