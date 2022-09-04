#!/usr/bin/python3

import os
import sys
import pathlib
import pickle
import glob
import subprocess

import numpy as np
import shutil
import cv2
import imageio
import tqdm

import face_alignment
from face_alignment.detection.sfd import FaceDetector

sys.path.append('..')
sys.path.append('../gonzalo')
from gonzalo.preprocess import ImageCropper

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def run_subprocess(command, working_dir=None):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=working_dir)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)

def get_extrinsics_given_landmarks(landmarks_2d, K):
    REFERENCE_LANDMARKS_3D = np.float32([
        [-2.0822412261220538, 1.697800522724807, 4.520114276952494],
        [-1.9801963763579469, 1.2931693187915096, 4.230872368967606],
        [-2.276916147971233, 0.930831168211703, 4.1528538228895435],
        [-1.8265286016971878, 0.8983242853496866, 4.356613008380495],
        [-2.791900772306086, 0.8693908260150472, 4.539910653815297],
        [-1.758480803201661, 0.7613888777243099, 5.032362007722343],
    ])
    
    _, Rvec, t = cv2.solvePnP(REFERENCE_LANDMARKS_3D, landmarks_2d, K, distCoeffs=None, useExtrinsicGuess = False)
    R, _  = cv2.Rodrigues(Rvec)
    
    Rt = np.hstack([R, t])
    f_row = np.zeros((1,4))
    f_row[0,-1] = 1
    Rt = np.vstack([
        Rt,
        f_row
    ])
    
    K = np.hstack([K, np.zeros((3,1))])
    K = np.vstack([
        K,
        f_row
    ])
    
    return K@Rt

image_cropper = ImageCropper()
def process_img(dir_path, out_path, scale_mat_0, focal_distance_x_face=14.0):
    working_dir = "./Graphonomy"
    
    ANCHOR_LANDMARK_IDXS = [8, 33, 37, 43, 0, 16]
    
    cur_img = imageio.imread(pathlib.Path(dir_path) / "rgb.jpg", pilmode='RGB')

    try:
        crop_rectangle, _, landmarks = image_cropper.crop_to_face(cur_img)
    except ImageCropper.BlinkException:
        logger.info(f"Blink in the image, skipping")

    l, t, r, b, _ = map(int, crop_rectangle)

    landmarks_2d = np.asarray(landmarks) # 68, 3
    face_height = landmarks_2d[:, 1].max() - landmarks_2d[:, 1].min()
    focal_distance_px = face_height * focal_distance_x_face

    landmarks_2d = landmarks_2d[ANCHOR_LANDMARK_IDXS, :2] # 6, 2

    human_data = {
        'crop_rectangles': [],
        'landmarks': [],
    }

    human_data['crop_rectangles'].append(crop_rectangle)
    human_data['landmarks'].append(landmarks)
    
    fx, fy = focal_distance_px, focal_distance_px
    h, w, _ = cur_img.shape
    cx = w/2
    cy = h/2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    our_world_mat = get_extrinsics_given_landmarks(landmarks_2d, K)
    our_world_mat = our_world_mat[:3] # 3x4 matrix

    # Save crop parameters and landmarks, just in case
    with open(out_path / "tabular_data.pkl", 'wb') as f:
        pickle.dump(human_data, f)
    
    undistorted_images_path = out_path / "image"
    masks_output_path = out_path / "mask"
    masks_output_path.mkdir(exist_ok=True)

    undistorted_images_path.mkdir(exist_ok=True)
    imageio.imwrite(undistorted_images_path / "00000.png", cur_img)

    if (pathlib.Path(dir_path) / "mask.png").exists():
        shutil.copy2(pathlib.Path(dir_path) / "mask.png", masks_output_path / "mask.png")
    else:
        command = [
            "python3", '-u', "./Graphonomy/exp/inference/inference_folder.py",
            '--images_path', f"{undistorted_images_path.resolve()}",
            '--output_dir', f"{masks_output_path.resolve()}",
            '--model_path', "data/model/universal_trained.pth",
            '--tta', "0.75,1.0,1.2,1.4",]
        run_subprocess(command, working_dir)

    cameras_sphere_npz_path = out_path / "cameras_sphere.npz"
    np.savez(
        cameras_sphere_npz_path,
        scale_mat_0 = scale_mat_0,
        world_mat_0 = our_world_mat
    )

    imageio.imwrite(undistorted_images_path / "00000.jpg", cur_img)
    
if __name__ == "__main__":
    data_path = "../../datasets/Gonzalo/004/2021-05-13-15-33-44/portrait_reconstruction/"
    cur_data_path = pathlib.Path(data_path)
    cameras_sphere_npz_path = cur_data_path / "cameras_sphere.npz"
    camera_matrices_file = np.load(cameras_sphere_npz_path)
    scale_mat_0 = camera_matrices_file['scale_mat_0']

    FOCAL_DISTANCE_X_FACE = 14.0

    in_path = pathlib.Path("inputs")
    
    for dir_path in tqdm.tqdm(glob.glob(str(in_path / "*"))):
        dir_name = os.path.basename(dir_path)
        out_path = pathlib.Path("outputs") / f"{dir_name}_{FOCAL_DISTANCE_X_FACE:.2f}Xface"
        if out_path.exists():
            print(f"{out_path} already exists, skipping")
            continue
        os.makedirs(str(out_path), exist_ok=True)
        process_img(dir_path, out_path, scale_mat_0, FOCAL_DISTANCE_X_FACE)

