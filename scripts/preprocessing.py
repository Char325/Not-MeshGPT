#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import trimesh
from tqdm import tqdm

# ----------------------------
# Utility: Normalize Mesh
# ----------------------------
def normalize_mesh(mesh):
    """
    Center mesh at origin and scale to unit sphere.
    """
    vertices = mesh.vertices
    vertices = vertices - vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(vertices, axis=1))
    vertices = vertices / scale
    mesh.vertices = vertices
    return mesh


# ----------------------------
# Utility: Sample Point Cloud
# ----------------------------
def mesh_to_pointcloud(mesh, n_points=2048):
    """
    Sample n_points from the surface of the mesh.
    Returns (n_points, 3) numpy array.
    """
    try:
        points, face_idx = mesh.sample(n_points, return_index=True)
        return points.astype(np.float32)
    except Exception as e:
        print(f"[WARNING] Sampling error: {e}")
        return None


# ----------------------------
# Main Preprocessing Function
# ----------------------------
def preprocess_modelnet10(input_dir, output_dir, n_points=2048):
    """
    Convert ModelNet10 OFF meshes → normalized point clouds (.npy).
    Keeps class/train/test structure.
    """
    classes = sorted(os.listdir(input_dir))
    print(f"Found classes: {classes}\n")

    for cls in classes:
        class_in = os.path.join(input_dir, cls)
        if not os.path.isdir(class_in):
            continue

        # Train/Test folders
        for split in ["train", "test"]:
            split_in_dir = os.path.join(class_in, split)
            if not os.path.exists(split_in_dir):
                continue

            split_out_dir = os.path.join(output_dir, cls, split)
            os.makedirs(split_out_dir, exist_ok=True)

            off_files = [
                f for f in os.listdir(split_in_dir) if f.endswith(".off")
            ]

            print(f"Processing: {cls}/{split}  ({len(off_files)} files)")

            for fname in tqdm(off_files):
                in_path = os.path.join(split_in_dir, fname)
                out_name = fname.replace(".off", ".npy")
                out_path = os.path.join(split_out_dir, out_name)

                try:
                    mesh = trimesh.load(in_path)

                    # Normalize
                    mesh = normalize_mesh(mesh)

                    # Sample point cloud
                    points = mesh_to_pointcloud(mesh, n_points=n_points)
                    if points is None:
                        continue

                    # Save
                    np.save(out_path, points)

                except Exception as e:
                    print(f"[ERROR] Failed on {in_path}: {e}")


# ----------------------------
# Argument Parser
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ModelNet10 into point clouds.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to raw ModelNet10 directory (contains class folders).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where processed .npy files should be saved.")
    parser.add_argument("--points", type=int, default=2048,
                        help="Number of sampled points per mesh (default: 2048).")

    args = parser.parse_args()

    print("\n========== MODELNET10 PREPROCESSING ==========\n")
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Points per mesh : {args.points}\n")

    preprocess_modelnet10(args.input_dir, args.output_dir, n_points=args.points)

    print("\n✨ Preprocessing complete! All point clouds saved.\n")
