import numpy as np
import open3d as o3d

pc = np.load("data/modelnet10_pc_2048/chair/train/chair_0001.npy")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([pcd])
