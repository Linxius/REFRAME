"""Check if blenderdataset.py loads our JSON correctly."""
import sys, numpy as np
sys.path.insert(0, 'D:/Code/REFRAME')
import torch
from argparse import Namespace
import trimesh
from reframe import BlenderDataset

device = torch.device('cpu')
args = Namespace(
    datadir='D:/Data/ref_real/gardenspheres_blender',
    scale=0.8, bound=1.0, foreground=0.5,
    refneus=1, dataset='blender', region=1,
    views_per_iter=1, wenvlearner=1, L=16,
    ssaa=1, device=0, test=0, uvmap=0,
    mlpoff=1, shading_percentage=1,
    resolutionx=360, resolutiony=720,
)

vertices = None
mesh_ = trimesh.load_mesh('D:/Code/da3recon/recon_output/07_reconstruction_pose.ply', process=False)
vertices = np.array(mesh_.vertices, dtype=np.float32)

ds = BlenderDataset(args, device=device, epsilon=0.5, type='train', vertices=vertices)

print(f'Images: {len(ds.images)}, H={ds.H}, W={ds.W}')
print(f'Poses shape: {ds.poses.shape}')
print(f'Camera radius: {ds.radius:.3f}')
print(f'Has scale_mat: {hasattr(ds, "scale_mat")}')
if hasattr(ds, 'scale_mat'):
    print(f'scale_mat diag: {np.diag(ds.scale_mat)[:3]}')

v = mesh_.vertices
inv_sm = np.linalg.inv(ds.scale_mat)
verts_t = v @ inv_sm[:3, :3].T + inv_sm[:3, 3]
print(f'Mesh (after scale_mat): {verts_t.min(0).round(3)} to {verts_t.max(0).round(3)}')

cam_centers = ds.poses[:, :3, 3].numpy()
print(f'Camera centers: {cam_centers.min(0).round(3)} to {cam_centers.max(0).round(3)}')
