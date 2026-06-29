"""Decimate mesh for faster training."""
import trimesh
import sys

src = sys.argv[1] if len(sys.argv) > 1 else 'D:/Code/da3recon/recon_output/07_reconstruction_pose.ply'
dst = sys.argv[2] if len(sys.argv) > 2 else 'D:/Data/ref_real/gardenspheres_blender/points_of_interest_decimated.ply'
target_faces = int(sys.argv[3]) if len(sys.argv) > 3 else 100000

m = trimesh.load(src, process=False)
print(f'Original: {len(m.vertices)} verts, {len(m.faces)} faces')
m = m.simplify_quadric_decimation(face_count=target_faces)
m.export(dst)
print(f'Decimated: {len(m.vertices)} verts, {len(m.faces)} faces')
print(f'Saved to {dst}')
