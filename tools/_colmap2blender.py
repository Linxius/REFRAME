"""Convert COLMAP dataset to blender format for REFRAME.

Based on da3recon/colmap_to_blender.py's correct conversion logic:
  R_nerf = R_c2w @ diag(1, -1, -1)

blenderdataset.py then does:
  new_rotation = R_nerf @ diag(1, -1, -1) = R_c2w  (for camera_R/camera_T near/far)
  self.mvps = projection @ inv(pose)  (pose has R_nerf, so inv gives OpenGL w2c)
"""
import numpy as np
import json
import os
import shutil
from pathlib import Path
from scipy.spatial.transform import Rotation


def read_colmap(dataset_path):
    """Read COLMAP data from images.txt (text format)."""
    sparse = Path(dataset_path) / "sparse" / "0"

    cameras = {}
    with open(sparse / "cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx, fy = params[0], params[1] if len(params) > 1 else params[0]
                cx, cy = params[2] if len(params) > 2 else width / 2, params[3] if len(params) > 3 else height / 2
            cameras[cam_id] = (width, height, fx, fy, cx, cy)

    images = {}
    with open(sparse / "images.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz]).reshape(3, 1)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3:] = t

        width, height, fx, fy, cx, cy = cameras[cam_id]
        images[img_id] = {
            "name": name, "w2c": w2c,
            "width": width, "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        }
    return images


def colmap_w2c_to_nerf_c2w(w2c):
    """Convert COLMAP w2c to NeRF/Blender c2w with flip baked in."""
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c
    flip = np.diag([1.0, -1.0, -1.0])
    R_nerf = R_c2w @ flip
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_nerf
    c2w[:3, 3] = t_c2w
    return c2w


def blender_c2w_to_w2c(c2w):
    """Simulate blenderdataset.py's w2c from stored c2w."""
    rotation = c2w[:3, :3]
    flip = np.diag([1.0, -1.0, -1.0])
    new_rotation = rotation @ flip
    w2c_rotation = np.linalg.inv(new_rotation)
    w2c_location = w2c_rotation @ (-c2w[:3, 3])
    return w2c_rotation, w2c_location


# Config
src = 'D:/Data/ref_real/gardenspheres'
dst = 'D:/Data/ref_real/gardenspheres_blender'
downscale = 4

# Read COLMAP
images_data = read_colmap(src)
sorted_ids = sorted(images_data.keys())
img_dir = Path(src) / "images"

entries = []
for img_id in sorted_ids:
    info = images_data[img_id]
    if (img_dir / info["name"]).exists():
        entries.append(info)

print(f'Images: {len(entries)}')
print(f'Camera: {entries[0]["width"]}x{entries[0]["height"]}, fx={entries[0]["fx"]:.1f}')

# Downscale intrinsics
W_orig = entries[0]["width"]
H_orig = entries[0]["height"]
fx_d = entries[0]["fx"] / downscale
fy_d = entries[0]["fy"] / downscale
cx_d = entries[0]["cx"] / downscale
cy_d = entries[0]["cy"] / downscale
W_d = W_orig // downscale
H_d = H_orig // downscale
print(f'Downsampled: {W_d}x{H_d}, fx={fx_d:.1f} fy={fy_d:.1f} cx={cx_d:.1f} cy={cy_d:.1f}')

# Create output
os.makedirs(os.path.join(dst, 'train'), exist_ok=True)
os.makedirs(os.path.join(dst, 'test'), exist_ok=True)

# Split: every 8th for test
all_indices = np.arange(len(entries))
val_ids = all_indices[::8]
train_ids = np.array([i for i in all_indices if i not in val_ids])

def build_transform(frames):
    return {
        "fl_x": float(fx_d), "fl_y": float(fy_d),
        "cx": float(cx_d), "cy": float(cy_d),
        "w": int(W_d), "h": int(H_d),
        "frames": frames
    }

train_frames = []
for i in train_ids:
    info = entries[i]
    src_img = src + '/images_4/' + info['name']
    if not os.path.exists(src_img):
        src_img = str(img_dir / info['name'])
    dst_img = os.path.join(dst, 'train', info['name'])
    if not os.path.exists(dst_img):
        shutil.copy2(src_img, dst_img)
    c2w = colmap_w2c_to_nerf_c2w(info['w2c'])
    train_frames.append({
        "file_path": f"./train/{info['name']}",
        "transform_matrix": c2w.tolist()
    })

test_frames = []
for i in val_ids:
    info = entries[i]
    src_img = src + '/images_4/' + info['name']
    if not os.path.exists(src_img):
        src_img = str(img_dir / info['name'])
    dst_img = os.path.join(dst, 'test', info['name'])
    if not os.path.exists(dst_img):
        shutil.copy2(src_img, dst_img)
    c2w = colmap_w2c_to_nerf_c2w(info['w2c'])
    test_frames.append({
        "file_path": f"./test/{info['name']}",
        "transform_matrix": c2w.tolist()
    })

with open(os.path.join(dst, 'transforms_train.json'), 'w') as f:
    json.dump(build_transform(train_frames), f, indent=2)
with open(os.path.join(dst, 'transforms_test.json'), 'w') as f:
    json.dump(build_transform(test_frames), f, indent=2)

print(f'Train: {len(train_frames)}, Test: {len(test_frames)}')
print(f'Output: {dst}')

# Roundtrip verification
print('\n--- Roundtrip verification ---')
max_diff = 0
for i in range(min(5, len(train_frames))):
    c2w = np.array(train_frames[i]['transform_matrix'])
    R_w2c, t_w2c = blender_c2w_to_w2c(c2w)
    w2c_check = np.eye(4)
    w2c_check[:3, :3] = R_w2c
    w2c_check[:3, 3] = t_w2c
    orig_w2c = entries[train_ids[i]]['w2c']
    diff = np.abs(w2c_check - orig_w2c).max()
    max_diff = max(max_diff, diff)
print(f'Max roundtrip error: {max_diff:.2e} (expect <1e-10)')
