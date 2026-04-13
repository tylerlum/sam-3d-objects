import numpy as np
import viser
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


def convert_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    # depth is either in meters or millimeters
    # Need to convert to meters
    # If the max value is greater than 100, then it's likely in mm
    in_mm = depth.max() > 100
    if in_mm:
        return depth / 1000
    else:
        return depth

def depth_to_points(
    depth_m: np.ndarray,
    K: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    stride: int = 1,
    max_depth_m: float = np.inf,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = depth_m.shape
    v_coords, u_coords = np.indices((h, w))
    if stride > 1:
        v_coords = v_coords[::stride, ::stride]
        u_coords = u_coords[::stride, ::stride]
        depth = depth_m[::stride, ::stride]
        if rgb is not None:
            colors = rgb[::stride, ::stride, :]
        else:
            colors = None
    else:
        depth = depth_m
        colors = rgb

    z = depth.reshape(-1)
    valid = (z >= 0.0) & (z < max_depth_m)
    z = z[valid]

    u = u_coords.reshape(-1)[valid]
    v = v_coords.reshape(-1)[valid]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_c = np.stack([x, y, z], axis=1)

    cols = colors.reshape(-1, 3)[valid]
    return pts_c, cols


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R_rc = T[:3, :3]
    t_rc = T[:3, 3]
    return (pts @ R_rc.T) + t_rc[None, :]

def compute_point_cloud_from_paths(rgb_path: Path, depth_path: Path, cam_intrinsics_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # Read in camera stuff
    assert rgb_path.exists(), f"RGB path {rgb_path} does not exist"
    assert depth_path.exists(), f"Depth path {depth_path} does not exist"
    assert cam_intrinsics_path.exists(), f"Cam intrinsics path {cam_intrinsics_path} does not exist"

    # RGB
    rgb = np.array(Image.open(rgb_path))
    assert len(rgb.shape) == 3, f"RGB shape: {rgb.shape}, expected: (H, W, 3)"
    H, W, C = rgb.shape
    assert C == 3, f"RGB shape: {rgb.shape}, expected: ({H}, {W}, 3)"

    # Depth
    depth = np.array(Image.open(depth_path))
    depth = convert_depth_to_meters(depth)
    assert depth.shape == (H, W), f"Depth shape: {depth.shape}, expected: ({H}, {W})"
    print(f"Min depth: {np.min(depth)}, Max depth: {np.max(depth)}")
    print(f"Mean depth: {np.mean(depth)}, Median depth: {np.median(depth)}")

    # Camera intrinsics
    K = np.loadtxt(cam_intrinsics_path)
    assert K.shape == (3, 3), f"K shape: {K.shape}, expected: (3, 3)"

    return compute_point_cloud(rgb=rgb, depth=depth, K=K)

def compute_point_cloud(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, subsample_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    assert len(rgb.shape) == 3, f"RGB shape: {rgb.shape}, expected: (H, W, 3)"
    H, W, C = rgb.shape
    assert C == 3, f"RGB shape: {rgb.shape}, expected: ({H}, {W}, 3)"
    assert depth.shape == (H, W), f"Depth shape: {depth.shape}, expected: ({H}, {W})"
    assert K.shape == (3, 3), f"K shape: {K.shape}, expected: (3, 3)"

    pts_c, cols = depth_to_points(
        depth, K, rgb=rgb, stride=1, max_depth_m=np.inf,
    )
    N_PTS = len(pts_c)
    assert pts_c.shape == (N_PTS, 3), f"pts_c shape: {pts_c.shape}, expected: ({N_PTS}, 3)"
    assert cols.shape == (N_PTS, 3), f"cols shape: {cols.shape}, expected: ({N_PTS}, 3)"
    print(f"N_PTS: {N_PTS}")

    pts_c = pts_c[::subsample_factor].astype(np.float32)
    cols = cols[::subsample_factor].astype(np.uint8)
    return pts_c, cols

def add_point_cloud_to_viser(pts_c: np.ndarray, cols: np.ndarray, T_W_C: Optional[np.ndarray] = None, point_size: float = 0.002, server: Optional[viser.ViserServer] = None, name: str = "/point_cloud") -> Tuple[viser.PointCloudHandle, viser.ViserServer]:
    if T_W_C is not None:
        pts = transform_points(T=T_W_C, pts=pts_c)
    else:
        pts = pts_c

    if server is None:
        server = viser.ViserServer()
    pcd_handle = server.scene.add_point_cloud(
        name,
        points=pts.astype(np.float32),
        colors=cols.astype(np.uint8),
        point_size=point_size,
    )

    return pcd_handle, server