import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward/rgb/frame_0000.png")
mask = load_mask("/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward/masks/00000.png")

USE_DEPTH_AND_CAM_K = False
if USE_DEPTH_AND_CAM_K:
    import numpy as np
    from PIL import Image
    depth_mm = np.array(Image.open("/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward/depth/frame_0000.png"))
    depth_m = depth_mm / 1000.0
    cam_K = np.loadtxt("/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward/cam_K.txt")
    assert cam_K.shape == (3, 3), f"cam_K.shape: {cam_K.shape}, expected: (3, 3)"

    # run model with depth and cam_K
    output = inference(image, mask, seed=42, depth=depth_m, cam_K=cam_K)
else:
    # run model
    output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")

output_mesh = output["mesh"][0]
vertices = output_mesh.vertices.cpu().numpy()
faces = output_mesh.faces.cpu().numpy()
import trimesh
mesh = trimesh.Trimesh(vertices, faces)
mesh.export("mesh.obj")
bounds = mesh.bounds
assert bounds.shape == (2, 3), f"bounds.shape: {bounds.shape}"
size = bounds[1] - bounds[0]
print(f"bounds: {bounds}")
print(f"size: {size}")

print(f"output.keys(): {output.keys()}")
for key, value in output.items():
    print(f"key: {key}, value: {value}")

import viser
import numpy as np
from scipy.spatial.transform import Rotation as R
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
server.scene.add_mesh_simple(
    name="/mesh",
    vertices=vertices,
    faces=faces,
)
pointmap = output["pointmap"].cpu().numpy().reshape(-1, 3)
pointmap_colors = (output["pointmap_colors"].cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
server.scene.add_point_cloud( name="/pointmap", points=pointmap, colors=pointmap_colors, point_size=0.002,)
scale = output["scale"][0].cpu().numpy()
translation = output["translation"][0].cpu().numpy()
translation_scale = output["translation_scale"].item()
rotation6d_normalized = output["6drotation_normalized"][0, 0].cpu().numpy()
coords_original = output["coords_original"].cpu().numpy()
coords = output["coords"].cpu().numpy()
rotation = output["rotation"][0].cpu().numpy()
print(f"scale: {scale}")
print(f"translation: {translation}")
print(f"translation_scale: {translation_scale}")
print(f"rotation6d_normalized: {rotation6d_normalized}")
print(f"coords_original: {coords_original.shape}")
print(f"coords: {coords.shape}")
print(f"rotation: {rotation}")
mesh_scaled = mesh.copy()
mesh_scaled.apply_transform(np.diag([scale[0], scale[1], scale[2], 1]))
server.scene.add_mesh_simple(
    name="/mesh_scaled",
    vertices=mesh_scaled.vertices,
    faces=mesh_scaled.faces,
)
# translation_scaled = translation * scale
translation_scaled = translation  # This one looks correct
print(f"translation_scaled = {translation_scaled}")
print(f"rotation = {rotation}")

def rot6d_to_matrix(rot6d):
    """
    rot6d: np.ndarray of shape (6,)
    returns: (3,3) rotation matrix
    """
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]

    # First basis vector
    b1 = a1 / np.linalg.norm(a1)

    # Make second vector orthogonal to first
    a2_ortho = a2 - np.dot(b1, a2) * b1
    b2 = a2_ortho / np.linalg.norm(a2_ortho)

    # Third basis via cross product
    b3 = np.cross(b1, b2)

    # Assemble rotation matrix (columns)
    Rmat = np.stack([b1, b2, b3], axis=1)
    return Rmat

rotation_matrix = rot6d_to_matrix(rotation6d_normalized)

mesh_moved_1 = mesh_scaled.copy()
T_1 = np.eye(4)
T_1[:3, :3] = rotation_matrix
T_1[:3, 3] = translation_scaled
mesh_moved_1.apply_transform(T_1)
server.scene.add_mesh_simple(
    name="/mesh_moved_1",
    vertices=mesh_moved_1.vertices,
    faces=mesh_moved_1.faces,
)
quat_xyzw_2 = np.array([rotation[0], rotation[1], rotation[2], rotation[3]])  # Not sure about this
rotation_matrix_2 = R.from_quat(quat_xyzw_2).as_matrix()
T_2 = np.eye(4)
T_2[:3, :3] = rotation_matrix_2
T_2[:3, 3] = translation_scaled
mesh_moved_2 = mesh_scaled.copy()
mesh_moved_2.apply_transform(T_2)
server.scene.add_mesh_simple(
    name="/mesh_moved_2",
    vertices=mesh_moved_2.vertices,
    faces=mesh_moved_2.faces,
)
print(f"rotation6d_normalized = {rotation6d_normalized}")

breakpoint()

