import sys
import numpy as np
import viser
import trimesh
from pathlib import Path

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward/rgb/frame_0000.png")
mask = load_mask("/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward/masks/00000.png")

USE_DEPTH_AND_CAM_K = True
if USE_DEPTH_AND_CAM_K:
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
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
splat_path = OUTPUT_DIR / "splat.ply"
output["gs"].save_ply(splat_path)
print(f"Your reconstruction has been saved to {splat_path}")

def get_meshes(output) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    from scipy.spatial.transform import Rotation as R

    BATCH_IDX = 0

    # Get the mesh
    output_mesh = output["mesh"][BATCH_IDX]
    vertices = output_mesh.vertices.cpu().numpy()
    faces = output_mesh.faces.cpu().numpy()

    # See if it has colors
    # 1. Check if attributes exist
    if output_mesh.vertex_attrs is not None:
        # 2. Extract attributes (N, 6)
        attrs = output_mesh.vertex_attrs.cpu().numpy()

        # 3. Slice the RGB channels (usually the first 3)
        # Note: These are likely in range [0, 1] floats. Trimesh handles this.
        vertex_colors = attrs[:, :3]

        # Optional: If colors look weird, they might be Normals or BGR.
        # But based on the "6 channel" comment, :3 is the standard guess.
    else:
        print("WARNING: No vertex attributes found (use_color=False?)")
        vertex_colors = None

    original_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)

    # Get the scale
    scale = output["scale"][0].cpu().numpy()
    mesh = original_mesh.copy()
    mesh.apply_transform(np.diag([scale[0], scale[1], scale[2], 1]))

    # Get the bounds
    bounds = mesh.bounds
    assert bounds.shape == (2, 3), f"bounds.shape: {bounds.shape}"
    size = bounds[1] - bounds[0]
    print(f"bounds: {bounds}")
    print(f"size: {size}")

    # Get the translation
    translation = output["translation"][BATCH_IDX].cpu().numpy()
    rotation = output["rotation"][BATCH_IDX].cpu().numpy()
    quat_wxyz = np.array([rotation[0], rotation[1], rotation[2], rotation[3]])  # [w, x, y, z]
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    rotation_matrix = R.from_quat(quat_xyzw).as_matrix().T  # Need to transpose this
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    posed_mesh = mesh.copy()
    posed_mesh.apply_transform(T)
    return original_mesh, mesh, posed_mesh

def get_point_cloud(output) -> tuple[np.ndarray, np.ndarray]:
    pointmap = output["pointmap"].cpu().numpy().reshape(-1, 3)
    pointmap_colors = (output["pointmap_colors"].cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
    return pointmap, pointmap_colors

original_mesh, mesh, posed_mesh = get_meshes(output)
pointmap, pointmap_colors = get_point_cloud(output)

original_mesh_path = OUTPUT_DIR / "original_mesh.obj"
mesh_path = OUTPUT_DIR / "mesh.obj"
posed_mesh_path = OUTPUT_DIR / "posed_mesh.obj"
original_mesh.export(original_mesh_path)
mesh.export(mesh_path)
posed_mesh.export(posed_mesh_path)
print(f"Saved to {original_mesh_path}, {mesh_path}, {posed_mesh_path}")

server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
server.scene.add_mesh_simple(
    name="/original_mesh",
    vertices=original_mesh.vertices,
    faces=original_mesh.faces,
)
server.scene.add_mesh_simple(
    name="/mesh",
    vertices=mesh.vertices,
    faces=mesh.faces,
)
server.scene.add_mesh_simple(
    name="/posed_mesh",
    vertices=posed_mesh.vertices,
    faces=posed_mesh.faces,
)
server.scene.add_point_cloud(
    name="/pointmap",
    points=pointmap,
    colors=pointmap_colors,
    point_size=0.002,
)
breakpoint()
