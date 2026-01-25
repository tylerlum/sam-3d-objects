import sys
import argparse
import viser
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask


def get_meshes(output) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    from scipy.spatial.transform import Rotation as R

    BATCH_IDX = 0

    # Get the mesh
    original_mesh = output["glb"]

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

    # GLB code does a change of basis from z-up to y-up (to_glb)
    # We need to invert this to get the original mesh
    # It used this rotation matrix with right-multiplication
    # So we use the same matrix with left-multiplication to invert it
    glb_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    glb_T = np.eye(4)
    glb_T[:3, :3] = glb_rotation
    mesh.apply_transform(glb_T)

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


def save_mesh(mesh: trimesh.Trimesh, name: str, parent_dir: Path):
    """
    Converts a vertex-colored mesh to a textured mesh (obj+mtl+png)
    and saves it in its own subdirectory to avoid file conflicts.
    """
    # 1. Create a subdirectory for this specific mesh
    # e.g. output/original_mesh/
    save_dir = parent_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Export as .glb
    file_path = save_dir / f"{name}.glb"
    mesh.export(file_path)

    # 3. Export as .obj
    # This will produce: '{name}.obj', 'material_0.mtl', 'material_0.png'
    new_mesh = trimesh.load(file_path)
    new_mesh.export(save_dir / f"{name}.obj", file_type="obj")
    return file_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on a directory with rgb, masks, and depth subdirectories")
    parser.add_argument("--input_dir", type=Path, help="Directory containing rgb/, masks/, and depth/ subdirectories")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="Output directory (default: output)")
    parser.add_argument("--mesh_mode", type=str, default="texture", help="Mesh mode: texture or vertex_color. texture requires nvdiffrast.")
    args = parser.parse_args()

    input_dir = args.input_dir
    OUTPUT_DIR = args.output_dir

    # Validate input directory structure
    rgb_dir = input_dir / "rgb"
    masks_dir = input_dir / "masks"
    depth_dir = input_dir / "depth"
    cam_K_path = input_dir / "cam_K.txt"

    assert rgb_dir.exists(), f"rgb directory not found: {rgb_dir}"
    assert masks_dir.exists(), f"masks directory not found: {masks_dir}"
    assert depth_dir.exists(), f"depth directory not found: {depth_dir}"
    assert cam_K_path.exists(), f"cam_K.txt not found: {cam_K_path}"

    # Get first image from each directory
    rgb_files = sorted(rgb_dir.glob("*.png"))
    mask_files = sorted(masks_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))

    assert len(rgb_files) > 0, f"No png files found in {rgb_dir}"
    assert len(mask_files) > 0, f"No png files found in {masks_dir}"
    assert len(depth_files) > 0, f"No png files found in {depth_dir}"

    rgb_path = rgb_files[0]
    mask_path = mask_files[0]
    depth_path = depth_files[0]

    print(f"Using rgb: {rgb_path}")
    print(f"Using mask: {mask_path}")
    print(f"Using depth: {depth_path}")

    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # load image and mask
    image = load_image(str(rgb_path))
    mask = load_mask(str(mask_path))

    # load depth and cam_K
    depth_mm = np.array(Image.open(depth_path))
    depth_m = depth_mm / 1000.0
    cam_K = np.loadtxt(cam_K_path)
    assert cam_K.shape == (3, 3), f"cam_K.shape: {cam_K.shape}, expected: (3, 3)"

    # run model with depth and cam_K
    output = inference(image, mask, seed=42, depth=depth_m, cam_K=cam_K, mesh_mode=args.mesh_mode)

    # export gaussian splat
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splat_path = OUTPUT_DIR / "splat.ply"
    output["gs"].save_ply(splat_path)
    print(f"Your reconstruction has been saved to {splat_path}")

    original_mesh, mesh, posed_mesh = get_meshes(output)
    pointmap, pointmap_colors = get_point_cloud(output)

    save_mesh(
        mesh=original_mesh,
        name="original_mesh",
        parent_dir=OUTPUT_DIR,
    )
    save_mesh(
        mesh=mesh,
        name="mesh",
        parent_dir=OUTPUT_DIR,
    )
    save_mesh(
        mesh=posed_mesh,
        name="posed_mesh",
        parent_dir=OUTPUT_DIR,
    )

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


if __name__ == "__main__":
    main()
