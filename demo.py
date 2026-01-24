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

USE_DEPTH_AND_CAM_K = True
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

mesh = output["mesh"][0]
vertices = mesh.vertices.cpu().numpy()
faces = mesh.faces.cpu().numpy()
import trimesh
trimesh = trimesh.Trimesh(vertices, faces)
trimesh.export("mesh.obj")
bounds = trimesh.bounds
assert bounds.shape == (2, 3), f"bounds.shape: {bounds.shape}"
size = bounds[1] - bounds[0]
print(f"bounds: {bounds}")
print(f"size: {size}")

print(f"output.keys(): {output.keys()}")
breakpoint()
