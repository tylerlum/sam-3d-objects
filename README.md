<!-- SimToolReal additions -->
# SimToolReal Pipeline

This repo now includes a SimToolReal-oriented inference path on top of the upstream SAM 3D Objects codebase.

At a high level, the local changes add:

- `run_inference.py` for running SAM 3D Objects on a directory containing RGB, mask, depth, and camera intrinsics.
- A `depth + cam_K` inference path so point clouds can be built from external depth rather than only the model-predicted pointmap.
- A `tyro`-based CLI for `run_inference.py`.
- UV-friendly environment fixes and runtime fixes, including support for textured output via `nvdiffrast`.

This section is specific to the SimToolReal workflow. The original upstream README is preserved below.

## SimToolReal Installation

These steps use `uv` with Python 3.11.

```bash
cd /home/tylerlum/github_repos/sam-3d-objects
uv venv .venv311 --python 3.11
source .venv311/bin/activate
```

Install PyTorch CUDA 12.1 wheels:

```bash
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the main compiled dependencies:

```bash
uv pip install --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" \
  spconv-cu121==2.3.8 \
  kaolin==0.17.0 \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
  "moge @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
```

Install the repo and the CLI dependency:

```bash
uv pip install tyro==0.9.35
```

`nvdiffrast` is only required for `--mesh_mode texture`. If you only want `vertex_color`, you can skip this step.

```bash
uv pip install --no-build-isolation \
  "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"
```

## Checkpoints

Authenticate with Hugging Face and download the SAM 3D Objects checkpoints:

```bash
source .venv311/bin/activate
hf auth login

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

After this, you should have `checkpoints/hf/pipeline.yaml` plus the related checkpoint files.

## CLI Usage

```bash
source .venv311/bin/activate
python run_inference.py --input_dir /path/to/sequence --mesh_mode texture
```

Current CLI help:

```text
usage: run_inference.py [-h] --input_dir PATH [--output_dir PATH] [--mesh_mode STR]

╭─ options ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                             │
│ --input_dir PATH        Directory containing `rgb/`, `masks/`, `depth/`, and `cam_K.txt`. (required)                │
│ --output_dir PATH       Output directory for splat and mesh artifacts. (default: output)                            │
│ --mesh_mode STR         Mesh mode: `texture` or `vertex_color`. `texture` requires `nvdiffrast`. (default: texture) │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Input Structure

`run_inference.py` expects an input directory shaped like:

```text
<input_dir>/
  cam_K.txt
  rgb/
    *.png
  masks/
    *.png
  depth/
    *.png
```

Behavior:

- The script reads the first `.png` file from `rgb/`, `masks/`, and `depth/`.
- `cam_K.txt` must be a `3 x 3` camera intrinsics matrix.
- `depth/` is expected to contain metric depth stored in millimeters; the script converts it to meters internally.

Example:

```bash
python run_inference.py \
  --input_dir /juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward \
  --mesh_mode texture
```

## Output Structure

By default, outputs are written to `output/` unless `--output_dir` is provided.

Expected outputs include:

```text
<output_dir>/
  splat.ply
  original_mesh/
    original_mesh.glb
    original_mesh.obj
    ...
  mesh/
    mesh.glb
    mesh.obj
    ...
  posed_mesh/
    posed_mesh.glb
    posed_mesh.obj
    ...
```

Notes:

- `splat.ply` is the exported Gaussian splat.
- `original_mesh`, `mesh`, and `posed_mesh` are each written into their own subdirectory.
- OBJ export may also generate material and texture sidecar files depending on mesh mode and export path.

---

## Original Upstream README

# SAM 3D

SAM 3D Objects is one part of SAM 3D, a pair of models for object and human mesh reconstruction.  If you’re looking for SAM 3D Body, [click here](https://github.com/facebookresearch/sam-3d-body).

# SAM 3D Objects

**SAM 3D Team**, [Xingyu Chen](https://scholar.google.com/citations?user=gjSHr6YAAAAJ&hl=en&oi=sra)\*, [Fu-Jen Chu](https://fujenchu.github.io/)\*, [Pierre Gleize](https://scholar.google.com/citations?user=4imOcw4AAAAJ&hl=en&oi=ao)\*, [Kevin J Liang](https://kevinjliang.github.io/)\*, [Alexander Sax](https://alexsax.github.io/)\*, [Hao Tang](https://scholar.google.com/citations?user=XY6Nh9YAAAAJ&hl=en&oi=sra)\*, [Weiyao Wang](https://sites.google.com/view/weiyaowang/home)\*, [Michelle Guo](https://scholar.google.com/citations?user=lyjjpNMAAAAJ&hl=en&oi=ao), [Thibaut Hardin](https://github.com/Thibaut-H), [Xiang Li](https://ryanxli.github.io/)⚬, [Aohan Lin](https://github.com/linaohan), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ziqi Ma](https://ziqi-ma.github.io/)⚬, [Anushka Sagar](https://www.linkedin.com/in/anushkasagar/), [Bowen Song](https://scholar.google.com/citations?user=QQKVkfcAAAAJ&hl=en&oi=sra)⚬, [Xiaodong Wang](https://scholar.google.com/citations?authuser=2&user=rMpcFYgAAAAJ), [Jianing Yang](https://jedyang.com/)⚬, [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen/)⚬, [Piotr Dollár](https://pdollar.github.io/)†, [Georgia Gkioxari](https://georgiagkioxari.com/)†, [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao)†§, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†§

***Meta Superintelligence Labs***

*Core contributor (Alphabetical, Equal Contribution), ⚬Intern, †Project leads, §Equal Contribution

[[`Paper`](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)] [[`Code`](https://github.com/facebookresearch/sam-3d-objects)] [[`Website`](https://ai.meta.com/sam3d/)] [[`Demo`](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d)] [[`Blog`](https://ai.meta.com/blog/sam-3d/)] [[`BibTeX`](#citing-sam-3d-objects)]

**SAM 3D Objects** is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image, excelling in real-world scenarios with occlusion and clutter by using progressive training and a data engine with human feedback. It outperforms prior 3D generation models in human preference tests on real-world objects and scenes. We released code, weights, online demo, and a new challenging benchmark.


<p align="center"><img src="doc/intro.png"/></p>

-----

<p align="center"><img src="doc/arch.png"/></p>

## Latest updates

**11/19/2025** - Checkpoints Launched, Web Demo and Paper are out.

## Installation

Follow the [setup](doc/setup.md) steps before running the following.

## Single or Multi-Object 3D Generation

SAM 3D Objects can convert masked objects in an image, into 3D models with pose, shape, texture, and layout. SAM 3D is designed to be robust in challenging natural images, handling small objects and occlusions, unusual poses, and difficult situations encountered in uncurated natural scenes like this kidsroom:

<p align="center">
  <img src="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png" width="55%"/>
  <img src="doc/kidsroom_transparent.gif" width="40%"/>
</p>

For a quick start, run `python demo.py` or use the the following lines of code:

```python
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image and mask
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
```

For  more details and multi-object reconstruction, please take a look at out two jupyter notebooks:
* [single object](notebook/demo_single_object.ipynb)
* [multi object](notebook/demo_multi_object.ipynb)


## SAM 3D Body

[SAM 3D Body (3DB)](https://github.com/facebookresearch/sam-3d-body) is a robust promptable foundation model for single-image 3D human mesh recovery (HMR).

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](notebook/demo_3db_mesh_alignment.ipynb).

## License

The SAM 3D Objects model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Objects project was made possible with the help of many contributors.

Robbie Adkins,
Paris Baptiste,
Karen Bergan,
Kai Brown,
Michelle Chan,
Ida Cheng,
Khadijat Durojaiye,
Patrick Edwards,
Daniella Factor,
Facundo Figueroa,
Rene  de la Fuente,
Eva Galper,
Cem Gokmen,
Alex He,
Enmanuel Hernandez,
Dex Honsa,
Leonna Jones,
Arpit Kalla,
Kris Kitani,
Helen Klein,
Kei Koyama,
Robert Kuo,
Vivian Lee,
Alex Lende,
Jonny Li,
Kehan Lyu,
Faye Ma,
Mallika Malhotra,
Sasha Mitts,
William Ngan,
George Orlin,
Peter Park,
Don Pinkus,
Roman Radle,
Nikhila Ravi,
Azita Shokrpour,
Jasmine Shone,
Zayida Suber,
Phillip Thomas,
Tatum Turner,
Joseph Walker,
Meng Wang,
Claudette Ward,
Andrew Westbury,
Lea Wilken,
Nan Yang,
Yael Yungster


## Citing SAM 3D Objects

If you use SAM 3D Objects in your research, please use the following BibTeX entry.

```
@article{sam3dteam2025sam3d3dfyimages,
      title={SAM 3D: 3Dfy Anything in Images}, 
      author={SAM 3D Team and Xingyu Chen and Fu-Jen Chu and Pierre Gleize and Kevin J Liang and Alexander Sax and Hao Tang and Weiyao Wang and Michelle Guo and Thibaut Hardin and Xiang Li and Aohan Lin and Jiawei Liu and Ziqi Ma and Anushka Sagar and Bowen Song and Xiaodong Wang and Jianing Yang and Bowen Zhang and Piotr Dollár and Georgia Gkioxari and Matt Feiszli and Jitendra Malik},
      year={2025},
      eprint={2511.16624},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16624}, 
}
```
