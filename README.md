# 🌊 SurGe: Improved Surface Geometry in Point Maps

[[`Paper`](https://github.com/karimknaebel/storage/releases/download/surge-assets/surge-v1.pdf)] [[`arXiv`](https://arxiv.org/abs/2605.31577)] [[`Project Page`](http://vision.rwth-aachen.de/surge)] [[`Weights`](https://huggingface.co/karimknaebel/surge-large/tree/main)] [[`Demo`](https://huggingface.co/spaces/karimknaebel/surge)] [[`BibTeX`](#-Citation)]

<table><tr><td><img width="2491" height="1291" alt="architecture" src="https://github.com/user-attachments/assets/aab62446-15ec-478a-92ba-d9fd1c117812" /></td></tr></table>

## 📢 News

- 2026-08-21: evaluation code released.
- 2026-06-01: arXiv v1, inference code, weights, and demo released.

## Installation

### From a local clone

Clone the repository:

```bash
git clone https://github.com/karimknaebel/surge
cd surge
```

Then install SurGe with the Inference CLI, Gradio app, and evaluation dependencies:

```bash
uv sync --all-extras
```

This creates a virtual environment in `.venv/`.
Run commands with `uv run` (e.g., `uv run surge-infer ...`), or just activate it as usual with `source .venv/bin/activate`.

Alternatively, with pip: `pip install -e ".[cli,app,eval]"`

### As a library

Use SurGe in an existing project:

```bash
uv add git+https://github.com/karimknaebel/surge
```

Alternatively, with pip: `pip install git+https://github.com/karimknaebel/surge`

## Usage

### Python API

SurGe expects image tensors in `BCHW` format with unnormalized RGB values in `[0, 1]`.
Do not apply ImageNet normalization or similar preprocessing.

```python
import torch
from surge import SurGe

model = SurGe.from_pretrained("karimknaebel/surge-large").eval().cuda()
image = torch.rand(1, 3, 518, 518, device="cuda")

with torch.autocast("cuda", dtype=torch.float16):
    result = model.infer(image)
points = result["points"]          # (B, H, W, 3)
depth = result["depth"]            # (B, H, W)
intrinsics = result["intrinsics"]  # (B, 3, 3)
```

### 🚀 Inference CLI

Run inference on an image or a directory of images:

```
uv run surge-infer path/to/image.jpg --output-dir output
```

By default, the Inference CLI writes `mesh.glb` for each input image.
Add output flags as needed:

```
uv run surge-infer path/to/images --save-maps --save-glb --save-ply
```

Useful options include `--max-size 1200`, `--tokens max`, `--fov-x 60`, `--fp16`, and `--filter-sky`.
For interactive viewing, use `--show-mesh` to open the reconstructed mesh with trimesh, or `--rerun` to log the inference results to a Rerun viewer.

### 🖥️ Gradio App

Launch the local demo app:

```
uv run surge-app
```

The app lets you upload an image, adjust the token budget and mesh cleanup settings, view the reconstructed mesh, and download the generated maps and geometry.

### 📊 Evaluation

Download MoGe's processed evaluation datasets from [Hugging Face](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) and extract them under `data/eval`:

```bash
mkdir -p data/eval
uv run hf download Ruicheng/monocular-geometry-evaluation \
  --repo-type dataset \
  --local-dir data/eval

cd data/eval
unzip '*.zip'
cd ../..
```

Run the full suite:

```bash
uv run surge-eval --output eval_output/surge.json
```

We include [official surge-large results](eval_output/surge-large.d9de0808.json) for reference.

### Output Conventions

Coordinate frames:

- `point_map.exr` and `point_normal_map.png`: RDF (OpenCV); `+X` right, `+Y` down, `+Z` forward.
- `mesh.glb` and `point_cloud.ply`: RUB; `+X` right, `+Y` up, `+Z` backward.

File formats:

- `depth_colorized.png`: colorized depth visualization.
- `point_map.exr`: float32 XYZ points.
  Masked pixels are `NaN`.
- `point_normal_map.png`: unit normals.
  RGB stores `[X, -Y, -Z]` mapped from `[-1, 1]` to `[0, 255]`; invalid pixels are `[127, 127, 127]`.
  Renormalize after decoding if needed.
- `intrinsics.json`: camera intrinsics as a 3×3 JSON array.
- `fov.json`: horizontal and vertical fields of view in degrees.

## 🧩 Neighborhood Attention Decoder (NAD) Module

The [NAD](src/surge/models/modules/heads/nad.py) is implemented as a reusable PyTorch module.
It is intentionally self-contained, so you can copy it into your project as a single file without pulling in the rest of SurGe.

## 🧩 Point Gradient Matching ($\mathcal{L}_{\mathrm{pgm}}$)

The self-contained [point gradient matching loss](src/surge/losses.py) used to train SurGe is included for reference.

## 🧩 Point Map Normal Mean Angular Error ($\mathrm{MAE}_{\mathrm{normal}}$)

The [normal mean angular error metric](src/surge/evaluation/metrics.py#L122-L132) used to evaluate surface normals is included in the evaluation code.

## ⚖️ License

The **SurGe code** is released under the MIT license.
The **SurGe weights** are released under CC BY-NC 4.0, due to the training datasets used.

## 🙏 Acknowledgments

We thank the [MoGe](https://github.com/microsoft/moge) project for their open-source code.

## 🎓 Citation

If you use our work in your research, please use the following BibTeX entry.

```
@article{knaebel2026surge,
    title     = {{SurGe}: Improved Surface Geometry in Point Maps},
    author    = {Knaebel, Karim and Martin Garcia, Gonzalo and Schmidt, Christian and Fradlin, Ilya and Nunes, Lucas and de Geus, Daan and Leibe, Bastian},
    year      = 2026,
    journal   = {arXiv preprint arXiv:2605.31577},
}
```
