# 🌊 SurGe: Improved Surface Geometry in Point Maps

[[`Paper`](https://github.com/karimknaebel/storage/releases/download/surge-assets/surge-v1.pdf)] [[`arXiv`](https://arxiv.org/abs/2605.31577)] [[`Project Page`](http://vision.rwth-aachen.de/surge)] [[`Weights`](https://huggingface.co/karimknaebel/surge-large/tree/main)] [[`Demo`](https://huggingface.co/spaces/karimknaebel/surge)] [[`BibTeX`](#-Citation)]

<table><tr><td><img width="2491" height="1291" alt="architecture" src="https://github.com/user-attachments/assets/aab62446-15ec-478a-92ba-d9fd1c117812" /></td></tr></table>

## 📢 News

- 2026-06-01: arXiv v1, inference code, weights and demo.

## Installation

```
uv sync --extra cli --extra app
```

## Usage

### Python API

SurGe expects image tensors in `BCHW` format with unnormalized RGB values in
`[0, 1]`. Do not apply ImageNet normalization or similar preprocessing.

> [!TIP]
If you want the exact model from the arXiv v1 paper, you can `git checkout v1` and specify corresponding checkpoint version `SurGe.from_pretrained("karimknaebel/surge-large", revision="v1")`.

```python
import torch
from surge import SurGe

model = SurGe.from_pretrained("karimknaebel/surge-large").eval().cuda()
image = torch.rand(1, 3, 518, 518, device="cuda")

result = model.infer(image, num_tokens="max")
points = result["points"]          # (B, H, W, 3)
depth = result["depth"]            # (B, H, W)
intrinsics = result["intrinsics"]  # (B, 3, 3)
```

`num_tokens` controls the encoder token budget. Use `"min"`, `"max"`, or an
integer value.

### CLI

Run inference on an image or a directory of images:

```
uv run --extra cli surge-cli path/to/image.jpg --output-dir output
```

By default, the CLI writes `mesh.glb` for each input image. Add output flags as
needed:

```
uv run --extra cli surge-cli path/to/images --save-maps --save-glb --save-ply
```

Useful options include `--max-size 1200`, `--tokens max`, `--fov-x 60`,
`--fp16`, and `--filter-sky`. For interactive viewing, use `--show-mesh` to
open the reconstructed mesh with trimesh, or `--rerun` to log the inference
results to a Rerun viewer.

### Gradio App

Launch the local demo app:

```
uv run --extra app surge-app
```

The app lets you upload an image, adjust the token budget and mesh cleanup, view
the reconstructed mesh, and download the generated maps and geometry.

### Output Conventions

`point_map.exr` and `point_normals_map.png` use OpenCV-style camera coordinates,
also known as RDF: `+X` right, `+Y` down, `+Z` forward. Exported `mesh.glb` and
`point_cloud.ply` are converted to RUB coordinates: `+X` right, `+Y` up, `+Z`
backward. This matches the usual right-handed, Y-up camera/view convention where
the camera looks along `-Z`, so points in front of the camera have negative `Z`.

`point_map.exr` stores float32 XYZ points. Masked pixels are written as `NaN`.

`point_normals_map.png` stores unit normals in the same RDF coordinates. The
red channel encodes `X` from `[-1, 1]` to `[0, 255]`; green encodes `-Y`; blue
encodes `-Z`. Invalid normal pixels are encoded as RGB `[127, 127, 127]`.
The sign flips make the visualization look like a conventional OpenGL-style
normal map. Renormalize after decoding if exact unit length is required.

## Neighborhood Attention Decoder (NAD) Module

The [NAD](src/surge/modules/heads/nad.py) is implemented as a reusable PyTorch module.
It is intentionally self-contained, so you can be copy it into your project as a single file without pulling in the rest of SurGe.

## ⚖️ License

The **SurGe code** is released under the MIT license, except for the **DINOv2 code** in `src/surge/_vendor/dinov2` which is released by Meta AI under the Apache 2.0 license.
The **SurGe weights** are released under CC BY-NC 4.0, due to the training datasets used.

## 🎓 Citation

If you use our work in your research, please use the following BibTeX entry.

```
@article{knaebel2026surge,
    title   = {{SurGe}: Improved Surface Geometry in Point Maps},
    author  = {Knaebel, Karim and Martin Garcia, Gonzalo and Schmidt, Christian and Fradlin, Ilya and Nunes, Lucas and de Geus, Daan and Leibe, Bastian},
    year    = {2026}
    journal = {arXiv preprint arXiv:2605.31577},
}
```
