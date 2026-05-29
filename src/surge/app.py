import os
import tempfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import gradio as gr
import numpy as np
import spaces
import torch

import surge.utils.vis
from surge.inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKEN_RANGE,
    build_mesh,
    infer_image,
    point_cloud_from_result,
    resize_to_max_size,
)
from surge import SurGe
from surge.utils.background import apply_background_filters
from surge.utils.io import save_glb, save_ply

print(
    f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda or 'none'}, "
    f"CUDA available {torch.cuda.is_available()}"
)

MAX_INPUT_SIZE = 1200
MIN_TOKENS, MAX_TOKENS = DEFAULT_TOKEN_RANGE
EXAMPLE_DIR = Path("example_images")
SKY_FILTER_EXAMPLES = {
    "buddah.jpg",
    "norway.jpg",
    "swiss.jpg",
    "london_street.jpg",
    "umic_building.jpg",
}
EDGE_CUTOFF_EXAMPLES = {
    "chairs.jpg": 0.003,
    "oranges.jpg": 0.002,
}
EXAMPLES = [
    [
        str(path),
        MAX_TOKENS,
        EDGE_CUTOFF_EXAMPLES.get(path.name, 0.01),
        True,
        path.name in SKY_FILTER_EXAMPLES,
        False,
        False,
    ]
    for path in sorted(EXAMPLE_DIR.iterdir() if EXAMPLE_DIR.exists() else [])
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = SurGe.from_pretrained(DEFAULT_CHECKPOINT).eval().to(DEVICE)
MESH_VIEWER_CSS = """
#mesh-viewer .model3D {
    background: radial-gradient(circle at center, #20242d 0, #111318 65%);
}

#mesh-viewer .mesh-loading-overlay {
    align-items: center;
    background: rgba(8, 10, 14, 0.72);
    color: white;
    display: none;
    flex-direction: column;
    font-size: 14px;
    gap: 10px;
    inset: 0;
    justify-content: center;
    pointer-events: none;
    position: absolute;
    z-index: 2;
}

#mesh-viewer .mesh-loading-overlay.is-visible {
    display: flex;
}

#mesh-viewer .mesh-loading-spinner {
    animation: mesh-loading-spin 0.9s linear infinite;
    border: 3px solid rgba(255, 255, 255, 0.28);
    border-top-color: white;
    border-radius: 50%;
    height: 34px;
    width: 34px;
}

@keyframes mesh-loading-spin {
    to {
        transform: rotate(360deg);
    }
}
"""
MESH_VIEWER_JS = """
(() => {
    const viewerId = "mesh-viewer";
    let activeLoads = 0;

    function overlay() {
        const root = document.getElementById(viewerId);
        if (!root) {
            return null;
        }
        const target = root.querySelector(".model3D") || root;
        let element = target.querySelector(".mesh-loading-overlay");
        if (!element) {
            element = document.createElement("div");
            element.className = "mesh-loading-overlay";
            element.innerHTML = `
                <div class="mesh-loading-spinner"></div>
                <div class="mesh-loading-status">Downloading mesh...</div>
            `;
            target.appendChild(element);
        }
        return element;
    }

    function setStatus(message) {
        const element = overlay();
        if (!element) {
            return;
        }
        element.querySelector(".mesh-loading-status").textContent = message;
        element.classList.add("is-visible");
    }

    const originalOpen = XMLHttpRequest.prototype.open;
    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function patchedOpen(method, url, ...args) {
        this.__surgeMeshUrl = String(url).toLowerCase().includes("mesh.glb");
        return originalOpen.call(this, method, url, ...args);
    };
    XMLHttpRequest.prototype.send = function patchedSend(...args) {
        if (this.__surgeMeshUrl) {
            activeLoads += 1;
            setStatus("Downloading mesh...");
            this.addEventListener("progress", (event) => {
                if (event.lengthComputable) {
                    setStatus(`Downloading mesh... ${Math.round(event.loaded / event.total * 100)}%`);
                }
            });
            this.addEventListener("loadend", () => {
                activeLoads = Math.max(0, activeLoads - 1);
                if (activeLoads === 0) {
                    overlay()?.classList.remove("is-visible");
                }
            }, { once: true });
        }
        return originalSend.apply(this, args);
    };
})();
"""


def _resize_for_inference(image: np.ndarray) -> np.ndarray:
    if max(image.shape[:2]) <= MAX_INPUT_SIZE:
        return image
    return resize_to_max_size(image, MAX_INPUT_SIZE)


def _rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        return image[..., :3]
    return image


def _save_points_exr(path: Path, points: np.ndarray, mask: np.ndarray) -> None:
    cv2.imwrite(
        str(path),
        cv2.cvtColor(
            np.where(mask[..., None], points, np.nan).astype(np.float32),
            cv2.COLOR_RGB2BGR,
        ),
        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
    )


def _save_png(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@spaces.GPU(duration=5)  # it's really more like <<1s, but just to be safe
def _inference_result(
    image: np.ndarray,
    inference_tokens: int,
    resize_outputs_to_input_size: bool,
) -> dict[str, np.ndarray]:
    return infer_image(
        MODEL,
        image,
        DEVICE,
        num_tokens=inference_tokens,
        resize_output=resize_outputs_to_input_size,
        force_projection=True,
        fov_x=None,
    )


def run_inference(
    input_image: np.ndarray,
    inference_tokens: int,
    mesh_edge_cutoff: float,
    resize_outputs_to_input_size: bool,
    filter_sky: bool,
    filter_black_background: bool,
    filter_white_background: bool,
) -> tuple[str, str, str, str, str, str, str]:
    if input_image is None:
        raise gr.Error("Upload an image first.")

    image = _resize_for_inference(_rgb_image(input_image))
    result = _inference_result(
        image,
        inference_tokens,
        resize_outputs_to_input_size,
    )
    result = apply_background_filters(
        result,
        filter_sky,
        filter_black_background,
        filter_white_background,
    )
    faces, vertices, vertex_colors, vertex_uvs, vertex_normals = build_mesh(
        result,
        edge_threshold=mesh_edge_cutoff,
    )
    point_cloud_vertices, point_cloud_colors, point_cloud_normals = (
        point_cloud_from_result(result)
    )

    output_dir = Path(tempfile.mkdtemp(prefix="surge-gradio-"))
    mesh_path = output_dir / "mesh.glb"
    color_point_cloud_path = output_dir / "point_cloud.ply"
    point_map_path = output_dir / "point_map.exr"
    image_path = output_dir / "image.png"
    depth_map_path = output_dir / "depth_map_colorized.png"
    point_normals_map_path = output_dir / "point_normals_map.png"

    save_glb(mesh_path, vertices, faces, vertex_uvs, result["image"], vertex_normals)
    save_ply(
        color_point_cloud_path,
        point_cloud_vertices,
        np.zeros((0, 3), dtype=np.int32),
        point_cloud_colors,
        point_cloud_normals,
    )
    _save_points_exr(point_map_path, result["points"], result["mask"])

    depth = surge.utils.vis.colorize_depth(result["depth"], result["mask"])
    point_normals_map = surge.utils.vis.colorize_normal(
        result["normal"],
        result["mask"] & result["normal_mask"],
    )
    _save_png(image_path, result["image"])
    _save_png(depth_map_path, depth)
    _save_png(point_normals_map_path, point_normals_map)
    return (
        str(mesh_path),
        str(mesh_path),
        str(color_point_cloud_path),
        str(point_map_path),
        str(image_path),
        str(depth_map_path),
        str(point_normals_map_path),
    )


with gr.Blocks(title="SurGe") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                "# 3D reconstruction with SurGe\n"
                "[GitHub](https://github.com/karimknaebel/surge) · "
                "[Project page](https://vision.rwth-aachen.de/surge)"
            )
            input_image = gr.Image(
                label="Input image",
                type="numpy",
                image_mode="RGB",
            )
            inference_tokens = gr.Slider(
                MIN_TOKENS,
                MAX_TOKENS,
                value=MAX_TOKENS,
                step=1,
                label="Inference tokens",
                info="Encoder token budget. Higher values preserve more detail.",
            )
            mesh_edge_cutoff = gr.Slider(
                0.001,
                0.05,
                value=0.01,
                step=0.001,
                label="Mesh edge cleanup",
                info="Lower values apply more aggressive cleanup around depth jumps.",
            )
            resize_outputs_to_input_size = gr.Checkbox(
                value=True,
                label="Resize outputs to input size",
            )
            filter_sky = gr.Checkbox(
                value=False,
                label="Filter sky",
            )
            filter_black_background = gr.Checkbox(
                value=False,
                label="Filter black background",
            )
            filter_white_background = gr.Checkbox(
                value=False,
                label="Filter white background",
            )
            reconstruct = gr.Button("Reconstruct", variant="primary")
        with gr.Column(scale=2):
            mesh_viewer = gr.Model3D(
                label="Mesh",
                height=620,
                zoom_speed=0.5,
                pan_speed=0.5,
                clear_color=(0.0, 0.0, 0.0, 0.0),
                elem_id="mesh-viewer",
            )
            with gr.Row():
                depth_map_preview = gr.Image(
                    label="Depth map (colorized)",
                    type="filepath",
                )
                point_normals_map_preview = gr.Image(
                    label="Point normals map",
                    type="filepath",
                )
            gr.Markdown("### Downloads")
            with gr.Row():
                mesh_download = gr.DownloadButton("mesh.glb")
                point_cloud_download = gr.DownloadButton("point_cloud.ply")
                point_map_download = gr.DownloadButton("point_map.exr")
                image_download = gr.DownloadButton("image.png")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[
            input_image,
            inference_tokens,
            mesh_edge_cutoff,
            resize_outputs_to_input_size,
            filter_sky,
            filter_black_background,
            filter_white_background,
        ],
        outputs=[
            mesh_viewer,
            mesh_download,
            point_cloud_download,
            point_map_download,
            image_download,
            depth_map_preview,
            point_normals_map_preview,
        ],
        fn=run_inference,
        cache_examples=True,
        cache_mode="lazy",
        examples_per_page=20,
        label="Examples",
    )

    reconstruct.click(
        run_inference,
        inputs=[
            input_image,
            inference_tokens,
            mesh_edge_cutoff,
            resize_outputs_to_input_size,
            filter_sky,
            filter_black_background,
            filter_white_background,
        ],
        outputs=[
            mesh_viewer,
            mesh_download,
            point_cloud_download,
            point_map_download,
            image_download,
            depth_map_preview,
            point_normals_map_preview,
        ],
    )


def main():
    demo.launch(css=MESH_VIEWER_CSS, js=MESH_VIEWER_JS)


if __name__ == "__main__":
    main()
