import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
import utils3d
from tqdm import tqdm

import surge.utils.vis
from surge.inference import (
    DEFAULT_CHECKPOINT,
    IMAGE_SUFFIXES,
    RDF_RUB_SCALE,
    TRIMESH_UV_OFFSET,
    TRIMESH_UV_SCALE,
    build_mesh,
    infer_image,
    point_cloud_from_result,
    resize_to_max_size,
)
from surge.utils.background import apply_background_filters
from surge import SurGe
from surge.utils.io import read_image, save_glb, save_ply

try:
    import rerun as rr
except ImportError:
    rr = None

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"


def parse_tokens(value: str) -> int | str:
    if value in {"min", "max"}:
        return value
    return int(value)


def collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return [input_path]


def output_dir_for(image_path: Path, input_path: Path, output_path: Path) -> Path:
    if input_path.is_dir():
        return output_path / image_path.relative_to(input_path).parent / image_path.stem
    return output_path / image_path.stem


def save_maps(save_path: Path, result: dict[str, np.ndarray]) -> None:
    image = result["image"]
    points = result["points"]
    depth = result["depth"]
    mask = result["mask"]
    normal_mask = mask & result["normal_mask"]
    normal = result["normal"]

    cv2.imwrite(str(save_path / "image.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(save_path / "depth_map_colorized.png"),
        cv2.cvtColor(surge.utils.vis.colorize_depth(depth, mask), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(save_path / "point_map.exr"),
        cv2.cvtColor(
            np.where(mask[..., None], points, np.nan).astype(np.float32),
            cv2.COLOR_RGB2BGR,
        ),
        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
    )
    cv2.imwrite(
        str(save_path / "point_normals_map.png"),
        cv2.cvtColor(
            surge.utils.vis.colorize_normal(normal, normal_mask),
            cv2.COLOR_RGB2BGR,
        ),
    )

    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(result["intrinsics"])
    (save_path / "fov.json").write_text(
        json.dumps(
            {
                "fov_x": round(float(np.rad2deg(fov_x)), 2),
                "fov_y": round(float(np.rad2deg(fov_y)), 2),
            },
            indent=2,
        )
        + "\n"
    )


def log_rerun(
    result: dict[str, np.ndarray],
    trimesh_vertices: np.ndarray,
    faces: np.ndarray,
    trimesh_vertex_uvs: np.ndarray,
    trimesh_vertex_normals: np.ndarray,
    image_path: Path,
    index: int,
) -> None:
    if rr is None:
        raise RuntimeError("rerun is not installed")

    rr.set_time("index", sequence=index)
    image = result["image"]
    points = result["points"]
    depth = result["depth"]
    normal = result["normal"]
    mask = result["mask"]
    normal_mask = mask & result["normal_mask"]
    normal_colors = surge.utils.vis.colorize_normal(normal, normal_mask)

    rr.log("metadata/input", rr.TextDocument(str(image_path)))
    rr.log("image/rgb", rr.Image(image))
    rr.log(
        "world/points",
        rr.Points3D(points[mask], colors=image[mask]),
    )
    rr.log(
        "image/depth",
        rr.DepthImage(np.where(mask, depth, np.nan).astype(np.float32)),
    )

    rr.log(
        "world/normals",
        rr.Points3D(points[normal_mask], colors=normal_colors[normal_mask]),
    )
    rr.log("image/normals", rr.Image(normal_colors))

    rr.log(
        "world/mesh",
        rr.Mesh3D(
            vertex_positions=trimesh_vertices * RDF_RUB_SCALE,
            triangle_indices=faces,
            vertex_normals=trimesh_vertex_normals * RDF_RUB_SCALE,
            vertex_texcoords=(trimesh_vertex_uvs - TRIMESH_UV_OFFSET)
            * TRIMESH_UV_SCALE,
            albedo_texture=image,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SurGe inference.")
    parser.add_argument(
        "input",
        type=Path,
        help="Input image or directory. JPG, JPEG, and PNG are supported.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to a SurGe checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("./output"),
        type=Path,
        help="Output directory for maps and exported geometry.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Torch device, for example "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--fov-x",
        dest="horizontal_fov",
        type=float,
        default=None,
        help="Known horizontal field of view in degrees. If omitted, SurGe estimates it.",
    )
    parser.add_argument(
        "--tokens",
        type=parse_tokens,
        default="max",
        help='Number of inference tokens, or "min"/"max".',
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Resize the input so its larger side matches this size before inference.",
    )
    parser.add_argument(
        "--native-output",
        action="store_true",
        help="Keep the model-native output resolution instead of resizing to input size.",
    )
    parser.add_argument(
        "--no-reproject",
        action="store_true",
        help="Keep the raw point map instead of reprojecting depth through the camera intrinsics.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 autocast for inference.",
    )
    parser.add_argument(
        "--unit-scale",
        action="store_true",
        help="Scale points and depth so the mean point distance is 1.",
    )
    parser.add_argument(
        "--mesh-edge-cutoff",
        type=float,
        default=0.01,
        help="Mesh edge cleanup. Lower values apply more aggressive cleanup around depth jumps.",
    )
    parser.add_argument(
        "--filter-sky",
        action="store_true",
        help="Remove sky regions from exported masks and geometry.",
    )
    parser.add_argument(
        "--filter-black-background",
        action="store_true",
        help="Remove near-black background regions from exported masks and geometry.",
    )
    parser.add_argument(
        "--filter-white-background",
        action="store_true",
        help="Remove near-white background regions from exported masks and geometry.",
    )
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Save image/depth/points/normal/fov maps.",
    )
    parser.add_argument(
        "--save-glb",
        action="store_true",
        help="Save a textured mesh.glb. Default when no output/view mode is selected.",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Save a vertex-colored point_cloud.ply.",
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Show the reconstructed mesh with trimesh.",
    )
    rerun_mode = parser.add_mutually_exclusive_group()
    rerun_mode.add_argument(
        "--rerun",
        action="store_true",
        help="Open a Rerun viewer and log inference results.",
    )
    rerun_mode.add_argument(
        "--rerun-connect",
        nargs="?",
        const="",
        default=None,
        metavar="URL",
        help="Log to a running Rerun viewer, optionally at URL.",
    )
    rerun_mode.add_argument(
        "--rerun-serve",
        action="store_true",
        help="Serve a Rerun web viewer.",
    )
    rerun_mode.add_argument(
        "--rerun-save",
        type=Path,
        metavar="PATH",
        help="Save a Rerun recording to PATH.",
    )
    rerun_mode.add_argument(
        "--rerun-stdout",
        action="store_true",
        help="Write Rerun data to stdout.",
    )
    return parser.parse_args()


def wants_rerun(args: argparse.Namespace) -> bool:
    return any(
        [
            args.rerun,
            args.rerun_connect is not None,
            args.rerun_serve,
            args.rerun_save is not None,
            args.rerun_stdout,
        ]
    )


def setup_rerun(args: argparse.Namespace) -> None:
    if rr is None:
        raise RuntimeError("rerun is not installed")

    rr.init("surge_infer", default_enabled=True, strict=True)
    recording = rr.get_global_data_recording()
    if args.rerun_stdout:
        recording.stdout()
    elif args.rerun_serve:
        rr.serve_web_viewer(open_browser=True, connect_to=recording.serve_grpc())
    elif args.rerun_connect is not None:
        recording.connect_grpc(args.rerun_connect or None)
    elif args.rerun_save is not None:
        recording.save(args.rerun_save)
    else:
        recording.spawn()


def main() -> None:
    args = parse_args()
    args.rerun = wants_rerun(args)
    if not any(
        [args.save_maps, args.save_glb, args.save_ply, args.show_mesh, args.rerun]
    ):
        args.save_glb = True

    input_path = args.input
    image_paths = collect_image_paths(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {input_path}")

    if args.rerun:
        setup_rerun(args)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log(
            "metadata/args",
            rr.TextDocument(
                "\n\n".join(f"**{k}**: {v}" for k, v in vars(args).items()),
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )

    device = torch.device(args.device)
    model = SurGe.from_pretrained(args.checkpoint).eval().to(device)

    for index, image_path in enumerate(
        tqdm(image_paths, desc="Inference", unit="image")
    ):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.fp16,
        ):
            result = infer_image(
                model=model,
                image=resize_to_max_size(read_image(image_path), args.max_size),
                device=device,
                num_tokens=args.tokens,
                resize_output=not args.native_output,
                force_projection=not args.no_reproject,
                fov_x=args.horizontal_fov,
                unit_scale=args.unit_scale,
            )
        result = apply_background_filters(
            result,
            args.filter_sky,
            args.filter_black_background,
            args.filter_white_background,
        )

        if args.save_maps or args.save_glb or args.save_ply:
            save_path = output_dir_for(image_path, input_path, args.output_dir)
            save_path.mkdir(exist_ok=True, parents=True)

        if args.save_maps:
            save_maps(save_path, result)

        if args.save_glb or args.show_mesh or args.rerun:
            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = build_mesh(
                result,
                args.mesh_edge_cutoff,
            )

        if args.rerun:
            log_rerun(
                result,
                vertices,
                faces,
                vertex_uvs,
                vertex_normals,
                image_path,
                index,
            )

        if args.save_glb:
            save_glb(
                save_path / "mesh.glb",
                vertices,
                faces,
                vertex_uvs,
                result["image"],
                vertex_normals,
            )

        if args.save_ply:
            point_cloud_vertices, point_cloud_colors, point_cloud_normals = (
                point_cloud_from_result(result)
            )
            save_ply(
                save_path / "point_cloud.ply",
                point_cloud_vertices,
                np.zeros((0, 3), dtype=np.int32),
                point_cloud_colors,
                point_cloud_normals,
            )

        if args.show_mesh:
            import trimesh

            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                vertex_normals=vertex_normals,
                faces=faces,
                process=False,
            ).show()


if __name__ == "__main__":
    main()
