import argparse
import json
from pathlib import Path
from typing import Any, Literal

import torch
from moge.test.dataloader import EvalDataLoaderPipeline
from moge.utils.tools import key_average
from tqdm import tqdm

from surge import SurGe
from surge.inference import DEFAULT_CHECKPOINT

from .metrics import compute_metrics


def _parse_tokens(value: str) -> int | Literal["min", "max"]:
    if value in {"min", "max"}:
        return value
    return int(value)


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    for benchmark in config.values():
        benchmark["path"] = str(Path("data/eval") / benchmark["path"])
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SurGe on the MoGe benchmarks."
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="SurGe checkpoint or Hub ID."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("benchmarks.json"),
        help="Evaluation benchmark configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/surge.json"),
        help="Destination for aggregate JSON metrics.",
    )
    parser.add_argument(
        "--tokens",
        type=_parse_tokens,
        default="max",
        help='Inference token count, or "min"/"max".',
    )
    parser.add_argument("--device", default="cuda", help="Torch inference device.")
    parser.add_argument(
        "--fp16", action="store_true", help="Use float16 autocast for inference."
    )
    return parser.parse_args()


def _prediction(
    model: SurGe,
    image: torch.Tensor,
    num_tokens: int | Literal["min", "max"],
    fp16: bool,
) -> dict[str, torch.Tensor]:
    with torch.autocast(
        device_type=image.device.type,
        dtype=torch.float16,
        enabled=fp16,
    ):
        output = model.infer(image[None], num_tokens=num_tokens)
    return {
        "points_affine_invariant": output["points"][0].float(),
        "depth_affine_invariant": output["depth"][0].float(),
        "intrinsics": output["intrinsics"][0].float(),
    }


def evaluate(
    model: SurGe,
    config: dict[str, dict[str, Any]],
    output_path: Path,
    num_tokens: int | Literal["min", "max"] = "max",
    fp16: bool = False,
) -> dict:
    device = next(model.parameters()).device
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for benchmark_name, raw_config in tqdm(config.items(), desc="Benchmarks"):
        benchmark_config = raw_config.copy()
        has_clean_normal = benchmark_config.pop("has_clean_normal", False)
        metrics = []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as data,
            tqdm(total=len(data), desc=benchmark_name, leave=False) as progress,
        ):
            for index in range(len(data)):
                sample = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in data.get().items()
                }
                sample["has_clean_normal"] = has_clean_normal
                metrics.append(
                    compute_metrics(
                        _prediction(model, sample["image"], num_tokens, fp16), sample
                    )
                )
                if index % 100 == 0 or index == len(data) - 1:
                    output_path.write_text(
                        json.dumps(
                            {**all_metrics, benchmark_name: key_average(metrics)},
                            indent=2,
                        )
                        + "\n"
                    )
                progress.update()
        all_metrics[benchmark_name] = key_average(metrics)
    output_path.write_text(json.dumps(all_metrics, indent=2) + "\n")
    return all_metrics


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    model = SurGe.from_pretrained(args.checkpoint, strict=True).eval().to(args.device)
    evaluate(model, config, args.output, args.tokens, args.fp16)


if __name__ == "__main__":
    main()
