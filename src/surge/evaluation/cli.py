import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch
from moge.test.dataloader import EvalDataLoaderPipeline
from moge.utils.tools import key_average

from surge import SurGe
from surge.inference import DEFAULT_CHECKPOINT

from .metadata import collect_metadata
from .metrics import compute_metrics
from .progress import EvaluationProgress


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
    parser.add_argument(
        "--device",
        help="Torch inference device. Defaults to the PyTorch build's accelerator, or CPU.",
    )
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
    meta = collect_metadata(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    def write_output():
        output_path.write_text(
            json.dumps(all_metrics | {"_meta": meta}, indent=2) + "\n"
        )

    write_output()
    with EvaluationProgress(len(config)) as progress:
        for benchmark_name, raw_config in config.items():
            benchmark_config = raw_config.copy()
            has_clean_normal = benchmark_config.pop("has_clean_normal", False)
            metrics = []
            with progress.dataset(benchmark_name) as task:
                with EvalDataLoaderPipeline(**benchmark_config) as data:
                    progress.start_samples(task, len(data))
                    for _ in range(len(data)):
                        sample = {
                            key: value.to(device)
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in data.get().items()
                        }
                        sample["has_clean_normal"] = has_clean_normal
                        if device.type != "cpu":
                            torch.accelerator.synchronize(device)
                        start = perf_counter()
                        prediction = _prediction(
                            model, sample["image"], num_tokens, fp16
                        )
                        if device.type != "cpu":
                            torch.accelerator.synchronize(device)
                        inference_s = perf_counter() - start
                        sample_metrics = compute_metrics(prediction, sample)
                        sample_metrics["runtime"] = {"inference_s": inference_s}
                        metrics.append(sample_metrics)
                        progress.advance_sample(task)
                all_metrics[benchmark_name] = key_average(metrics)
                write_output()
        meta["finished_at"] = (
            datetime.now().astimezone().replace(microsecond=0).isoformat()
        )
        write_output()
    return all_metrics


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.accelerator.current_accelerator() or torch.device("cpu")
    )
    model = SurGe.from_pretrained(args.checkpoint, strict=True).eval().to(device)
    evaluate(model, config, args.output, args.tokens, args.fp16)


if __name__ == "__main__":
    main()
