from __future__ import annotations

import argparse
from pathlib import Path

from .coco_data import download_coco_val2017
from .config import load_config
from .experiment import run_benchmark
from .plotting import create_plots
from .preview import create_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topic 6 object detection robustness benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download official COCO val2017 data")
    download.add_argument("--root", default="data/coco", help="Dataset destination")
    download.add_argument("--keep-archives", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Run the robustness benchmark")
    benchmark.add_argument("--config", default="configs/coco_val2017.yaml")

    plot = subparsers.add_parser("plot", help="Create plots from metrics.csv")
    plot.add_argument("--metrics", required=True)
    plot.add_argument("--output-dir")

    preview = subparsers.add_parser("preview", help="Render clean, corrupted, and recovered examples")
    preview.add_argument("--config", default="configs/smoke_test.yaml")
    preview.add_argument("--severity", type=int, default=3)
    preview.add_argument("--image-id", type=int)
    preview.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        images, annotations = download_coco_val2017(args.root, keep_archives=args.keep_archives)
        print(f"Images: {images}")
        print(f"Annotations: {annotations}")
    elif args.command == "benchmark":
        config = load_config(args.config)
        metrics = run_benchmark(config)
        print(f"Metrics: {metrics}")
    elif args.command == "plot":
        outputs = create_plots(args.metrics, args.output_dir)
        for output in outputs:
            print(output)
    elif args.command == "preview":
        config = load_config(args.config)
        output = create_preview(config, args.severity, args.image_id, args.output)
        print(output)


if __name__ == "__main__":
    main()

