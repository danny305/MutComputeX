from subprocess import run
from argparse import ArgumentParser
from pathlib import Path
import os


def cli():
    cwd = os.getcwd()

    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        default=Path(f"{cwd}/../data/norbelladine_4OMTase/boxes/4OMTase_dataset.pkl"),
        type=Path,
    )
    parser.add_argument(
        "--model-dir",
        default=Path("../models"),
        type=Path,
        help="directory where model checkpoints are located",
    )
    parser.add_argument(
        "--model-glob",
        type=str,
        default="*",
        help="glob to select specific models/directories in the model-dir folder",
    )
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


def main(args):
    cmd = f"docker run -v {args.data.resolve()}:/input/input_file.pkl "

    cmd += f"-v {args.model_dir}:/models "

    if args.out_file:
        cmd += f"-v {args.out_file.parent.resolve()}:/output "

    cmd += (
        f"-t mutcomputex:latest python generate_predictions.py "
        "--data /input/input_file.pkl ",
        "--model-dir /models",
    )

    if args.out_file:
        cmd += f"--out-file /output/{args.out_file.name} "

    if args.use_cpu:
        cmd += "--use-cpu"

    run(cmd, shell=True)


if __name__ == "__main__":
    args = cli()
    main(args)
