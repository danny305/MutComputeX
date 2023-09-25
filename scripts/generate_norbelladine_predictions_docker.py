from subprocess import run
from argparse import ArgumentParser
from pathlib import Path
import os


def get_args():
    cwd = os.getcwd()

    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        default=Path(f"{cwd}/../data/norbelladine_4OMTase/boxes/4OMTase_dataset.pkl"),
        type=Path,
    )
    parser.add_argument("--out-file", default=None)
    return parser.parse_args()


def main(data: Path, out_file: Path):

    command = f"docker run -v {data.resolve()}:/input/input_file.pkl "
    if out_file:
        command += f"-v {out_file.parent.resolve}:/ouptut "

    command += (
        f"-t mutcomputex:latest python generate_norbelladine_predictions.py "
        "--data /input/input_file.pkl "
    )

    if out_file:
        command += f"--out-file /output/{out_file.name}"

    run(command, shell=True)


if __name__ == "__main__":
    main(**vars(get_args()))
