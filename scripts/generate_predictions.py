from pathlib import Path
from typing import List
from os import environ
from argparse import ArgumentParser
from pprint import pprint

from MutComputeX.inference import EnsembleCIFPredictor


def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Pickle file with serialized microenvironments and snapshots",
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
    parser.add_argument(
        "--data",
        default=Path("../data/norbelladine_4OMTase/boxes/4OMTase_dataset.pkl"),
        type=Path,
        help="Pickle file with serialized microenvironments and snapshots",
    )
    parser.add_argument("--out-file", default=None, type=Path)
    parser.add_argument("--use-cpu", action="store_true")

    args = parser.parse_args()

    assert args.model_dir.is_dir()
    assert args.data.is_file()
    assert (
        args.data.suffix == ".pkl"
    ), f"{args.data.resolve()} must be a pickle file (.pkl)"

    return args


def generate_inference(models: List[Path], data: Path, out_file: Path = None):
    """
    Generate mutation inferences with:
    - tf trained models
    - serialized snapshot/boxes pickle file
    """

    predictor = EnsembleCIFPredictor(models, [0])

    predictor.predict(data)

    predictor.to_csv(out_file)


if __name__ == "__main__":
    args = cli()

    if args.use_cpu:
        environ["CUDA_VISIBLE_DEVICES"] = ""

    else:
        environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

    models = [
        m_dir.resolve()
        for m_dir in args.model_dir.glob(args.model_glob)
        if m_dir.is_dir()
    ]

    print(f"cli options:")
    pprint(vars(args))
    print(f"\nSelected model directories:")
    pprint(models)

    generate_inference(models, args.data, args.out_file)
