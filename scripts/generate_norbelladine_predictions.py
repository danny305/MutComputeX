from pathlib import Path
from typing import List
from os import environ

from MutComputeX.inference import EnsembleCIFPredictor


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
    environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    model_dir = Path("../models/")

    models = [m_dir for m_dir in model_dir.iterdir() if m_dir.is_dir()]

    data = Path("../data/norbelladine_4OMTase/boxes/4OMTase_dataset.pkl")

    generate_inference(models, data)
