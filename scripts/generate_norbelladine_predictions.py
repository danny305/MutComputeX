from pathlib import Path
from os import environ
from pprint import pprint

from generate_predictions import generate_inference


if __name__ == "__main__":
    model_dir = Path("../models")
    data = Path("../data/norbelladine_4OMTase/boxes/4OMTase_dataset.pkl")
    out_file = None

    model_glob = "*"
    use_cpu = True

    if use_cpu:
        environ["CUDA_VISIBLE_DEVICES"] = ""

    else:
        environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

    models = [m_dir.resolve() for m_dir in model_dir.glob(model_glob) if m_dir.is_dir()]

    print(f"\nSelected model directories:")
    pprint(models)

    generate_inference(models, data, out_file)
