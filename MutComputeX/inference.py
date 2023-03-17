from pathlib import Path
from typing import Set, List

import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow.keras as K
from tensorflow.keras.models import Model

from MutComputeX.data_loader import load_dataset
from MutComputeX.post_process import (
    prediction_df,
    average_predictions,
    prediction_accuracy,
)


def load_model(model_dir: Path) -> Model:
    assert isinstance(model_dir, Path)
    assert model_dir.is_dir()

    return K.models.load_model(str(model_dir))


def load_ensemble_models(model_dirs: List[Path]) -> List[Model]:
    assert all([isinstance(m_dir, Path) and m_dir.is_dir() for m_dir in model_dirs])

    ens_model = [load_model(m_dir) for m_dir in model_dirs]

    return ens_model


class EnsembleCIFPredictor:
    def __init__(
        self,
        model_dirs: List[Path],
        gpu_idx: Set[int] = [0],
        ensemble_name: str = "EnsResNet",
    ):
        assert all([isinstance(model_dir, Path) for model_dir in model_dirs])
        for model in model_dirs:
            assert model.is_dir(), model.resolve()
        assert all([model_dir.is_dir() for model_dir in model_dirs])

        self.gpu_idx = sorted(set(gpu_idx))
        self.gpu_names = [f"/GPU:{idx}" for idx in self.gpu_idx]
        self.strategy = tf.distribute.MirroredStrategy(devices=self.gpu_names)

        self.models = []
        for model_dir in model_dirs:
            with self.strategy.scope():
                model = load_model(model_dir)
                model.model_name = f"{model_dir.stem}"

            self.models.append(model)

        self.model_dirs = model_dirs
        self.model_name = ensemble_name

        self.snapshots = None
        self.predictions = None


    def predict(self, pkl_data: Path) -> pd.DataFrame:
        assert isinstance(pkl_data, Path)
        assert pkl_data.is_file()
        assert pkl_data.suffix == ".pkl"

        self.dataset = pkl_data
        snapshots, boxes = load_dataset(pkl_data)

        self.model_predictions = {}
        for model in self.models:
            print(f"Generating predictions with model - {model.model_name}")

            with self.strategy.scope():
                predictions = prediction_df(snapshots, model.predict(boxes, verbose=1))
                self.model_predictions[model.model_name] = predictions

        self.predictions = average_predictions(self.model_predictions.values())

        return self.predictions


    def to_csv(self, out_file: Path = None, include_model_name=True) -> Path:
        assert self.predictions is not None
        assert self.dataset is not None

        predictions = self.predictions.copy()

        accuracy = prediction_accuracy(predictions)

        if out_file is None:
            out_dir = self.dataset.parent.parent / "predictions"
            out_dir.mkdir(0o770, parents=True, exist_ok=True)
            out_file = out_dir / f"{self.dataset.stem}_predictions.csv"

        out_file = out_file.with_suffix(".csv")

        if include_model_name:
            predictions = predictions.assign(model=self.model_name, accuracy=accuracy)
            model_col = predictions.pop("model")
            accuracy_col = predictions.pop("accuracy")
            predictions.insert(0, "model", model_col)
            predictions.insert(1, "accuracy", accuracy_col)

        else:
            predictions = predictions.assign(accuracy=accuracy)
            accuracy_col = predictions.pop("accuracy")
            predictions.insert(0, "accuracy", accuracy_col)

        predictions.to_csv(out_file, index=False)

        print(f"Wrote predictions: {out_file.resolve()}")

        return out_file
