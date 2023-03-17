from pathlib import Path
from typing import List, Dict, Tuple, Union
import pickle as pkl

import numpy as np




def load_dataset(pkl_file: Path) -> Tuple[List[Dict[str,Union[str,int]]], np.ndarray]:
    """Snapshots provide the residue order of the boxes"""
    
    assert isinstance(pkl_file, Path)
    assert pkl_file.is_file(), pkl_file.resolve()
    assert pkl_file.suffix == ".pkl", pkl_file.suffix

    with pkl_file.open("rb") as f:
        protein_data = pkl.load(f)

    assert "snapshots" in protein_data.keys(), protein_data.keys()
    assert "boxes" in protein_data.keys(), protein_data.keys()

    snapshots = protein_data["snapshots"]
    boxes = protein_data["boxes"]

    print(f"Loaded data: {boxes.shape}")

    return snapshots, boxes