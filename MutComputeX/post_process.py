from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd

RESIDUES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


PREDICTION_AA_COL_NAMES = [
    "prALA",
    "prARG",
    "prASN",
    "prASP",
    "prCYS",
    "prGLN",
    "prGLU",
    "prGLY",
    "prHIS",
    "prILE",
    "prLEU",
    "prLYS",
    "prMET",
    "prPHE",
    "prPRO",
    "prSER",
    "prTHR",
    "prTRP",
    "prTYR",
    "prVAL",
]

PREDICTION_HEADER = [
    "pdb_id",
    "chain_id",
    "pos",
    "wtAA",
    "prAA",
    "wt_prob",
    "pred_prob",
    "avg_log_ratio",
    *PREDICTION_AA_COL_NAMES,
]


def prediction_df(
    snapshots: Iterable[dict], predictions: Iterable[Tuple[int]]
) -> pd.DataFrame:
    assert all([isinstance(ss, dict) for ss in snapshots]), type(list(snapshots)[0])

    res_dict = {r: i for i, r in enumerate(RESIDUES)}

    rows = []
    for s, p in zip(snapshots, predictions):
        idx = np.argmax(p)
        wt_aa = s["label"]
        wt_pr = p[res_dict[wt_aa]]
        pred_prob = p[idx]
        pr_aa = RESIDUES[idx]

        # Check if wt_pred is 0 so you dont divide by 0
        if wt_pr < np.finfo(float).eps:
            wt_pr = np.finfo(float).eps

        log_rat = np.log2(pred_prob / wt_pr)
        chain_id = s["chain_id"]
        pos = s["res_seq_num"]

        if s["type"] == "FILE_CHAIN_RESIDUE":
            source = s["filename"]

        else:
            source = ""

        row = [source, chain_id, pos, wt_aa, pr_aa, wt_pr, pred_prob, log_rat]
        row.extend([aa_prob for aa_prob in p])

        rows.append(row)

    return pd.DataFrame(rows, columns=PREDICTION_HEADER).sort_values(
        PREDICTION_HEADER[:3]
    )


def concat_predictions(DFs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(DFs, axis=0, ignore_index=True).sort_values(
        ["pdb_id", "chain_id", "pos", "wtAA"], ascending=[True, True, True, True]
    )


def find_predAA(row: pd.Series) -> Tuple[str, float]:
    assert isinstance(row, pd.Series)

    idx = np.argmax(row)
    pred_AA, pred_prob = row.index[idx].replace("pr", ""), row.iloc[idx]

    return pred_AA, pred_prob


def update_wt_prob(row: pd.Series) -> float:
    assert isinstance(row, pd.Series)

    wt_col = f"pr{row['wtAA']}"

    return row[wt_col]


def calc_log_odds(row: pd.Series) -> float:
    assert isinstance(row, pd.Series)
    assert "pred_prob" in row.index.tolist()
    assert "wt_prob" in row.index.tolist()

    return abs(round(np.log2(row["pred_prob"] / row["wt_prob"]), 4))


def average_predictions(prediction_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    df = concat_predictions(prediction_dfs)

    groupby_cols = ['pdb_id', 'chain_id', 'pos', 'wtAA']

    df = df[[*groupby_cols, *PREDICTION_AA_COL_NAMES]]

    avg_df = (
        df.groupby(groupby_cols, as_index=False)
        .mean()
        .round(6)
    )

    avg_df["wt_prob"] = avg_df.apply(update_wt_prob, axis=1)

    avg_df[["prAA", "pred_prob"]] = avg_df[PREDICTION_AA_COL_NAMES].apply(
        find_predAA, axis=1, result_type="expand"
    )

    avg_df["avg_log_ratio"] = avg_df[["wt_prob", "pred_prob"]].apply(
        calc_log_odds, axis=1
    )

    avg_df = avg_df[PREDICTION_HEADER]

    return avg_df


def prediction_accuracy(
    prediction_df: pd.DataFrame, wt_col: str = "wtAA", pr_col: str = "prAA"
) -> float:
    assert wt_col in prediction_df.columns
    assert pr_col in prediction_df.columns

    total = len(prediction_df)

    correct = 0
    for idx, row in prediction_df.iterrows():
        if row[wt_col] == row[pr_col]:
            correct += 1

    return round(correct / total, 5)
