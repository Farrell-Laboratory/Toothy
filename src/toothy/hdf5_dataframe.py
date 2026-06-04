"""Save and load pandas DataFrames in HDF5 files using h5py (no PyTables)."""

import pickle

import h5py
import numpy as np
import pandas as pd

_FORMAT = "pickle_v1"
_ATTR = "toothy_df_format"


def _normalize_key(key: str) -> str:
    return key.lstrip("/")


def save_df(h5_path, key: str, df: pd.DataFrame) -> None:
    key = _normalize_key(key)
    payload = np.frombuffer(
        pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8
    )
    with h5py.File(h5_path, "a") as f:
        if key in f:
            del f[key]
        ds = f.create_dataset(key, data=payload)
        ds.attrs[_ATTR] = _FORMAT


def load_df(h5_path, key: str) -> pd.DataFrame:
    key = _normalize_key(key)
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            raise KeyError(f"HDF5 key not found: {key}")
        node = f[key]
        if _ATTR in node.attrs and node.attrs[_ATTR] == _FORMAT:
            return pickle.loads(node[()].tobytes())
        raise ValueError(
            f"Key '{key}' is not a Toothy pickle DataFrame "
            "(legacy PyTables format is no longer supported)."
        )
