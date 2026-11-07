"""Bundled assets: pre-trained models, movement graphs and radar images.

Everything lives under ``poswing/assets`` so that a plain ``pip install`` ships a
usable Mirage model without touching the ESTA dataset.  Radar images and the
``map_data.json`` calibration come from awpy (MIT, see ``assets/radar/LICENSE-awpy.txt``).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def bundled_maps() -> list[str]:
    """Maps with a bundled win-probability model."""
    return sorted(p.stem for p in (ASSETS / "models").glob("de_*.json"))


def model_path(map_name: str) -> Path | None:
    p = ASSETS / "models" / f"{map_name}.json"
    return p if p.exists() else None


def graph_path(map_name: str) -> Path | None:
    p = ASSETS / "graphs" / f"{map_name}.json"
    return p if p.exists() else None


@lru_cache(maxsize=None)
def bundled_model(map_name: str):
    from .winprob import WinProbModel

    p = model_path(map_name)
    return WinProbModel.load(p) if p else None


@lru_cache(maxsize=None)
def bundled_graph(map_name: str):
    from .engine import TransitionGraph

    p = graph_path(map_name)
    return TransitionGraph.load(p) if p else None


@lru_cache(maxsize=None)
def radar_meta() -> dict[str, dict]:
    return json.loads((ASSETS / "radar" / "map_data.json").read_text())


def radar_path(map_name: str) -> Path | None:
    p = ASSETS / "radar" / f"{map_name}.png"
    return p if p.exists() else None


def radar_info(map_name: str) -> dict | None:
    """Calibration for world -> radar pixel: px = (x - pos_x) / scale, py = (pos_y - y) / scale."""
    meta = radar_meta().get(map_name)
    if meta is None or radar_path(map_name) is None:
        return None
    return {"map_name": map_name, "size": 1024, **meta}


def to_radar(map_name: str, x: float, y: float) -> tuple[float, float] | None:
    m = radar_meta().get(map_name)
    if m is None:
        return None
    return (x - m["pos_x"]) / m["scale"], (m["pos_y"] - y) / m["scale"]
