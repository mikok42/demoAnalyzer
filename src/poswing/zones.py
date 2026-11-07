"""Map zoning: assign every position sample to a named section of the map.

Three strategies, in order of preference:

* ``place``   – the nav-mesh place name the game itself stores on the player
                (``last_place_name``: "BombsiteA", "TSpawn", "Connector", ...).
                Zero configuration and already "sections which make sense".
* ``custom``  – user supplied polygons (JSON: ``{"zones": [{"name", "polygon": [[x, y], ...]}]}``)
                for finer or coarser callouts.  Falls back to ``place`` for points
                outside every polygon.
* ``grid``    – square cells of ``cell`` units, for maps without place names.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

UNKNOWN = "Unknown"


def humanize(place: str) -> str:
    """'BombsiteA' -> 'Bombsite A', 'TopofMid' -> 'Top of Mid'."""
    if not isinstance(place, str) or not place:
        return UNKNOWN
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", place)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = s.replace("Bombsite A", "A Site").replace("Bombsite B", "B Site")
    s = s.replace("Topof", "Top of").replace("Backof", "Back of")
    return s.strip()


@dataclass
class Zoner:
    strategy: str = "place"  # place | custom | grid
    cell: float = 512.0
    polygons: list[tuple[str, np.ndarray]] = field(default_factory=list)
    merge: dict[str, str] = field(default_factory=dict)  # raw zone -> merged zone

    @classmethod
    def from_args(cls, strategy: str = "place", cell: float = 512.0, custom: str | Path | None = None,
                  merge: dict[str, str] | None = None) -> "Zoner":
        z = cls(strategy=strategy, cell=cell, merge=merge or {})
        if custom:
            z.load_custom(custom)
            z.strategy = "custom"
        return z

    def load_custom(self, path: str | Path) -> None:
        spec = json.loads(Path(path).read_text())
        self.polygons = []
        for zone in spec.get("zones", []):
            poly = np.asarray(zone["polygon"], dtype=float)
            if poly.ndim != 2 or poly.shape[0] < 3:
                raise ValueError(f"zone {zone.get('name')} needs at least 3 vertices")
            self.polygons.append((zone["name"], poly))
        self.merge.update(spec.get("merge", {}))

    # ------------------------------------------------------------------ core
    def assign_one(self, place: str | None, x: float | None = None, y: float | None = None) -> str:
        if self.strategy == "place" or x is None or y is None or pd.isna(x) or pd.isna(y):
            label = humanize(place) if isinstance(place, str) and place.strip() else UNKNOWN
            return self.merge.get(label, label)
        df = pd.DataFrame({"place": [place], "x": [float(x)], "y": [float(y)]})
        return str(self.assign(df).iloc[0])

    def assign(self, samples: pd.DataFrame) -> pd.Series:
        """Return a zone label per sample row."""
        if self.strategy == "grid":
            labels = self._grid(samples)
        elif self.strategy == "custom":
            labels = self._custom(samples)
        else:
            labels = self._place(samples)
        if self.merge:
            labels = labels.map(lambda z: self.merge.get(z, z))
        return labels

    def _place(self, samples: pd.DataFrame) -> pd.Series:
        if "place" not in samples:
            return self._grid(samples)
        place = samples["place"].astype("object")
        missing = place.isna() | (place.astype(str).str.strip() == "")
        labels = place.where(~missing, None).map(lambda p: humanize(p) if p else UNKNOWN)
        if missing.all() and len(samples):
            return self._grid(samples)
        return labels.astype(str)

    def _grid(self, samples: pd.DataFrame) -> pd.Series:
        gx = np.floor(samples["x"].to_numpy(dtype=float) / self.cell).astype(int)
        gy = np.floor(samples["y"].to_numpy(dtype=float) / self.cell).astype(int)
        return pd.Series([f"cell_{a}_{b}" for a, b in zip(gx, gy)], index=samples.index, dtype=str)

    def _custom(self, samples: pd.DataFrame) -> pd.Series:
        xs = samples["x"].to_numpy(dtype=float)
        ys = samples["y"].to_numpy(dtype=float)
        out = np.full(len(samples), None, dtype=object)
        for name, poly in self.polygons:
            inside = _points_in_polygon(xs, ys, poly) & pd.isna(out)
            out[inside] = name
        labels = pd.Series(out, index=samples.index, dtype=object)
        fallback = self._place(samples)
        return labels.where(labels.notna(), fallback).astype(str)


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting."""
    n = len(poly)
    inside = np.zeros(len(xs), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > ys) != (yj > ys)) & (xs < (xj - xi) * (ys - yi) / ((yj - yi) or 1e-12) + xi)
        inside ^= cond
        j = i
    return inside
