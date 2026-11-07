"""Normalised match representation produced by the parser (or the synthetic generator).

Everything downstream (zoning, win-probability, swing, rating, analysis) works on
``MatchData`` so that the demo format is isolated to ``parse.py``.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

FORMAT_VERSION = 1

ROUND_COLUMNS = [
    "round_num", "start_tick", "freeze_end_tick", "end_tick", "winner", "reason",
    "plant_tick", "plant_site", "t_team", "ct_team", "t_equip", "ct_equip",
    "t_buy", "ct_buy", "ct_has_kit",
]
KILL_COLUMNS = [
    "tick", "round_num", "t", "attacker", "victim", "assister", "assistedflash",
    "weapon", "headshot", "attacker_team", "victim_team", "attacker_place",
    "victim_place", "attacker_x", "attacker_y", "victim_x", "victim_y",
    "attacker_cat", "victim_cat",
]
DAMAGE_COLUMNS = ["tick", "round_num", "t", "attacker", "victim", "dmg", "weapon", "attacker_team", "victim_team"]
BLIND_COLUMNS = ["tick", "round_num", "t", "attacker", "victim", "duration", "attacker_team", "victim_team"]
BOMB_COLUMNS = ["tick", "round_num", "t", "event", "player", "site"]
SAMPLE_COLUMNS = [
    "tick", "round_num", "t", "steamid", "team", "is_alive", "x", "y", "z", "place",
    "health", "equip", "has_defuser", "is_event",
]


@dataclass
class MatchData:
    """One parsed map (one demo)."""

    source: str
    map_name: str
    tick_rate: float
    players: pd.DataFrame  # steamid, name
    rounds: pd.DataFrame
    kills: pd.DataFrame
    damages: pd.DataFrame
    blinds: pd.DataFrame
    bomb: pd.DataFrame
    samples: pd.DataFrame
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    def player_name(self, steamid) -> str:
        row = self.players.loc[self.players["steamid"] == steamid, "name"]
        return str(row.iloc[0]) if len(row) else str(steamid)

    def team_names(self) -> list[str]:
        names = set(self.rounds["t_team"]).union(self.rounds["ct_team"])
        return sorted(n for n in names if isinstance(n, str) and n)

    def side_of(self, team_name: str, round_num: int) -> str | None:
        r = self.rounds.loc[self.rounds["round_num"] == round_num]
        if r.empty:
            return None
        r = r.iloc[0]
        if r["t_team"] == team_name:
            return "T"
        if r["ct_team"] == team_name:
            return "CT"
        return None

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        payload = {
            "format": FORMAT_VERSION,
            "source": self.source,
            "map_name": self.map_name,
            "tick_rate": self.tick_rate,
            "players": self.players,
            "rounds": self.rounds,
            "kills": self.kills,
            "damages": self.damages,
            "blinds": self.blinds,
            "bomb": self.bomb,
            "samples": self.samples,
            "extra": self.extra,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "MatchData":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("format") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported cache format {payload.get('format')}")
        payload.pop("format")
        return cls(**payload)


def load_many(paths: Iterable[str | Path]) -> list[MatchData]:
    return [MatchData.load(p) for p in paths]


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
