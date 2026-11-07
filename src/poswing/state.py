"""Game states: the unit the win-probability model, swing and engine reason about."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .constants import BOMB_DEAD_ZONE_KIT_S, BOMB_DEAD_ZONE_S, BOMB_TIME_S, ROUND_TIME_S
from .data import MatchData
from .zones import Zoner

BUY_RANK = {"eco": 0, "semi-eco": 1, "semi-buy": 2, "full-buy": 3}


@dataclass(frozen=True)
class GameState:
    map_name: str
    t_alive: int
    ct_alive: int
    bomb_planted: bool
    time_left: float  # seconds until round timer / bomb explosion
    t_buy: str = "full-buy"
    ct_buy: str = "full-buy"
    ct_has_kit: bool = True
    t_zones: tuple[str, ...] = field(default_factory=tuple)  # zones of alive Ts
    ct_zones: tuple[str, ...] = field(default_factory=tuple)
    defusing: bool = False
    bomb_site: str | None = None  # "A" / "B" once planted (None if unknown or pre-plant)

    # ------------------------------------------------------------------ helpers
    @property
    def phase(self) -> str:
        """'pre', or 'postA' / 'postB' (plain 'post' when the site is unknown)."""
        return phase_label(self.bomb_planted, self.bomb_site)

    def occupancy(self) -> Counter:
        c: Counter = Counter()
        for z in self.t_zones:
            c[("T", z)] += 1
        for z in self.ct_zones:
            c[("CT", z)] += 1
        return c

    def zones_of(self, side: str) -> tuple[str, ...]:
        return self.t_zones if side == "T" else self.ct_zones

    def with_zones(self, side: str, zones: tuple[str, ...]) -> "GameState":
        return replace(self, t_zones=tuple(zones)) if side == "T" else replace(self, ct_zones=tuple(zones))

    def signature(self, side: str) -> str:
        """Human readable multiset of zones, e.g. 'A Site x2, Connector, Mid x2'."""
        c = Counter(self.zones_of(side))
        return ", ".join(f"{z} x{n}" if n > 1 else z for z, n in sorted(c.items()))

    def terminal_t_win(self) -> float | None:
        """Win probability fixed by the rules (None when the round is still open)."""
        if self.ct_alive <= 0:
            return 1.0
        if self.t_alive <= 0:
            return 0.03 if self.bomb_planted else 0.0
        if self.bomb_planted and not self.defusing:
            limit = BOMB_DEAD_ZONE_KIT_S if self.ct_has_kit else BOMB_DEAD_ZONE_S
            if self.time_left < limit:
                return 1.0
        if not self.bomb_planted and self.time_left <= 0:
            return 0.0
        return None


def phase_label(planted: bool, site: str | None) -> str:
    if not planted:
        return "pre"
    return f"post{site}" if site in ("A", "B") else "post"


def plant_sites(match: MatchData) -> dict[int, str]:
    """Bomb site per round ('A'/'B').  Demos store an entity index as the site, so
    the site is read off the planter's nav place at the plant tick when needed."""
    out: dict[int, str] = {}
    for r in match.rounds.itertuples(index=False):
        site = getattr(r, "plant_site", None)
        if isinstance(site, str) and site.upper() in ("A", "B"):
            out[int(r.round_num)] = site.upper()
    if match.bomb.empty:
        return out
    plants = match.bomb[match.bomb["event"] == "plant"].drop_duplicates("round_num")
    need = plants[~plants["round_num"].astype(int).isin(out)]
    if need.empty:
        return out
    s = match.samples
    for p in need.itertuples(index=False):
        rnum = int(p.round_num)
        if pd.isna(p.player):
            continue
        rows = s[(s["round_num"] == rnum) & (s["steamid"] == int(p.player)) & (s["tick"] <= int(p.tick))]
        if rows.empty:
            continue
        place = str(rows.sort_values("tick")["place"].iloc[-1])
        if place.endswith("A"):
            out[rnum] = "A"
        elif place.endswith("B"):
            out[rnum] = "B"
    return out


def time_left_at(t: float, plant_t: float | None) -> tuple[bool, float]:
    """Round clock: (bomb_planted, seconds left) for round time ``t`` since freeze end."""
    if plant_t is not None and not np.isnan(plant_t) and t >= plant_t:
        return True, max(0.0, BOMB_TIME_S - (t - plant_t))
    return False, max(0.0, ROUND_TIME_S - t)


# ---------------------------------------------------------------------- building
def add_zones(match: MatchData, zoner: Zoner) -> pd.DataFrame:
    samples = match.samples.copy()
    samples["zone"] = zoner.assign(samples)
    return samples


def build_states(match: MatchData, zoner: Zoner) -> tuple[pd.DataFrame, list[GameState]]:
    """One GameState per sampled tick.

    Returns a metadata frame (round_num, tick, t, is_event, t_win, weight) aligned
    with the list of states.  ``weight`` is 1 / samples-in-round so that every round
    counts the same when fitting, regardless of its length.
    """
    samples = add_zones(match, zoner).sort_values(["round_num", "tick"], kind="stable")
    rounds = match.rounds.set_index("round_num")
    plant_t = _plant_times(match)
    defuse_windows = _defuse_windows(match)

    # numpy views: a Python loop over ticks with pandas slicing is far too slow for many demos
    rnum_a = samples["round_num"].to_numpy()
    tick_a = samples["tick"].to_numpy()
    t_a = samples["t"].to_numpy(dtype=float)
    alive_a = samples["is_alive"].to_numpy(dtype=bool)
    team_a = samples["team"].to_numpy(dtype=object)
    zone_a = samples["zone"].to_numpy(dtype=object)
    event_a = samples["is_event"].to_numpy(dtype=bool)
    groups = samples.groupby(["round_num", "tick"], sort=True).indices

    meta_rows: list[dict] = []
    states: list[GameState] = []
    for (rnum, tick), idx in groups.items():
        r = rounds.loc[rnum]
        t = float(t_a[idx[0]])
        al = idx[alive_a[idx]]
        is_t = team_a[al] == "T"
        planted, left = time_left_at(t, plant_t.get(rnum))
        st = GameState(
            map_name=match.map_name,
            t_alive=int(is_t.sum()),
            ct_alive=int((~is_t).sum()),
            bomb_planted=planted,
            time_left=left,
            t_buy=str(r["t_buy"]), ct_buy=str(r["ct_buy"]),
            ct_has_kit=bool(r["ct_has_kit"]),
            t_zones=tuple(sorted(map(str, zone_a[al[is_t]]))),
            ct_zones=tuple(sorted(map(str, zone_a[al[~is_t]]))),
            defusing=_in_windows(t, defuse_windows.get(rnum, [])),
        )
        states.append(st)
        meta_rows.append({
            "round_num": int(rnum), "tick": int(tick), "t": t,
            "is_event": bool(event_a[idx[0]]),
            "t_win": 1.0 if r["winner"] == "T" else 0.0,
        })
    meta = pd.DataFrame(meta_rows)
    if not meta.empty:
        counts = meta.groupby("round_num")["tick"].transform("count")
        meta["weight"] = 1.0 / counts
    return meta, states


def _plant_times(match: MatchData) -> dict[int, float]:
    out: dict[int, float] = {}
    if match.bomb.empty:
        return out
    plants = match.bomb[match.bomb["event"] == "plant"]
    for row in plants.itertuples(index=False):
        out.setdefault(int(row.round_num), float(row.t))
    return out


def _defuse_windows(match: MatchData) -> dict[int, list[tuple[float, float]]]:
    """Intervals during which a CT is on the bomb (begin_defuse -> defuse/abort guess)."""
    out: dict[int, list[tuple[float, float]]] = {}
    if match.bomb.empty:
        return out
    for rnum, grp in match.bomb.groupby("round_num"):
        begins = grp[grp["event"] == "begin_defuse"]["t"].tolist()
        ends = grp[grp["event"] == "defuse"]["t"].tolist()
        windows = []
        for b in begins:
            later = [e for e in ends if e >= b]
            windows.append((b, later[0] if later else b + 10.0))
        out[int(rnum)] = windows
    return out


def _in_windows(t: float, windows: list[tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in windows)
