"""Position analytics on top of the win-probability model.

* ``zone_values``   – which sections of the map are worth holding, per side and phase.
* ``team_setups``   – a team's recurring setups at a reference moment with their expected
                      and actual win rates: flaws are common setups with a negative edge, or
                      setups that lose more than the model expects (execution problem).
* ``reactions``     – given what the enemy is doing at a moment, which own setups did best
                      (empirically and according to the model).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MatchData
from .state import GameState
from .winprob import WinProbModel


@dataclass
class Dataset:
    """Several matches merged: states + metadata (with match/team columns)."""

    meta: pd.DataFrame
    states: list[GameState]
    matches: list[MatchData]

    @classmethod
    def build(cls, matches: list[MatchData], zoner) -> "Dataset":
        from .state import build_states

        metas, states = [], []
        for mi, m in enumerate(matches):
            meta, st = build_states(m, zoner)
            if meta.empty:
                continue
            rounds = m.rounds.set_index("round_num")
            meta = meta.copy()
            meta["match"] = mi
            meta["map_name"] = m.map_name
            meta["t_team"] = meta["round_num"].map(rounds["t_team"])
            meta["ct_team"] = meta["round_num"].map(rounds["ct_team"])
            meta["first_contact_t"] = meta["round_num"].map(_first_contact(m))
            metas.append(meta)
            states.extend(st)
        meta_all = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
        return cls(meta_all, states, matches)

    def fit_model(self, map_name: str | None = None, **kw) -> WinProbModel:
        idx = self.meta.index if map_name is None else self.meta.index[self.meta["map_name"] == map_name]
        states = [self.states[i] for i in idx]
        n_rounds = int(self.meta.loc[idx].groupby(["match", "round_num"]).ngroups)
        model = WinProbModel(**kw)
        return model.fit(states, self.meta.loc[idx, "t_win"].to_numpy(), self.meta.loc[idx, "weight"].to_numpy(), n_rounds)

    def maps(self) -> list[str]:
        return sorted(self.meta["map_name"].unique()) if not self.meta.empty else []


def _first_contact(m: MatchData) -> dict[int, float]:
    out: dict[int, float] = {}
    for df in (m.damages, m.kills):
        if df.empty:
            continue
        for rnum, t in df.groupby("round_num")["t"].min().items():
            out[int(rnum)] = min(out.get(int(rnum), 1e9), float(t))
    return out


# --------------------------------------------------------------------- zone values
def zone_values(ds: Dataset, model: WinProbModel, map_name: str, min_rounds: float = 3.0) -> pd.DataFrame:
    idx = ds.meta.index[ds.meta["map_name"] == map_name]
    occ_w: dict[tuple, float] = defaultdict(float)  # weighted rounds with >=1 player there
    occ_win: dict[tuple, float] = defaultdict(float)
    base_w: dict[tuple, float] = defaultdict(float)
    base_win: dict[tuple, float] = defaultdict(float)
    for i in idx:
        st = ds.states[i]
        if st.terminal_t_win() is not None:
            continue
        w, y = float(ds.meta.at[i, "weight"]), float(ds.meta.at[i, "t_win"])
        for side, zones in (("T", st.t_zones), ("CT", st.ct_zones)):
            win = y if side == "T" else 1.0 - y
            base_w[(side, st.phase)] += w
            base_win[(side, st.phase)] += w * win
            for z in set(zones):
                occ_w[(side, st.phase, z)] += w
                occ_win[(side, st.phase, z)] += w * win
    coefs = {(r["side"], r["phase"], r["zone"]): r for r in model.zone_coefficients()}
    rows = []
    for key, w in occ_w.items():
        side, phase, zone = key
        if w < min_rounds:
            continue
        c = coefs.get(key, {"logit": 0.0, "pp_at_50": 0.0})
        base_rate = base_win[(side, phase)] / max(1e-9, base_w[(side, phase)])
        rows.append({
            "side": side, "phase": phase, "zone": zone, "rounds_weighted": round(w, 1),
            "win_rate_when_held": occ_win[key] / w, "side_win_rate": base_rate,
            "lift_pp": 100 * (occ_win[key] / w - base_rate),
            "model_pp_at_50": c["pp_at_50"], "model_logit": c["logit"],
        })
    return pd.DataFrame(rows).sort_values(["side", "phase", "model_pp_at_50"], ascending=[True, True, False]).reset_index(drop=True)


# --------------------------------------------------------------------- team setups
def _pick_moment(meta_round: pd.DataFrame, at: str | float) -> int | None:
    """Index of the sample for the reference moment: seconds, or 'contact' for first contact."""
    if meta_round.empty:
        return None
    if at == "contact":
        target = float(meta_round["first_contact_t"].iloc[0])
        if not np.isfinite(target) or target > 1e8:
            return None
        target = max(0.0, target - 1.0)
    else:
        target = float(at)
    pre = meta_round[meta_round["t"] <= target]
    if pre.empty:
        return None
    return int(pre.index[-1])


def team_setups(ds: Dataset, model: WinProbModel, team: str, side: str, map_name: str,
                at: str | float = 20.0, min_n: int = 2) -> pd.DataFrame:
    rows = []
    sub = ds.meta[(ds.meta["map_name"] == map_name) & (ds.meta[f"{side.lower()}_team"] == team)]
    for (mi, rnum), grp in sub.groupby(["match", "round_num"]):
        i = _pick_moment(grp, at)
        if i is None:
            continue
        st = ds.states[i]
        if st.terminal_t_win() is not None:
            continue
        p_pos, p_base = model.p_side(st, side, True), model.p_side(st, side, False)
        win = float(grp["t_win"].iloc[0]) if side == "T" else 1.0 - float(grp["t_win"].iloc[0])
        rows.append({"match": mi, "round_num": rnum, "setup": st.signature(side), "enemy": st.signature("CT" if side == "T" else "T"),
                     "p_pos": p_pos, "p_base": p_base, "edge": p_pos - p_base, "won": win})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("setup").agg(n=("won", "size"), win_rate=("won", "mean"), p_model=("p_pos", "mean"),
                                  p_base=("p_base", "mean"), edge=("edge", "mean")).reset_index()
    agg["execution_gap"] = agg["win_rate"] - agg["p_model"]
    agg["edge_pp"] = agg["edge"] * 100
    agg = agg[agg["n"] >= min_n].sort_values(["edge", "n"], ascending=[True, False]).reset_index(drop=True)
    agg.attrs["per_round"] = df
    return agg


# ----------------------------------------------------------------------- reactions
def _similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Multiset Jaccard similarity of two zone tuples."""
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 1.0


def reactions(ds: Dataset, model: WinProbModel, map_name: str, side: str, enemy_zones: tuple[str, ...],
              at: str | float = 20.0, min_similarity: float = 0.6, top: int = 10,
              template: GameState | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What should ``side`` do when the enemy is in ``enemy_zones``?

    Returns (empirical, model_ranked): setups that were actually played against similar
    enemy setups with their outcomes, and all observed own setups at that moment ranked by
    the model against exactly this enemy setup.
    """
    enemy = "CT" if side == "T" else "T"
    sub = ds.meta[ds.meta["map_name"] == map_name]
    emp_rows, own_setups = [], {}
    for (mi, rnum), grp in sub.groupby(["match", "round_num"]):
        i = _pick_moment(grp, at)
        if i is None:
            continue
        st = ds.states[i]
        if st.terminal_t_win() is not None or st.t_alive == 0 or st.ct_alive == 0:
            continue
        own = st.zones_of(side)
        own_setups.setdefault(st.signature(side), own)
        sim = _similarity(st.zones_of(enemy), enemy_zones)
        if sim >= min_similarity:
            win = float(grp["t_win"].iloc[0]) if side == "T" else 1.0 - float(grp["t_win"].iloc[0])
            emp_rows.append({"setup": st.signature(side), "similarity": sim, "won": win})
    empirical = pd.DataFrame(emp_rows)
    if not empirical.empty:
        empirical = empirical.groupby("setup").agg(n=("won", "size"), win_rate=("won", "mean"),
                                                   similarity=("similarity", "mean")).reset_index()
        empirical = empirical.sort_values(["win_rate", "n"], ascending=False).head(top).reset_index(drop=True)

    base = template or GameState(map_name=map_name, t_alive=5, ct_alive=5, bomb_planted=False,
                                 time_left=115.0 - (float(at) if at != "contact" else 25.0))
    base = base.with_zones(enemy, tuple(enemy_zones))
    ranked_rows = []
    for sig, zones in own_setups.items():
        if len(zones) != (base.t_alive if side == "T" else base.ct_alive):
            continue
        st = base.with_zones(side, tuple(zones))
        ranked_rows.append({"setup": sig, "p_model": model.p_side(st, side, True),
                            "edge_pp": 100 * model.positional_edge(st, side)})
    ranked = pd.DataFrame(ranked_rows)
    if not ranked.empty:
        ranked = ranked.sort_values("p_model", ascending=False).head(top).reset_index(drop=True)
    return empirical, ranked


def parse_zone_spec(spec: str) -> tuple[str, ...]:
    """'A Site x2, Connector, Mid=2' -> ('A Site','A Site','Connector','Mid','Mid')."""
    zones: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        n = 1
        for sep in (" x", "=", "*"):
            if sep in part:
                name, cnt = part.rsplit(sep, 1)
                if cnt.strip().isdigit():
                    part, n = name.strip(), int(cnt)
                    break
        zones.extend([part] * n)
    return tuple(sorted(zones))
