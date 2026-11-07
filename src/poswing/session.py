"""One analysed match, ready to be served: model, swing, box score, per-round frames and the engine.

The web app and anything else that wants JSON-friendly views of a demo should go
through ``MatchSession`` so that the numbers shown everywhere come from the same code
path as the CLI commands.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from . import resources
from .analysis import Dataset, zone_values
from .data import MatchData
from .economy import DuelTable
from .engine import Engine, Situation, TransitionGraph, situation_at
from .rating import box_score
from .state import plant_sites
from .swing import SwingResult, compute_swing
from .winprob import WinProbModel
from .zones import Zoner

ROUND_SECONDS = 115.0
BOMB_SECONDS = 40.0


def _clean(v: Any) -> Any:
    """Make numpy / pandas scalars JSON friendly, NaN -> None."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NaT:
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def records(df: pd.DataFrame, columns: list[str] | None = None, ndigits: int | None = 3) -> list[dict]:
    if df is None or df.empty:
        return []
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    out = []
    for row in df[cols].itertuples(index=False):
        d = {}
        for c, v in zip(cols, row):
            v = _clean(v)
            if ndigits is not None and isinstance(v, float):
                v = round(v, ndigits)
            d[c] = v
        out.append(d)
    return out


@dataclass
class MatchSession:
    match: MatchData
    zoner: Zoner = field(default_factory=Zoner)
    model_path: str | None = None  # explicit model file; None -> bundled model or a fit on the demo itself
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------------ model
    @cached_property
    def dataset(self) -> Dataset:
        return Dataset.build([self.match], self.zoner)

    @cached_property
    def _model_and_source(self) -> tuple[WinProbModel, str]:
        if self.model_path:
            m = WinProbModel.load(self.model_path)
            if self.match.map_name in m.maps:
                return m, "file"
        bundled = resources.bundled_model(self.match.map_name)
        if bundled is not None:
            return bundled, "bundled"
        return self.dataset.fit_model(self.match.map_name), "fitted on this demo"

    @property
    def model(self) -> WinProbModel:
        return self._model_and_source[0]

    @property
    def model_source(self) -> str:
        return self._model_and_source[1]

    @cached_property
    def graph(self) -> TransitionGraph:
        own = TransitionGraph.from_matches([self.match], self.zoner, self.match.map_name)
        bundled = resources.bundled_graph(self.match.map_name)
        return bundled.merge(own) if bundled is not None else own

    @cached_property
    def engine(self) -> Engine:
        return Engine(self.model, self.graph)

    # --------------------------------------------------------------- analysis
    @cached_property
    def swing(self) -> SwingResult:
        meta = self.dataset.meta
        return compute_swing(self.match, self.model, self.zoner, meta_states=(meta, self.dataset.states))

    @cached_property
    def box(self) -> pd.DataFrame:
        duels = DuelTable().fit(self.match.kills)
        return box_score(self.match, self.swing, duels)

    @cached_property
    def win_prob(self) -> pd.DataFrame:
        """P(T wins) per sampled tick, with and without positions."""
        meta = self.dataset.meta
        states = self.dataset.states
        out = meta[["round_num", "tick", "t"]].copy()
        out["p_t"] = self.model.predict(states, positional=True)
        out["p_base"] = self.model.predict(states, positional=False)
        return out

    @cached_property
    def zone_centroids(self) -> dict[str, tuple[float, float]]:
        """Where a zone 'is' on the map: mean of the demo's alive samples, nav-mesh centre as a fallback."""
        out: dict[str, tuple[float, float]] = {}
        try:
            from .esta import nav_places

            nav = nav_places(self.match.map_name)
            df = pd.DataFrame({"zone": nav.names, "x": nav.cx, "y": nav.cy})
            for z, g in df.groupby("zone"):
                out[str(z)] = (float(g["x"].median()), float(g["y"].median()))
        except Exception:  # noqa: BLE001 - no nav data for this map is fine
            pass
        s = self.samples_z
        s = s[s["is_alive"]]
        agg = s.groupby("zone").agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"))
        for z, r in agg.iterrows():
            if r["n"] >= 5:
                out[str(z)] = (float(r["x"]), float(r["y"]))
        return out

    @cached_property
    def samples_z(self) -> pd.DataFrame:
        from .state import add_zones

        return add_zones(self.match, self.zoner)

    def warm(self) -> None:
        """Compute everything the UI needs up front (call from a background thread)."""
        with self._lock:
            _ = self.model, self.swing, self.box, self.win_prob, self.graph, self.zone_centroids

    # --------------------------------------------------------------- summaries
    def rounds_table(self) -> list[dict]:
        m = self.match
        sites = plant_sites(m)
        score: dict[str, int] = {}
        rows = []
        swing_rounds = self.swing.rounds.set_index("round_num") if not self.swing.rounds.empty else None
        for r in m.rounds.itertuples(index=False):
            rnum = int(r.round_num)
            winner_team = r.t_team if r.winner == "T" else r.ct_team
            before = dict(score)
            score[str(winner_team)] = score.get(str(winner_team), 0) + 1
            kills = m.kills[m.kills["round_num"] == rnum]
            row = {
                "round_num": rnum, "winner": r.winner, "winner_team": winner_team, "reason": r.reason,
                "t_team": r.t_team, "ct_team": r.ct_team, "t_buy": r.t_buy, "ct_buy": r.ct_buy,
                "t_equip": r.t_equip, "ct_equip": r.ct_equip,
                "planted": bool(pd.notna(r.plant_tick)), "plant_site": sites.get(rnum),
                "plant_t": ((float(r.plant_tick) - float(r.freeze_end_tick)) / m.tick_rate) if pd.notna(r.plant_tick) else None,
                "duration": (int(r.end_tick) - int(r.freeze_end_tick)) / m.tick_rate,
                "kills": int(len(kills)),
                "score_before": {str(k): int(v) for k, v in before.items()},
                "score_after": {str(k): int(v) for k, v in score.items()},
            }
            if swing_rounds is not None and rnum in swing_rounds.index:
                sr = swing_rounds.loc[rnum]
                row["p0_t"] = float(sr["p0_t"])
            rows.append(_clean(row))
        return rows

    def summary(self) -> dict:
        m = self.match
        teams = m.team_names()
        final: dict[str, int] = {}
        for r in m.rounds.itertuples(index=False):
            w = r.t_team if r.winner == "T" else r.ct_team
            final[str(w)] = final.get(str(w), 0) + 1
        players = m.players.copy()
        team_of = m.samples.groupby("steamid").apply(lambda g: g.sort_values("round_num")["team"].iloc[0])
        # a player's team name: the name of the side they played in their first round
        first_round = m.rounds.iloc[0]
        team_name = {}
        for sid in players["steamid"]:
            side = team_of.get(sid)
            team_name[int(sid)] = str(first_round["t_team"] if side == "T" else first_round["ct_team"])
        box = self.box.copy()
        if not box.empty:
            box["team"] = box["steamid"].map(team_name)
        sw = self.swing.players.copy()
        if not sw.empty:
            sw["team"] = sw["steamid"].map(team_name)
        events = self.swing.events
        key = []
        if not events.empty:
            ev = events[events["component"] == "swing"].copy()
            # keep the actor's side of each event: plant credited to Ts, defuse to CTs; residuals are not moments
            ev = ev[~((ev["kind"] == "plant") & (ev["team"] == "CT"))]
            ev = ev[~((ev["kind"] == "defuse") & (ev["team"] == "T"))]
            ev = ev[~ev["kind"].isin(["round_end"])]
            ev["abs"] = ev["delta"].abs()
            ev = ev.sort_values("abs", ascending=False).drop_duplicates(["round_num", "tick", "kind"]).head(12)
            for e in ev.itertuples(index=False):
                key.append({"round_num": int(e.round_num), "tick": int(e.tick), "t": float(e.t), "kind": e.kind,
                            "steamid": int(e.steamid), "name": m.player_name(int(e.steamid)), "team": e.team,
                            "delta_pp": round(100 * float(e.delta), 1)})
        return _clean({
            "source": m.source, "map_name": m.map_name, "tick_rate": m.tick_rate, "n_rounds": m.n_rounds,
            "teams": teams, "final_score": final,
            "model": {"source": self.model_source, "n_rounds": self.model.n_rounds, "n_states": self.model.n_states,
                      "features": len(self.model.positional.features), "target": self.model.positional.target},
            "radar": resources.radar_info(m.map_name),
            "players": [{"steamid": int(s), "name": str(n), "team": team_name.get(int(s))}
                        for s, n in zip(players["steamid"], players["name"])],
            "rounds": self.rounds_table(),
            "box_score": records(box),
            "swing": records(sw),
            "key_moments": key,
        })

    # ------------------------------------------------------------ round detail
    def round_detail(self, round_num: int) -> dict:
        m = self.match
        r = m.rounds[m.rounds["round_num"] == round_num]
        if r.empty:
            raise KeyError(f"round {round_num} not in match")
        r = r.iloc[0]
        s = self.samples_z[self.samples_z["round_num"] == round_num].sort_values(["tick", "steamid"])
        wp = self.win_prob[self.win_prob["round_num"] == round_num].set_index("tick")
        plant_t = ((float(r["plant_tick"]) - float(r["freeze_end_tick"])) / m.tick_rate) if pd.notna(r["plant_tick"]) else None
        frames = []
        for tick, g in s.groupby("tick", sort=True):
            t = float(g["t"].iloc[0])
            players = [[int(p.steamid), bool(p.is_alive), round(float(p.x), 1), round(float(p.y), 1), round(float(p.z), 1),
                        int(p.health), str(p.zone)] for p in g.itertuples(index=False)]
            frames.append({
                "tick": int(tick), "t": round(t, 3),
                "p_t": round(float(wp.at[tick, "p_t"]), 4) if tick in wp.index else None,
                "p_base": round(float(wp.at[tick, "p_base"]), 4) if tick in wp.index else None,
                "players": players,
            })
        kills = m.kills[m.kills["round_num"] == round_num]
        flags = self.swing.flags
        flags = flags[flags["round_num"] == round_num].set_index("tick") if not flags.empty else None
        kill_rows = []
        for k in kills.itertuples(index=False):
            row = {"tick": int(k.tick), "t": float(k.t), "attacker": _clean(k.attacker), "victim": int(k.victim),
                   "assister": _clean(k.assister), "weapon": k.weapon, "headshot": bool(k.headshot),
                   "attacker_team": k.attacker_team, "victim_team": k.victim_team,
                   "ax": _clean(k.attacker_x), "ay": _clean(k.attacker_y), "vx": _clean(k.victim_x), "vy": _clean(k.victim_y),
                   "attacker_name": m.player_name(int(k.attacker)) if pd.notna(k.attacker) else None,
                   "victim_name": m.player_name(int(k.victim))}
            if flags is not None and int(k.tick) in flags.index:
                f = flags.loc[int(k.tick)]
                if isinstance(f, pd.DataFrame):
                    f = f.iloc[0]
                row.update({"is_trade": bool(f["is_trade"]), "failed_trade": bool(f["failed_trade"]),
                            "trade_denial": bool(f["trade_denial"])})
            kill_rows.append(_clean(row))
        bomb = m.bomb[m.bomb["round_num"] == round_num] if not m.bomb.empty else pd.DataFrame()
        bomb_rows = []
        for b in bomb.itertuples(index=False):
            pos = None
            if pd.notna(b.player):
                at = s[(s["steamid"] == int(b.player)) & (s["tick"] <= int(b.tick))]
                if not at.empty:
                    last = at.iloc[-1]
                    pos = [round(float(last["x"]), 1), round(float(last["y"]), 1)]
            bomb_rows.append(_clean({"tick": int(b.tick), "t": float(b.t), "event": b.event, "player": _clean(b.player),
                                     "player_name": m.player_name(int(b.player)) if pd.notna(b.player) else None,
                                     "site": plant_sites(m).get(round_num) if b.event == "plant" else None, "pos": pos}))
        ev = self.swing.events
        ev = ev[ev["round_num"] == round_num] if not ev.empty else ev
        events = [{"tick": int(e.tick), "t": float(e.t), "kind": e.kind, "steamid": int(e.steamid),
                   "name": m.player_name(int(e.steamid)), "team": e.team, "component": e.component,
                   "delta": round(float(e.delta), 4)} for e in ev.itertuples(index=False)]
        roster = {int(sid): {"name": m.player_name(int(sid)), "team": str(team)}
                  for sid, team in s.groupby("steamid")["team"].first().items()}
        return _clean({
            "round_num": int(round_num), "winner": r["winner"], "reason": r["reason"],
            "t_team": r["t_team"], "ct_team": r["ct_team"], "t_buy": r["t_buy"], "ct_buy": r["ct_buy"],
            "plant_t": plant_t, "plant_site": plant_sites(m).get(round_num),
            "duration": (int(r["end_tick"]) - int(r["freeze_end_tick"])) / m.tick_rate,
            "freeze_end_tick": int(r["freeze_end_tick"]), "tick_rate": m.tick_rate,
            "roster": roster, "frames": frames, "kills": kill_rows, "bomb": bomb_rows, "events": events,
        })

    # ---------------------------------------------------------------- engine
    def suggest(self, round_num: int, t: float, side: str, horizon: float = 10.0, top: int = 5,
                enemy_mode: str = "static", max_movers: int | None = None, beam: int = 40) -> dict:
        sit: Situation = situation_at(self.match, self.zoner, round_num, t=t)
        if not sit.own(side):
            raise ValueError(f"no alive {side} players at {t:.0f}s")
        with self._lock:
            lines, p_now = self.engine.suggest(sit, side, horizon=horizon, top=top, beam=beam,
                                               enemy_mode=enemy_mode, max_movers=max_movers)
        cent = self.zone_centroids
        enemy = "CT" if side == "T" else "T"
        own = sit.own(side)
        out_lines = []
        for ln in lines:
            moves = []
            for name, a, b in ln.moves:
                pl = next((p for p in own if p.name == name and p.zone == a), None)
                moves.append({"name": name, "steamid": pl.steamid if pl else None, "from": a, "to": b,
                              "to_xy": list(cent.get(b, (None, None))), "from_xy": list(cent.get(a, (None, None)))})
            out_lines.append({"p_win": round(ln.p_win, 4), "delta_pp": round(100 * ln.delta, 2),
                              "zones": list(ln.zones), "moves": moves})
        return _clean({
            "situation": sit.describe(), "round_num": round_num, "t": sit.t, "tick": sit.tick, "side": side,
            "team": sit.teams.get(side), "enemy_team": sit.teams.get(enemy),
            "p_now": round(p_now, 4), "p_base": round(self.model.p_side(sit.state, side, False), 4),
            "phase": sit.state.phase, "horizon": horizon, "enemy_mode": enemy_mode,
            "own": [{"steamid": p.steamid, "name": p.name, "zone": p.zone} for p in sit.own(side)],
            "enemy": [{"steamid": p.steamid, "name": p.name, "zone": p.zone} for p in sit.own(enemy)],
            "lines": out_lines,
        })

    def zone_table(self, min_rounds: float = 1.0) -> list[dict]:
        """Every zone the model has a coefficient for, with this demo's empirical numbers where it has any."""
        zv = zone_values(self.dataset, self.model, self.match.map_name, min_rounds=min_rounds)
        seen = {(r.side, r.phase, r.zone) for r in zv.itertuples(index=False)} if not zv.empty else set()
        extra = [{"side": c["side"], "phase": c["phase"], "zone": c["zone"], "rounds_weighted": 0.0,
                  "win_rate_when_held": None, "side_win_rate": None, "lift_pp": None,
                  "model_pp_at_50": c["pp_at_50"], "model_logit": c["logit"]}
                 for c in self.model.zone_coefficients() if (c["side"], c["phase"], c["zone"]) not in seen]
        out = pd.concat([zv, pd.DataFrame(extra)], ignore_index=True) if extra else zv
        if out.empty:
            return []
        out = out.sort_values(["side", "phase", "model_pp_at_50"], ascending=[True, True, False])
        return records(out)
