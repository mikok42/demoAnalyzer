"""Engine mode: give it a demo and a moment, get the best plays for a team.

Like a chess engine, it enumerates legal "moves" (each alive player can stay or move
to a zone reachable within the horizon, reachability learned from how players
actually move between zones in the data), evaluates the resulting positions with the
win-probability model and returns the top lines.  The opponent can be assumed
static or predicted from the empirical zone transitions.  A retrieval "book" shows
what teams did in similar situations and how it went.
"""
from __future__ import annotations

import heapq
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .analysis import Dataset, _similarity
from .data import MatchData
from .state import GameState, add_zones, phase_label, plant_sites, time_left_at
from .winprob import WinProbModel
from .zones import Zoner


# ------------------------------------------------------------------- situation
@dataclass
class PlayerPos:
    steamid: int
    name: str
    team: str
    zone: str
    alive: bool
    health: float


@dataclass
class Situation:
    match: MatchData
    round_num: int
    tick: int
    t: float
    state: GameState
    players: list[PlayerPos]
    teams: dict[str, str]  # side -> team name

    def own(self, side: str) -> list[PlayerPos]:
        return [p for p in self.players if p.team == side and p.alive]

    def describe(self) -> str:
        st = self.state
        clock = (f"bomb on {st.bomb_site or '?'} {st.time_left:.0f}s" if st.bomb_planted else f"{st.time_left:.0f}s left")
        lines = [
            f"{self.match.map_name} round {self.round_num} @ {self.t:.1f}s (tick {self.tick}) - {clock}, "
            f"{st.t_alive}v{st.ct_alive} (T v CT), T buy {st.t_buy}, CT buy {st.ct_buy}",
            f"  T  ({self.teams.get('T', 'T')}): " + ", ".join(f"{p.name}@{p.zone}" for p in self.own("T")),
            f"  CT ({self.teams.get('CT', 'CT')}): " + ", ".join(f"{p.name}@{p.zone}" for p in self.own("CT")),
        ]
        return "\n".join(lines)


def situation_at(match: MatchData, zoner: Zoner, round_num: int, t: float | None = None,
                 tick: int | None = None) -> Situation:
    s = add_zones(match, zoner)
    s = s[s["round_num"] == round_num]
    if s.empty:
        raise ValueError(f"round {round_num} not in demo (has {match.n_rounds} rounds)")
    if tick is None:
        if t is None:
            raise ValueError("give either t (seconds into the round) or tick")
        candidates = s[s["t"] <= t]
        tick = int(candidates["tick"].max()) if not candidates.empty else int(s["tick"].min())
    else:
        candidates = s[s["tick"] <= tick]
        tick = int(candidates["tick"].max()) if not candidates.empty else int(s["tick"].min())
    grp = s[s["tick"] == tick]
    r = match.rounds.set_index("round_num").loc[round_num]
    t_now = float(grp["t"].iloc[0])
    plant_t = None
    if pd.notna(r["plant_tick"]):
        plant_t = (float(r["plant_tick"]) - float(r["freeze_end_tick"])) / match.tick_rate
    planted, left = time_left_at(t_now, plant_t)
    site = plant_sites(match).get(round_num) if planted else None
    alive = grp[grp["is_alive"]]
    state = GameState(
        map_name=match.map_name, t_alive=int((alive["team"] == "T").sum()), ct_alive=int((alive["team"] == "CT").sum()),
        bomb_planted=planted, time_left=left, t_buy=str(r["t_buy"]), ct_buy=str(r["ct_buy"]), ct_has_kit=bool(r["ct_has_kit"]),
        t_zones=tuple(sorted(alive.loc[alive["team"] == "T", "zone"])), ct_zones=tuple(sorted(alive.loc[alive["team"] == "CT", "zone"])),
        bomb_site=site,
    )
    players = [PlayerPos(int(p.steamid), match.player_name(int(p.steamid)), str(p.team), str(p.zone), bool(p.is_alive), float(p.health))
               for p in grp.itertuples(index=False)]
    return Situation(match, round_num, tick, t_now, state, players, {"T": str(r["t_team"]), "CT": str(r["ct_team"])})


# --------------------------------------------------------------- movement graph
@dataclass
class TransitionGraph:
    """How players move between zones on a map, learned from the samples."""

    map_name: str
    edge_times: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    later: dict[tuple[str, str, str], Counter] = field(default_factory=dict)  # (side, phase, zone) -> zones ~horizon later
    horizon_used: float = 10.0

    @classmethod
    def from_matches(cls, matches: list[MatchData], zoner: Zoner, map_name: str, horizon: float = 10.0) -> "TransitionGraph":
        g = cls(map_name=map_name, horizon_used=horizon)
        for m in matches:
            if m.map_name != map_name:
                continue
            s = add_zones(m, zoner)
            s = s[s["is_alive"]].sort_values(["round_num", "steamid", "tick"])
            plant = {int(b.round_num): float(b.t) for b in m.bomb[m.bomb["event"] == "plant"].itertuples(index=False)} if not m.bomb.empty else {}
            sites = plant_sites(m)
            for (rnum, sid), grp in s.groupby(["round_num", "steamid"]):
                zones = grp["zone"].tolist()
                ts = grp["t"].tolist()
                side = grp["team"].iloc[0]
                last_change_t = ts[0]
                for i in range(1, len(zones)):
                    if zones[i] != zones[i - 1]:
                        dt = ts[i] - last_change_t
                        g.edge_times.setdefault((zones[i - 1], zones[i]), []).append(max(0.5, dt))
                        last_change_t = ts[i]
                # where is the player `horizon` seconds later?
                j = 0
                for i in range(len(zones)):
                    while j < len(ts) and ts[j] < ts[i] + horizon:
                        j += 1
                    if j >= len(ts):
                        break
                    planted = rnum in plant and ts[i] >= plant[rnum]
                    phase = phase_label(planted, sites.get(int(rnum)))
                    g.later.setdefault((side, phase, zones[i]), Counter())[zones[j]] += 1
        return g

    def zones(self) -> set[str]:
        out: set[str] = set()
        for a, b in self.edge_times:
            out.add(a)
            out.add(b)
        return out

    def _medians(self) -> dict[str, list[tuple[str, float]]]:
        """Median transition time per edge, grouped by origin (cached; cheap to recompute)."""
        cache = getattr(self, "_med_cache", None)
        if cache is None or cache[0] != len(self.edge_times):
            out: dict[str, list[tuple[str, float]]] = defaultdict(list)
            seen: set[tuple[str, str]] = set()
            for (a, b), times in self.edge_times.items():
                out[a].append((b, float(np.median(times))))
                seen.add((a, b))
            for a, edges in getattr(self, "_loaded_medians", {}).items():
                for b, t in edges:
                    if (a, b) not in seen:
                        out[a].append((b, t))
            cache = (len(self.edge_times), dict(out))
            self._med_cache = cache
        return cache[1]

    # -- persistence (medians and transition counts only; raw edge samples are not kept)
    def to_dict(self) -> dict:
        med = {a: {b: round(t, 2) for b, t in edges} for a, edges in self._medians().items()}
        counts = {(a, b): len(ts) for (a, b), ts in self.edge_times.items()}
        return {
            "map_name": self.map_name, "horizon_used": self.horizon_used,
            "medians": med,
            "edge_counts": {f"{a}|{b}": n for (a, b), n in counts.items()},
            "later": {"|".join(k): dict(c) for k, c in self.later.items()},
        }

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "TransitionGraph":
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        g = cls(map_name=d["map_name"], horizon_used=d.get("horizon_used", 10.0))
        g._loaded_medians = {a: [(b, float(t)) for b, t in edges.items()] for a, edges in d["medians"].items()}
        g.later = {tuple(k.split("|")): Counter(v) for k, v in d["later"].items()}
        g._edge_counts = {tuple(k.split("|")): int(n) for k, n in d.get("edge_counts", {}).items()}
        return g

    def merge(self, other: "TransitionGraph") -> "TransitionGraph":
        """New graph with both sets of observations (e.g. the demo's own movement on top of a bundled graph).

        Neither input is modified, so a cached bundled graph can be merged repeatedly.
        """
        g = TransitionGraph(map_name=self.map_name, horizon_used=self.horizon_used)
        g._loaded_medians = getattr(self, "_loaded_medians", {})
        g.later = {k: Counter(c) for k, c in self.later.items()}
        for k, c in other.later.items():
            g.later.setdefault(k, Counter()).update(c)
        g.edge_times = {e: list(ts) for e, ts in self.edge_times.items()}
        for e, ts in other.edge_times.items():
            g.edge_times.setdefault(e, []).extend(ts)
        return g

    def travel_time(self, a: str, b: str) -> float:
        for z, t in self._medians().get(a, []):
            if z == b:
                return t
        return float("inf")

    def reachable(self, zone: str, horizon: float) -> dict[str, float]:
        """Dijkstra over median observed transition times."""
        med = self._medians()
        dist = {zone: 0.0}
        heap = [(0.0, zone)]
        while heap:
            d, z = heapq.heappop(heap)
            if d > dist.get(z, float("inf")):
                continue
            for b, t in med.get(z, []):
                nd = d + t
                if nd <= horizon and nd < dist.get(b, float("inf")):
                    dist[b] = nd
                    heapq.heappush(heap, (nd, b))
        return dist

    def predict(self, side: str, phase: str, zone: str) -> dict[str, float]:
        c = self.later.get((side, phase, zone))
        if not c:
            return {zone: 1.0}
        n = sum(c.values())
        return {z: k / n for z, k in c.items()}


# ---------------------------------------------------------------------- engine
@dataclass
class Line:
    zones: tuple[str, ...]
    p_win: float
    delta: float
    moves: list[tuple[str, str, str]]  # (player, from, to)

    def describe(self) -> str:
        if not self.moves:
            mv = "hold current positions"
        else:
            mv = "; ".join(f"{p}: {a} -> {b}" for p, a, b in self.moves)
        return f"P(win) {100 * self.p_win:5.1f}%  ({100 * self.delta:+.1f} pp)  {mv}"


class Engine:
    def __init__(self, model: WinProbModel, graph: TransitionGraph, move_cost: float = 0.003, seed: int = 7):
        self.model = model
        self.graph = graph
        self.move_cost = move_cost
        self.rng = random.Random(seed)

    def _enemy_setups(self, sit: Situation, side: str, horizon: float, mode: str, n_samples: int) -> list[tuple[str, ...]]:
        enemy = "CT" if side == "T" else "T"
        foes = sit.own(enemy)
        if mode == "static" or not foes:
            return [tuple(sorted(p.zone for p in foes))]
        dists = [self.graph.predict(enemy, sit.state.phase, p.zone) for p in foes]
        out = []
        for _ in range(n_samples):
            zs = []
            for d in dists:
                r, acc = self.rng.random(), 0.0
                pick = next(iter(d))
                for z, pz in d.items():
                    acc += pz
                    if r <= acc:
                        pick = z
                        break
                zs.append(pick)
            out.append(tuple(sorted(zs)))
        return out

    def evaluate(self, base: GameState, side: str, own_zones: tuple[str, ...], enemy_setups: list[tuple[str, ...]]) -> float:
        enemy = "CT" if side == "T" else "T"
        states = [base.with_zones(side, own_zones).with_zones(enemy, ez) for ez in enemy_setups]
        p = self.model.predict(states, positional=True)
        return float(np.mean(p if side == "T" else 1.0 - p))

    def suggest(self, sit: Situation, side: str, horizon: float = 10.0, top: int = 5, beam: int = 40,
                enemy_mode: str = "static", n_samples: int = 24, max_movers: int | None = None) -> tuple[list[Line], float]:
        own = sit.own(side)
        if not own:
            raise ValueError(f"no alive {side} players at that moment")
        base = replace(sit.state, time_left=max(0.0, sit.state.time_left - horizon))
        enemy_setups = self._enemy_setups(sit, side, horizon, enemy_mode, n_samples)
        current = tuple(p.zone for p in own)
        p_now = self.evaluate(base, side, tuple(sorted(current)), enemy_setups)

        options = [sorted(self.graph.reachable(p.zone, horizon).items(), key=lambda kv: kv[1]) for p in own]
        # beam search over players; each candidate is a full assignment
        beam_set: list[tuple[float, tuple[str, ...]]] = [(p_now, current)]
        for i, opts in enumerate(options):
            expanded: dict[tuple[str, ...], float] = {}
            for _, assign in beam_set:
                for zone, _eta in opts:
                    cand = list(assign)
                    cand[i] = zone
                    cand_t = tuple(cand)
                    if cand_t in expanded:
                        continue
                    movers = sum(1 for a, b in zip(cand_t, current) if a != b)
                    if max_movers is not None and movers > max_movers:
                        continue
                    score = self.evaluate(base, side, tuple(sorted(cand_t)), enemy_setups) - self.move_cost * movers
                    expanded[cand_t] = score
            beam_set = heapq.nlargest(beam, ((s, a) for a, s in expanded.items()))
        # distinct by sorted multiset of zones, best first
        seen: set[tuple[str, ...]] = set()
        lines: list[Line] = []
        for score, assign in sorted(beam_set, reverse=True):
            key = tuple(sorted(assign))
            if key in seen:
                continue
            seen.add(key)
            p = self.evaluate(base, side, key, enemy_setups)
            moves = [(pl.name, a, b) for pl, a, b in zip(own, current, assign) if a != b]
            lines.append(Line(assign, p, p - p_now, moves))
            if len(lines) >= top:
                break
        return lines, p_now


# ------------------------------------------------------------------------ book
def book(ds: Dataset, sit: Situation, side: str, horizon: float = 10.0, time_tol: float = 12.0,
         min_enemy_sim: float = 0.5, top: int = 8) -> pd.DataFrame:
    """Similar historical situations: what did this side do next, and did they win?"""
    enemy = "CT" if side == "T" else "T"
    st0 = sit.state
    sub = ds.meta[(ds.meta["map_name"] == sit.match.map_name)]
    rows = []
    for (mi, rnum), grp in sub.groupby(["match", "round_num"]):
        near = grp[(grp["t"] - sit.t).abs() <= time_tol]
        if near.empty:
            continue
        i = int((near["t"] - sit.t).abs().idxmin())
        st = ds.states[i]
        if st.phase != st0.phase or st.t_alive != st0.t_alive or st.ct_alive != st0.ct_alive:
            continue
        e_sim = _similarity(st.zones_of(enemy), st0.zones_of(enemy))
        if e_sim < min_enemy_sim:
            continue
        later = grp[grp["t"] >= float(ds.meta.at[i, "t"]) + horizon]
        j = int(later.index[0]) if not later.empty else i
        st_later = ds.states[j]
        win = float(grp["t_win"].iloc[0]) if side == "T" else 1.0 - float(grp["t_win"].iloc[0])
        rows.append({"match": mi, "round": rnum, "own_now": st.signature(side), "own_later": st_later.signature(side),
                     "enemy_sim": e_sim, "own_sim": _similarity(st.zones_of(side), st0.zones_of(side)), "won": win})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby("own_later").agg(n=("won", "size"), win_rate=("won", "mean"), enemy_sim=("enemy_sim", "mean"),
                                      own_sim=("own_sim", "mean")).reset_index()
    return agg.sort_values(["win_rate", "n"], ascending=False).head(top).reset_index(drop=True)
