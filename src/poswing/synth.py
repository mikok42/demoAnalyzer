"""Synthetic matches with known ground truth, for tests and for trying the tool without demos.

The fake map is a chain of zones.  Some zones are "strong" for a side: a player
standing there wins duels more often.  This lets tests check that the model recovers
the zone values and that the engine recommends moving towards strong zones.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .constants import BOMB_TIME_S, ROUND_TIME_S
from .data import BLIND_COLUMNS, BOMB_COLUMNS, MatchData, empty_frame

ZONES = ["TSpawn", "TRamp", "Mid", "Connector", "ASite", "BSite", "CTSpawn", "Window"]
ADJ = {
    "TSpawn": ["TRamp"], "TRamp": ["TSpawn", "Mid", "BSite"], "Mid": ["TRamp", "Connector", "Window"],
    "Connector": ["Mid", "ASite"], "ASite": ["Connector", "CTSpawn"], "BSite": ["TRamp", "CTSpawn"],
    "CTSpawn": ["ASite", "BSite", "Window"], "Window": ["Mid", "CTSpawn"],
}
STRONG = {"T": {"TRamp": 0.25, "BSite": 0.15}, "CT": {"Window": 0.3, "Connector": 0.15}}
SITES = {"ASite", "BSite"}


def synth_match(n_rounds: int = 60, seed: int = 0, tick_rate: float = 64.0, source: str = "synthetic") -> MatchData:
    rng = random.Random(seed)
    t_ids = [1000 + i for i in range(5)]
    ct_ids = [2000 + i for i in range(5)]
    names = {sid: f"T{sid - 1000}" for sid in t_ids} | {sid: f"CT{sid - 2000}" for sid in ct_ids}
    rounds, kills, dmgs, bombs, samples = [], [], [], [], []
    tick = 1000
    for rnum in range(1, n_rounds + 1):
        half_swapped = rnum > n_rounds // 2
        t_side, ct_side = (ct_ids, t_ids) if half_swapped else (t_ids, ct_ids)
        team_of = {sid: "T" for sid in t_side} | {sid: "CT" for sid in ct_side}
        pos = {sid: "TSpawn" for sid in t_side} | {sid: "CTSpawn" for sid in ct_side}
        alive = {sid: True for sid in pos}
        freeze_end = tick
        t_equip = rng.choice([4000, 12000, 26000])
        ct_equip = rng.choice([4000, 12000, 26000])
        planted_at = None
        winner, reason, end_t = None, None, None
        t = 0.0
        while winner is None:
            # record sample
            for sid in pos:
                samples.append({"tick": freeze_end + int(t * tick_rate), "round_num": rnum, "t": t, "steamid": sid,
                                "team": team_of[sid], "is_alive": alive[sid], "x": float(ZONES.index(pos[sid]) * 500),
                                "y": float(hash((sid, rnum)) % 400), "z": 0.0, "place": pos[sid], "health": 100.0 if alive[sid] else 0.0,
                                "equip": (t_equip if team_of[sid] == "T" else ct_equip) / 5, "has_defuser": team_of[sid] == "CT",
                                "is_event": False})
            # move
            for sid in pos:
                if alive[sid] and rng.random() < 0.25:
                    pos[sid] = rng.choice(ADJ[pos[sid]])
            # duels between players in the same zone
            for z in ZONES:
                ts = [s for s in t_side if alive[s] and pos[s] == z]
                cts = [s for s in ct_side if alive[s] and pos[s] == z]
                if ts and cts and rng.random() < 0.6:
                    p_t = 0.5 + STRONG["T"].get(z, 0.0) - STRONG["CT"].get(z, 0.0)
                    p_t += 0.08 * (len(ts) - len(cts))
                    if rng.random() < p_t:
                        killer, victim = rng.choice(ts), rng.choice(cts)
                    else:
                        killer, victim = rng.choice(cts), rng.choice(ts)
                    alive[victim] = False
                    ktick = freeze_end + int((t + 0.5) * tick_rate)
                    kills.append({"tick": ktick, "round_num": rnum, "t": t + 0.5, "attacker": killer, "victim": victim,
                                  "assister": pd.NA, "assistedflash": False, "weapon": "ak47", "headshot": rng.random() < 0.4,
                                  "attacker_team": team_of[killer], "victim_team": team_of[victim], "attacker_place": z,
                                  "victim_place": z, "attacker_x": 0.0, "attacker_y": 0.0, "victim_x": 0.0, "victim_y": 0.0,
                                  "attacker_cat": "rifle1", "victim_cat": "rifle1"})
                    dmgs.append({"tick": ktick, "round_num": rnum, "t": t + 0.5, "attacker": killer, "victim": victim,
                                 "dmg": 100.0, "weapon": "ak47", "attacker_team": team_of[killer], "victim_team": team_of[victim]})
                    for sid in pos:
                        samples.append({"tick": ktick - 1, "round_num": rnum, "t": t + 0.5 - 1 / tick_rate, "steamid": sid,
                                        "team": team_of[sid], "is_alive": alive[sid] or sid == victim, "x": 0.0, "y": 0.0, "z": 0.0,
                                        "place": pos[sid], "health": 100.0, "equip": 0.0, "has_defuser": False, "is_event": True})
            # plant
            if planted_at is None and t > 20:
                planters = [s for s in t_side if alive[s] and pos[s] in SITES]
                if planters and rng.random() < 0.5:
                    planted_at = t + 0.7
                    btick = freeze_end + int(planted_at * tick_rate)
                    bombs.append({"tick": btick, "round_num": rnum, "t": planted_at, "event": "plant", "player": planters[0], "site": pos[planters[0]]})
            t += 1.0
            t_alive = sum(alive[s] for s in t_side)
            ct_alive = sum(alive[s] for s in ct_side)
            if ct_alive == 0:
                winner, reason = "T", "t_elimination"
            elif t_alive == 0 and planted_at is None:
                winner, reason = "CT", "ct_elimination"
            elif planted_at is not None and t - planted_at >= BOMB_TIME_S:
                winner, reason = "T", "bomb_exploded"
            elif planted_at is not None and t_alive == 0 and rng.random() < 0.5:
                winner, reason = "CT", "bomb_defused"
                bombs.append({"tick": freeze_end + int(t * tick_rate), "round_num": rnum, "t": t, "event": "defuse",
                              "player": next(s for s in ct_side if alive[s]), "site": None})
            elif planted_at is None and t >= ROUND_TIME_S:
                winner, reason = "CT", "time_expired"
            if winner is not None:
                end_t = t
        end_tick = freeze_end + int(end_t * tick_rate)
        for sid in pos:
            samples.append({"tick": end_tick, "round_num": rnum, "t": end_t, "steamid": sid, "team": team_of[sid],
                            "is_alive": alive[sid], "x": 0.0, "y": 0.0, "z": 0.0, "place": pos[sid], "health": 100.0,
                            "equip": 0.0, "has_defuser": False, "is_event": False})
        rounds.append({"round_num": rnum, "start_tick": freeze_end - 15 * 64, "freeze_end_tick": freeze_end, "end_tick": end_tick,
                       "winner": winner, "reason": reason,
                       "plant_tick": (freeze_end + int(planted_at * tick_rate)) if planted_at is not None else np.nan,
                       "plant_site": None, "t_team": "Alpha" if not half_swapped else "Bravo",
                       "ct_team": "Bravo" if not half_swapped else "Alpha", "t_equip": float(t_equip), "ct_equip": float(ct_equip),
                       "t_buy": _buy(t_equip), "ct_buy": _buy(ct_equip), "ct_has_kit": True})
        tick = end_tick + 10 * 64

    kills_df = pd.DataFrame(kills)
    for c in ("attacker", "victim", "assister"):
        kills_df[c] = kills_df[c].astype("Int64")
    samples_df = pd.DataFrame(samples).sort_values(["tick", "steamid"]).drop_duplicates(["tick", "steamid"]).reset_index(drop=True)
    players = pd.DataFrame({"steamid": list(names), "name": list(names.values())})
    return MatchData(source=source, map_name="de_synth", tick_rate=tick_rate, players=players, rounds=pd.DataFrame(rounds),
                     kills=kills_df, damages=pd.DataFrame(dmgs), blinds=empty_frame(BLIND_COLUMNS),
                     bomb=pd.DataFrame(bombs) if bombs else empty_frame(BOMB_COLUMNS), samples=samples_df)


def _buy(v: float) -> str:
    from .economy import buy_type

    return buy_type(v)
