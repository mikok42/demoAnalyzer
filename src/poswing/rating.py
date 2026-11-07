"""Player box score with the Rating 3.0 sub-metrics.

HLTV has not published the Rating 3.0 formula, only its ingredients (Kills,
Damage, Survival, KAST, Multi-kills, Round Swing - all eco-adjusted).  This module
computes every ingredient from the demo and combines them into a clearly labelled
*approximation* (``rating_approx``) built on the widely reverse-engineered Rating 2.0
regression.  Use the ingredients for analysis; use the composite for ordering.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .constants import ASSIST_DAMAGE, TRADE_WINDOW_S
from .data import MatchData
from .economy import DuelTable
from .swing import SwingResult

UTILITY_WEAPONS = {"hegrenade", "inferno", "molotov", "incgrenade", "decoy", "flashbang", "smokegrenade"}


def box_score(match: MatchData, swing: SwingResult | None = None, duels: DuelTable | None = None) -> pd.DataFrame:
    duels = duels or DuelTable().fit(match.kills)
    kills = match.kills
    dmg = match.damages
    rounds = match.rounds
    n_rounds_total = len(rounds)
    players = sorted(set(match.samples["steamid"]))
    rounds_played = match.samples.groupby("steamid")["round_num"].nunique()

    stats = {sid: defaultdict(float) for sid in players}
    for sid in players:
        stats[sid]["rounds"] = int(rounds_played.get(sid, 0))

    # damage ------------------------------------------------------------------
    enemy_dmg = dmg[dmg["attacker_team"] != dmg["victim_team"]]
    for sid, d in enemy_dmg.groupby("attacker")["dmg"].sum().items():
        stats[int(sid)]["damage"] = float(d)
    util = enemy_dmg[enemy_dmg["weapon"].isin(UTILITY_WEAPONS)]
    for sid, d in util.groupby("attacker")["dmg"].sum().items():
        stats[int(sid)]["utility_damage"] = float(d)

    # kills / deaths / assists -------------------------------------------------
    kast_rounds: dict[int, set] = defaultdict(set)
    for rnum, grp in kills.groupby("round_num"):
        grp = grp.sort_values("tick")
        first = True
        kills_in_round: dict[int, int] = defaultdict(int)
        deaths: list[tuple[float, int, int | None, str]] = []
        kill_times: dict[int, list[float]] = defaultdict(list)
        round_dmg = dmg[dmg["round_num"] == rnum]
        for k in grp.itertuples(index=False):
            victim = int(k.victim)
            attacker = int(k.attacker) if pd.notna(k.attacker) else None
            vteam, ateam = str(k.victim_team), (str(k.attacker_team) if pd.notna(k.attacker_team) else None)
            stats[victim]["deaths"] += 1
            if attacker is not None and ateam != vteam:
                a = stats[attacker]
                a["kills"] += 1
                kills_in_round[attacker] += 1
                kast_rounds[attacker].add(rnum)
                if bool(k.headshot):
                    a["headshots"] += 1
                a["kills_adj"] += duels.kill_value(ateam, k.attacker_cat, k.victim_cat)
                stats[victim]["deaths_adj"] += duels.death_value(vteam, k.victim_cat, k.attacker_cat)
                if first:
                    a["opening_kills"] += 1
                    stats[victim]["opening_deaths"] += 1
                # assists: 40+ damage on the victim or a flash assist
                vd = round_dmg[(round_dmg["victim"] == victim) & (round_dmg["tick"] <= k.tick)
                               & (round_dmg["attacker_team"] == ateam) & (round_dmg["attacker"] != attacker)]
                for sid, d in vd.groupby("attacker")["dmg"].sum().items():
                    if d >= ASSIST_DAMAGE:
                        stats[int(sid)]["assists"] += 1
                        kast_rounds[int(sid)].add(rnum)
                if bool(k.assistedflash) and pd.notna(k.assister) and int(k.assister) != attacker:
                    stats[int(k.assister)]["flash_assists"] += 1
                    kast_rounds[int(k.assister)].add(rnum)
                # trades
                for (t_d, v_d, k_d, team_d) in reversed(deaths):
                    if float(k.t) - t_d > TRADE_WINDOW_S:
                        break
                    if team_d == ateam and k_d == victim:
                        a["trade_kills"] += 1
                        stats[v_d]["traded_deaths"] += 1
                        kast_rounds[v_d].add(rnum)
                        break
                for (t_d, v_d, k_d, team_d) in reversed(deaths):
                    if float(k.t) - t_d > TRADE_WINDOW_S:
                        break
                    if team_d == vteam and k_d is not None and k_d != victim:
                        if not any(t_d < kt <= float(k.t) for kt in kill_times[victim]):
                            stats[victim]["failed_trades"] += 1
                        break
                kill_times[attacker].append(float(k.t))
                if len([kt for kt in kill_times[attacker] if float(k.t) - kt <= TRADE_WINDOW_S]) >= 2:
                    a["trade_denials"] += 1
            elif attacker is not None and ateam == vteam:
                stats[attacker]["team_kills"] += 1
            first = False
            deaths.append((float(k.t), victim, attacker, vteam))
        for sid, n in kills_in_round.items():
            if n >= 2:
                stats[sid]["multikill_rounds"] += 1
            stats[sid][f"{min(n, 5)}k"] += 1

    # survival + clutches ------------------------------------------------------
    for r in rounds.itertuples(index=False):
        rnum = int(r.round_num)
        s = match.samples[(match.samples["round_num"] == rnum)]
        if s.empty:
            continue
        last = s[s["tick"] == s["tick"].max()]
        for row in last.itertuples(index=False):
            if bool(row.is_alive):
                stats[int(row.steamid)]["survived"] += 1
                kast_rounds[int(row.steamid)].add(rnum)
        _clutches(match, rnum, r.winner, stats)

    swing_map = {}
    if swing is not None and not swing.players.empty:
        swing_map = swing.players.set_index("steamid").to_dict("index")

    rows = []
    for sid in players:
        st = stats[sid]
        n = max(1, int(st["rounds"]))
        kpr, dpr, adr = st["kills"] / n, st["deaths"] / n, st["damage"] / n
        apr = st["assists"] / n
        kast = 100.0 * len(kast_rounds[sid]) / n
        kpr_adj, dpr_adj = st["kills_adj"] / n, st["deaths_adj"] / n
        mk = (st["2k"] * 4 + st["3k"] * 9 + st["4k"] * 16 + st["5k"] * 25) / n
        sw = swing_map.get(sid, {})
        swing_pct = float(sw.get("swing_pct", 0.0))
        impact2 = 2.13 * kpr + 0.42 * apr - 0.41
        rating2 = 0.0073 * kast + 0.3591 * kpr - 0.5329 * dpr + 0.2372 * impact2 + 0.0032 * adr + 0.1587
        impact3 = 1.4 * kpr_adj + 0.42 * apr + 0.25 * mk - 0.41 + 6.0 * swing_pct / 100.0
        rating3 = 0.0073 * kast + 0.3591 * kpr_adj - 0.5329 * dpr_adj + 0.2372 * impact3 + 0.0032 * adr + 0.1587
        rows.append({
            "steamid": sid, "name": match.player_name(sid), "rounds": int(st["rounds"]),
            "kills": int(st["kills"]), "deaths": int(st["deaths"]), "assists": int(st["assists"]),
            "flash_assists": int(st["flash_assists"]), "hs_pct": 100.0 * st["headshots"] / max(1, st["kills"]),
            "kpr": kpr, "dpr": dpr, "adr": adr, "kast": kast,
            "kpr_adj": kpr_adj, "dpr_adj": dpr_adj,
            "opening_kills": int(st["opening_kills"]), "opening_deaths": int(st["opening_deaths"]),
            "multikill_rounds": int(st["multikill_rounds"]), "2k": int(st["2k"]), "3k": int(st["3k"]),
            "4k": int(st["4k"]), "5k": int(st["5k"]), "multikill_rating": mk,
            "clutches_won": int(st["clutches_won"]), "clutch_attempts": int(st["clutch_attempts"]),
            "trade_kills": int(st["trade_kills"]), "traded_deaths": int(st["traded_deaths"]),
            "failed_trades": int(st["failed_trades"]), "trade_denials": int(st["trade_denials"]),
            "team_kills": int(st["team_kills"]), "utility_damage": st["utility_damage"],
            "survived": int(st["survived"]),
            "swing_pct": swing_pct, "added_pct": float(sw.get("added_pct", 0.0)),
            "deducted_pct": float(sw.get("deducted_pct", 0.0)),
            "pos_move_pct": float(sw.get("pos_move_pct", 0.0)), "pos_death_pct": float(sw.get("pos_death_pct", 0.0)),
            "pos_total_pct": float(sw.get("pos_total_pct", 0.0)),
            "rating2_approx": rating2, "rating_approx": rating3,
        })
    out = pd.DataFrame(rows).sort_values("rating_approx", ascending=False).reset_index(drop=True)
    out.attrs["n_rounds"] = n_rounds_total
    return out


def _clutches(match: MatchData, rnum: int, winner: str, stats) -> None:
    """1vX situations: last player alive on a side with X>=1 enemies alive."""
    s = match.samples[match.samples["round_num"] == rnum]
    if s.empty:
        return
    alive = s.pivot_table(index="tick", columns="steamid", values="is_alive", aggfunc="last").ffill().fillna(False).astype(bool)
    team = s.groupby("steamid")["team"].first()
    for side in ("T", "CT"):
        enemy = "CT" if side == "T" else "T"
        own_cols = [c for c in alive.columns if team.get(c) == side]
        foe_cols = [c for c in alive.columns if team.get(c) == enemy]
        own_n = alive[own_cols].sum(axis=1).to_numpy()
        foe_n = alive[foe_cols].sum(axis=1).to_numpy()
        hits = np.nonzero((own_n == 1) & (foe_n >= 1))[0]
        if len(hits) == 0:
            continue
        row = alive.iloc[hits[0]]
        sid = int(next(c for c in own_cols if row[c]))
        stats[sid]["clutch_attempts"] += 1
        if winner == side:
            stats[sid]["clutches_won"] += 1
