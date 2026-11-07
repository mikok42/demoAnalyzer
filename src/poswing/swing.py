"""Round swing attribution.

The HLTV-style part (``swing``) follows the published Rating 3.0 description: the
team's win probability before and after each kill, credit split between the
finisher, damage dealers, flash assister and traded team-mate, deaths deducted from
the victim, bomb plants and round-end residuals shared by the team, kills in the
"unstoppable bomb" window ignored, and no negative swing for a kill.  It uses the
*baseline* (position-free) model so that it stays comparable with HLTV's numbers.

The positional part is new:

* ``pos_move``  – change in the team's positional edge (``p_pos - p_base``) between two
  samples with no event in between, credited to the players who changed zone.
* ``pos_death`` – ``-|c|`` where ``c`` is the victim's positional contribution at the
  moment of death.  Dying in a good spot wastes it, dying in a bad spot means the
  player chose a bad spot; both cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .constants import TRADE_WINDOW_S
from .data import MatchData
from .state import GameState, add_zones, build_states, plant_sites
from .winprob import WinProbModel, _sigmoid
from .zones import Zoner


@dataclass
class CreditShares:
    finisher: float = 0.5
    damage: float = 0.3
    flash: float = 0.1
    trade: float = 0.1


@dataclass
class SwingResult:
    events: pd.DataFrame  # one row per (event, player) credit
    players: pd.DataFrame  # aggregated per player
    rounds: pd.DataFrame  # per round: p0, outcome, team totals (zero-sum check)
    flags: pd.DataFrame = field(default_factory=pd.DataFrame)  # per kill: trade / failed trade / denial


def compute_swing(match: MatchData, model: WinProbModel, zoner: Zoner,
                  shares: CreditShares | None = None,
                  meta_states: tuple[pd.DataFrame, list[GameState]] | None = None) -> SwingResult:
    shares = shares or CreditShares()
    meta, states = meta_states if meta_states is not None else build_states(match, zoner)
    if meta.empty:
        raise ValueError("no states to compute swing on")
    samples_z = add_zones(match, zoner)
    sites = plant_sites(match)
    lookup = {(int(r), int(t)): i for i, (r, t) in enumerate(zip(meta["round_num"], meta["tick"]))}
    side_of = match.samples.groupby(["round_num", "steamid"])["team"].first().to_dict()
    roster = match.samples.groupby(["round_num", "team"])["steamid"].agg(lambda s: sorted(set(s))).to_dict()

    rows: list[dict] = []
    round_rows: list[dict] = []
    flag_rows: list[dict] = []

    for r in match.rounds.itertuples(index=False):
        rnum = int(r.round_num)
        idx_round = meta.index[meta["round_num"] == rnum].tolist()
        if not idx_round:
            continue
        ticks_round = meta.loc[idx_round, "tick"].tolist()
        first_state = states[idx_round[0]]
        p0 = model.p_t(first_state, positional=False)
        outcome = 1.0 if r.winner == "T" else 0.0

        kills = match.kills[match.kills["round_num"] == rnum].sort_values("tick")
        dmg = match.damages[match.damages["round_num"] == rnum]
        plants = match.bomb[(match.bomb["round_num"] == rnum) & (match.bomb["event"] == "plant")]
        events = [("kill", int(k.tick), k) for k in kills.itertuples(index=False)]
        events += [("plant", int(b.tick), b) for b in plants.itertuples(index=False)]
        events.sort(key=lambda e: e[1])

        running = first_state
        death_times: list[tuple[float, int, int | None, str]] = []  # (t, victim, killer, victim_team)
        kill_times: dict[int, list[float]] = {}
        added: dict[int, float] = {}
        added_post: dict[int, float] = {}
        sum_t = 0.0
        last_sync_tick = None

        for kind, tick, ev in events:
            before_idx = lookup.get((rnum, tick - 1))
            if before_idx is not None and last_sync_tick != tick:
                # re-sync positions/clock from the sampled state, keep already applied deaths
                sampled = states[before_idx]
                running = _merge_running(running, sampled)
                last_sync_tick = tick
            before = running
            if kind == "plant":
                after = replace(before, bomb_planted=True, time_left=40.0, bomb_site=sites.get(rnum))
                d_t = model.p_t(after, False) - model.p_t(before, False)
                alive_t = [sid for sid in roster.get((rnum, "T"), []) if _alive(sid, rnum, tick, match)]
                if not alive_t:
                    alive_t = roster.get((rnum, "T"), [])
                for sid in alive_t:
                    rows.append(_row(rnum, tick, float(ev.t), "plant", sid, "T", d_t / len(alive_t), "swing"))
                    added[sid] = added.get(sid, 0.0) + max(0.0, d_t / len(alive_t))
                alive_ct = [sid for sid in roster.get((rnum, "CT"), []) if _alive(sid, rnum, tick, match)] or roster.get((rnum, "CT"), [])
                for sid in alive_ct:
                    rows.append(_row(rnum, tick, float(ev.t), "plant", sid, "CT", -d_t / len(alive_ct), "swing"))
                sum_t += d_t
                running = after
                continue
            k = ev
            victim = int(k.victim)
            vteam = str(k.victim_team)
            attacker = int(k.attacker) if pd.notna(k.attacker) else None
            ateam = str(k.attacker_team) if pd.notna(k.attacker_team) else None
            vzone = zoner.assign_one(k.victim_place, k.victim_x, k.victim_y)
            after = _remove_player(before, vteam, vzone)
            killer_side = ateam if ateam in ("T", "CT") else ("CT" if vteam == "T" else "T")
            delta = model.p_side(after, killer_side, False) - model.p_side(before, killer_side, False)
            team_kill = attacker is not None and ateam == vteam

            # positional contribution of the victim at the moment of death
            c = _zone_contribution(model, before, vteam, vzone)
            rows.append(_row(rnum, tick, float(k.t), "death_pos", victim, vteam, -abs(c), "pos_death"))

            if team_kill:
                # the attacker's own team lost probability; charge it to the attacker
                d_own = model.p_side(after, vteam, False) - model.p_side(before, vteam, False)
                rows.append(_row(rnum, tick, float(k.t), "team_kill", attacker, vteam, min(0.0, d_own), "swing"))
                rows.append(_row(rnum, tick, float(k.t), "death", victim, vteam, 0.0, "swing"))
                sum_t += (min(0.0, d_own) if vteam == "T" else -min(0.0, d_own))
                running = after
                death_times.append((float(k.t), victim, attacker, vteam))
                continue

            delta = max(0.0, delta)  # no negative swing for a kill (Oct 2026 rule)
            # victim side
            rows.append(_row(rnum, tick, float(k.t), "death", victim, vteam, -delta, "swing"))
            sum_t += delta if killer_side == "T" else -delta

            if attacker is None:
                # world / suicide: the enemy team gains but nobody did anything
                for sid in roster.get((rnum, killer_side), []):
                    rows.append(_row(rnum, tick, float(k.t), "gift", sid, killer_side, delta / max(1, len(roster.get((rnum, killer_side), []))), "swing"))
                running = after
                death_times.append((float(k.t), victim, None, vteam))
                continue

            # trade bookkeeping -------------------------------------------------
            is_trade = False
            traded_mate: int | None = None
            for (t_d, v_d, k_d, team_d) in reversed(death_times):
                if float(k.t) - t_d > TRADE_WINDOW_S:
                    break
                if team_d == killer_side and k_d == victim:
                    is_trade, traded_mate = True, v_d
                    break
            failed_trade = False
            for (t_d, v_d, k_d, team_d) in reversed(death_times):
                if float(k.t) - t_d > TRADE_WINDOW_S:
                    break
                if team_d == vteam and k_d is not None and k_d != victim:
                    # a team-mate died moments ago and this player died instead of trading
                    failed_trade = not any(kt > t_d for kt in kill_times.get(victim, []) if kt <= float(k.t))
                    break
            kill_times.setdefault(attacker, []).append(float(k.t))
            denial = len([kt for kt in kill_times[attacker] if float(k.t) - kt <= TRADE_WINDOW_S]) >= 2
            flag_rows.append({"round_num": rnum, "tick": tick, "attacker": attacker, "victim": victim,
                              "is_trade": is_trade, "failed_trade": failed_trade, "trade_denial": denial})

            # credit split ------------------------------------------------------
            credits: dict[tuple[int, str], float] = {}

            def give(sid: int, part: str, amount: float):
                credits[(sid, part)] = credits.get((sid, part), 0.0) + amount

            give(attacker, "kill", delta * shares.finisher)
            vd = dmg[(dmg["victim"] == victim) & (dmg["attacker_team"] == killer_side) & (dmg["tick"] <= tick)]
            dmg_by = vd.groupby("attacker")["dmg"].sum()
            if dmg_by.sum() > 0:
                for sid, d in dmg_by.items():
                    give(int(sid), "damage", delta * shares.damage * float(d) / float(dmg_by.sum()))
            else:
                give(attacker, "damage", delta * shares.damage)
            flasher = int(k.assister) if (bool(k.assistedflash) and pd.notna(k.assister)) else None
            if flasher is not None and side_of.get((rnum, flasher)) == killer_side:
                give(flasher, "flash", delta * shares.flash)
            else:
                give(attacker, "kill", delta * shares.flash)
            if is_trade and traded_mate is not None:
                give(traded_mate, "traded", delta * shares.trade)
            else:
                give(attacker, "kill", delta * shares.trade)

            for (sid, part), amount in credits.items():
                rows.append(_row(rnum, tick, float(k.t), part, sid, killer_side, amount, "swing"))
                added[sid] = added.get(sid, 0.0) + amount
                if before.bomb_planted:
                    added_post[sid] = added_post.get(sid, 0.0) + amount

            running = after
            death_times.append((float(k.t), victim, attacker, vteam))

        # round end residual ----------------------------------------------------
        end_idx = lookup.get((rnum, int(r.end_tick)))
        last_state = _merge_running(running, states[end_idx]) if end_idx is not None else running
        p_last = model.p_t(last_state, False)
        # residual closes the round to the outcome after clamped kill credits (sum_t)
        residual_t = outcome - p0 - sum_t
        winner = r.winner
        loser = "CT" if winner == "T" else "T"
        win_roster = roster.get((rnum, winner), [])
        lose_roster = roster.get((rnum, loser), [])
        gain = abs(residual_t)
        t_end = float(last_state_time(r, match))
        defuses = match.bomb[(match.bomb["round_num"] == rnum) & (match.bomb["event"] == "defuse")]
        if winner == "CT" and not defuses.empty and pd.notna(defuses.iloc[0]["player"]):
            # the defuser converted the remaining probability; half goes to them directly
            defuser = int(defuses.iloc[0]["player"])
            rows.append(_row(rnum, int(r.end_tick), t_end, "defuse", defuser, "CT", gain * 0.5, "swing"))
            added[defuser] = added.get(defuser, 0.0) + gain * 0.5
            gain *= 0.5
        contrib_src = added_post if (last_state.bomb_planted and added_post) else added
        contrib = {sid: contrib_src.get(sid, 0.0) for sid in win_roster}
        tot = sum(v for v in contrib.values() if v > 0)
        for sid in win_roster:
            share = (max(0.0, contrib[sid]) / tot) if tot > 0 else (1.0 / max(1, len(win_roster)))
            rows.append(_row(rnum, int(r.end_tick), t_end, "round_end", sid, winner, gain * share, "swing"))
        gain = abs(residual_t)
        survivors = [sid for sid in lose_roster if _alive(sid, rnum, int(r.end_tick), match)]
        targets = survivors or lose_roster
        for sid in targets:
            rows.append(_row(rnum, int(r.end_tick), t_end, "round_end", sid, loser, -gain / max(1, len(targets)), "swing"))

        # positional movement swing ---------------------------------------------
        rows.extend(_movement_rows(samples_z, model, meta, states, idx_round, rnum))

        round_rows.append({"round_num": rnum, "winner": winner, "p0_t": p0, "p_last_t": p_last,
                           "residual_t": residual_t, "n_events": len(events)})

    events_df = pd.DataFrame(rows)
    players_df = _aggregate(events_df, match)
    return SwingResult(events=events_df, players=players_df, rounds=pd.DataFrame(round_rows),
                       flags=pd.DataFrame(flag_rows))


# ------------------------------------------------------------------------ helpers
def last_state_time(r, match: MatchData) -> float:
    return (int(r.end_tick) - int(r.freeze_end_tick)) / match.tick_rate


def _row(rnum, tick, t, kind, sid, team, delta, component) -> dict:
    return {"round_num": rnum, "tick": tick, "t": t, "kind": kind, "steamid": int(sid), "team": team,
            "delta": float(delta), "component": component}


def _alive(sid: int, rnum: int, tick: int, match: MatchData) -> bool:
    s = match.samples
    rows = s[(s["round_num"] == rnum) & (s["steamid"] == sid) & (s["tick"] <= tick)]
    if rows.empty:
        return True
    return bool(rows.iloc[-1]["is_alive"])


def _merge_running(running: GameState, sampled: GameState) -> GameState:
    """Take positions and clock from the sample, but never resurrect a player."""
    st = sampled
    if sampled.t_alive > running.t_alive:
        st = replace(st, t_alive=running.t_alive, t_zones=running.t_zones)
    if sampled.ct_alive > running.ct_alive:
        st = replace(st, ct_alive=running.ct_alive, ct_zones=running.ct_zones)
    return st


def _remove_player(st: GameState, side: str, zone: str) -> GameState:
    zones = list(st.zones_of(side))
    if zone in zones:
        zones.remove(zone)
    elif zones:
        zones.pop()
    if side == "T":
        return replace(st, t_alive=max(0, st.t_alive - 1), t_zones=tuple(zones))
    return replace(st, ct_alive=max(0, st.ct_alive - 1), ct_zones=tuple(zones))


def _zone_contribution(model: WinProbModel, st: GameState, side: str, zone: str) -> float:
    """Win-probability the player's zone adds for their side (sign matters)."""
    stacked = list(st.zones_of(side)).count(zone) > 1
    feat = f"{side}|{st.phase}|{zone}" + ("|extra" if stacked else "")
    j = model.positional.index.get(feat)
    if j is None or st.terminal_t_win() is not None:
        return 0.0
    # approximate local effect of this feature at the current baseline logit
    base = model.baseline.logit(st)
    coef = float(model.positional.coef[j])
    p0 = float(_sigmoid(base))
    p1 = float(_sigmoid(base + coef))
    diff = p1 - p0
    return diff if side == "T" else -diff


def _movement_rows(samples_z: pd.DataFrame, model, meta, states, idx_round, rnum) -> list[dict]:
    rows: list[dict] = []
    s = samples_z[samples_z["round_num"] == rnum]
    if s.empty:
        return rows
    zone_tab = s.pivot_table(index="tick", columns="steamid", values="zone", aggfunc="last")
    alive_tab = s.pivot_table(index="tick", columns="steamid", values="is_alive", aggfunc="last").reindex(columns=zone_tab.columns)
    team_of = s.groupby("steamid")["team"].first().reindex(zone_tab.columns).to_numpy(dtype=object)
    t_of = s.groupby("tick")["t"].first()
    sids = zone_tab.columns.to_numpy()
    tick_pos = {int(t): i for i, t in enumerate(zone_tab.index)}
    zones = zone_tab.to_numpy(dtype=object)
    alive = alive_tab.to_numpy(dtype=object) == True  # noqa: E712 - nullable bools
    sub = [states[i] for i in idx_round]
    edges_t = model.predict(sub, True) - model.predict(sub, False)
    is_event = meta.loc[idx_round, "is_event"].to_numpy()
    ticks = meta.loc[idx_round, "tick"].to_numpy()
    for k in range(len(idx_round) - 1):
        if is_event[k] or is_event[k + 1]:
            continue
        d_edge_t = float(edges_t[k + 1] - edges_t[k])
        if abs(d_edge_t) < 1e-9:
            continue
        ia, ib = tick_pos.get(int(ticks[k])), tick_pos.get(int(ticks[k + 1]))
        if ia is None or ib is None:
            continue
        both = alive[ia] & alive[ib]
        moved = both & (zones[ia] != zones[ib])
        if not moved.any():
            continue
        t_b = float(t_of.loc[int(ticks[k + 1])])
        for side, sign in (("T", 1.0), ("CT", -1.0)):
            side_movers = np.nonzero(moved & (team_of == side))[0]
            for j in side_movers:
                rows.append(_row(rnum, int(ticks[k + 1]), t_b, "move", int(sids[j]), side,
                                 sign * d_edge_t / len(side_movers), "pos_move"))
    return rows


def _aggregate(events: pd.DataFrame, match: MatchData) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rounds_played = match.samples.groupby("steamid")["round_num"].nunique()
    piv = events.pivot_table(index="steamid", columns="component", values="delta", aggfunc="sum", fill_value=0.0)
    for c in ("swing", "pos_move", "pos_death"):
        if c not in piv:
            piv[c] = 0.0
    kinds = events[events["component"] == "swing"].pivot_table(index="steamid", columns="kind", values="delta", aggfunc="sum", fill_value=0.0)
    out = pd.DataFrame(index=piv.index)
    out["name"] = [match.player_name(s) for s in out.index]
    out["rounds"] = rounds_played.reindex(out.index).fillna(0).astype(int)
    n = out["rounds"].replace(0, 1)
    out["swing_pct"] = piv["swing"] / n * 100
    pos = events[events["component"] == "swing"]
    out["added_pct"] = pos[pos["delta"] > 0].groupby("steamid")["delta"].sum().reindex(out.index).fillna(0) / n * 100
    out["deducted_pct"] = pos[pos["delta"] < 0].groupby("steamid")["delta"].sum().reindex(out.index).fillna(0) / n * 100
    for k in ("kill", "damage", "flash", "traded", "death", "plant", "defuse", "round_end", "team_kill"):
        out[f"{k}_pct"] = (kinds[k] if k in kinds else 0.0)
        out[f"{k}_pct"] = pd.Series(out[f"{k}_pct"], index=out.index).fillna(0) / n * 100
    out["pos_move_pct"] = piv["pos_move"] / n * 100
    out["pos_death_pct"] = piv["pos_death"] / n * 100
    out["pos_total_pct"] = out["pos_move_pct"] + out["pos_death_pct"]
    out["swing_pos_aware_pct"] = out["swing_pct"] + out["pos_total_pct"]
    return out.reset_index().sort_values("swing_pct", ascending=False)
