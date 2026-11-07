"""Turn a CS2 ``.dem`` file (HLTV / GOTV / FACEIT) into ``MatchData`` using demoparser2."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ROUND_TIME_S, TEAM_NUM
from .data import (
    BLIND_COLUMNS, BOMB_COLUMNS, DAMAGE_COLUMNS, KILL_COLUMNS, ROUND_COLUMNS, SAMPLE_COLUMNS,
    MatchData, empty_frame,
)
from .economy import buy_type, equip_category

log = logging.getLogger(__name__)

ROUND_END_REASON = {
    1: "bomb_exploded", 7: "bomb_defused", 8: "ct_elimination", 9: "t_elimination",
    10: "draw", 12: "time_expired", 16: "game_start", 17: "t_surrender", 18: "ct_surrender",
}

TICK_PROPS = [
    "X", "Y", "Z", "last_place_name", "is_alive", "team_num", "health",
    "current_equip_value", "has_defuser", "team_clan_name", "inventory", "game_time",
]


def parse_demo(path: str | Path, sample_hz: float = 1.0) -> MatchData:
    """Parse one demo.  ``sample_hz`` is the regular position sampling frequency."""
    from demoparser2 import DemoParser  # imported lazily so the rest works without it

    path = str(path)
    parser = DemoParser(path)
    header = parser.parse_header()
    map_name = header.get("map_name", "unknown")

    events = dict(parser.parse_events([
        "round_freeze_end", "round_end", "round_start", "round_officially_ended",
        "round_announce_match_start", "bomb_planted", "bomb_defused", "bomb_exploded",
        "bomb_begindefuse", "player_blind",
    ]))

    freeze_ends = _ticks(events.get("round_freeze_end"))
    round_ends = events.get("round_end")
    if round_ends is None or round_ends.empty or not freeze_ends:
        raise ValueError(f"{path}: no rounds found")
    round_ends = round_ends.sort_values("tick")

    match_start = _ticks(events.get("round_announce_match_start"))
    match_start_tick = match_start[-1] if match_start else 0
    # the match-start announcement fires on the first live freeze-end; everything earlier is warmup
    freeze_ends = [t for t in freeze_ends if t >= match_start_tick]
    round_ends = round_ends[round_ends["tick"] >= match_start_tick]

    rounds = _build_rounds(freeze_ends, round_ends)
    if rounds.empty:
        raise ValueError(f"{path}: no complete rounds after match start")

    tick_rate = _tick_rate(parser, rounds)

    kills_raw = parser.parse_event(
        "player_death",
        player=["last_place_name", "X", "Y", "team_num", "inventory"],
        other=["is_bomb_planted"],
    )
    hurt_raw = parser.parse_event("player_hurt", player=["team_num"])

    kills = _kills(kills_raw, rounds, tick_rate)
    damages = _damages(hurt_raw, rounds, tick_rate)
    blinds = _blinds(events.get("player_blind"), rounds, tick_rate)
    bomb = _bomb(events, rounds, tick_rate)
    rounds = _attach_plants(rounds, bomb)

    sample_ticks = _sample_ticks(rounds, kills, bomb, tick_rate, sample_hz)
    ticks_df = parser.parse_ticks(TICK_PROPS, ticks=sample_ticks)
    samples, players, round_meta = _samples(ticks_df, rounds, kills, bomb, tick_rate)
    rounds = rounds.merge(round_meta, on="round_num", how="left")

    kills = _team_names_on(kills, samples)
    damages = _team_names_on(damages, samples)
    blinds = _team_names_on(blinds, samples)

    return MatchData(
        source=path, map_name=map_name, tick_rate=tick_rate, players=players,
        rounds=rounds[ROUND_COLUMNS], kills=kills[KILL_COLUMNS], damages=damages[DAMAGE_COLUMNS],
        blinds=blinds[BLIND_COLUMNS], bomb=bomb[BOMB_COLUMNS], samples=samples[SAMPLE_COLUMNS],
        extra={"header": header},
    )


# --------------------------------------------------------------------------- rounds
def _ticks(df: pd.DataFrame | None) -> list[int]:
    if df is None or df.empty or "tick" not in df:
        return []
    return sorted(int(t) for t in df["tick"].tolist())


def _build_rounds(freeze_ends: list[int], round_ends: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ends = round_ends["tick"].to_numpy()
    for i, fe in enumerate(freeze_ends):
        later = ends[ends > fe]
        if len(later) == 0:
            break
        end_tick = int(later[0])
        # a freeze-end followed by another freeze-end before any round-end means a restart
        if i + 1 < len(freeze_ends) and freeze_ends[i + 1] < end_tick:
            continue
        row = round_ends[round_ends["tick"] == end_tick].iloc[0]
        winner = _winner(row.get("winner"))
        if winner is None:
            continue
        reason = _reason(row.get("reason"))
        rows.append({
            "round_num": len(rows) + 1,
            "start_tick": int(fe),  # refined below
            "freeze_end_tick": int(fe),
            "end_tick": end_tick,
            "winner": winner,
            "reason": reason,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    prev_end = df["end_tick"].shift(1)
    df["start_tick"] = np.where(prev_end.notna(), prev_end.fillna(0).astype(int) + 1, np.maximum(0, df["freeze_end_tick"] - 20 * 64))
    return df


REASON_STRINGS = {
    "target_bombed": "bomb_exploded", "bomb_defused": "bomb_defused", "ct_killed": "t_elimination",
    "t_killed": "ct_elimination", "target_saved": "time_expired", "round_draw": "draw",
    "t_surrender": "t_surrender", "ct_surrender": "ct_surrender",
}


def _reason(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str):
        v = value.strip().lower()
        return REASON_STRINGS.get(v, v)
    try:
        return ROUND_END_REASON.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def _winner(value) -> str | None:
    """round_end.winner is a team number in some demos and 'CT'/'T' in others."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        v = value.strip().upper()
        return {"CT": "CT", "T": "T", "TERRORIST": "T", "2": "T", "3": "CT"}.get(v)
    try:
        return TEAM_NUM.get(int(value))
    except (TypeError, ValueError):
        return None


def _tick_rate(parser, rounds: pd.DataFrame) -> float:
    r = rounds.iloc[0]
    a, b = int(r["freeze_end_tick"]), int(r["end_tick"])
    try:
        df = parser.parse_ticks(["game_time"], ticks=[a, b])
        ga = df.loc[df["tick"] == a, "game_time"].iloc[0]
        gb = df.loc[df["tick"] == b, "game_time"].iloc[0]
        rate = (b - a) / (gb - ga)
        if 30 < rate < 200:
            return float(round(rate))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("tick rate detection failed (%s); assuming 64", exc)
    return 64.0


def assign_round(ticks: pd.Series, rounds: pd.DataFrame) -> pd.Series:
    """Map ticks to round numbers (NaN when outside every live round)."""
    starts = rounds["freeze_end_tick"].to_numpy()
    ends = rounds["end_tick"].to_numpy()
    nums = rounds["round_num"].to_numpy()
    idx = np.searchsorted(starts, ticks.to_numpy(), side="right") - 1
    out = np.full(len(ticks), np.nan)
    ok = idx >= 0
    ok_idx = idx[ok]
    valid = ticks.to_numpy()[ok] <= ends[ok_idx]
    out_ok = np.where(valid, nums[ok_idx], np.nan)
    out[ok] = out_ok
    return pd.Series(out, index=ticks.index)


def _round_time(df: pd.DataFrame, rounds: pd.DataFrame, tick_rate: float) -> pd.Series:
    fe = rounds.set_index("round_num")["freeze_end_tick"]
    return (df["tick"] - df["round_num"].map(fe)) / tick_rate


# --------------------------------------------------------------------------- events
def _col(raw: pd.DataFrame, name: str, default=None) -> pd.Series:
    if name in raw:
        return raw[name]
    return pd.Series([default] * len(raw), index=raw.index, dtype="object")


def _sid(s: pd.Series) -> pd.Series:
    """Steam ids arrive as strings from events and ints from ticks; normalise to nullable Int64.

    Never go through float: 17-digit ids lose their last digits in float64.
    """
    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return pd.NA
    return pd.Series([conv(v) for v in s], index=s.index, dtype="Int64")


def _kills(raw: pd.DataFrame, rounds: pd.DataFrame, tick_rate: float) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_frame(KILL_COLUMNS)
    df = pd.DataFrame({
        "tick": raw["tick"].astype(int),
        "attacker": _sid(_col(raw, "attacker_steamid")),
        "victim": _sid(raw["user_steamid"]),
        "assister": _sid(_col(raw, "assister_steamid")),
        "assistedflash": _col(raw, "assistedflash", False).fillna(False).astype(bool),
        "weapon": _col(raw, "weapon", ""),
        "headshot": _col(raw, "headshot", False).fillna(False).astype(bool),
        "attacker_team": _col(raw, "attacker_team_num").map(TEAM_NUM),
        "victim_team": _col(raw, "user_team_num").map(TEAM_NUM),
        "attacker_place": _col(raw, "attacker_last_place_name"),
        "victim_place": _col(raw, "user_last_place_name"),
        "attacker_x": _col(raw, "attacker_X"), "attacker_y": _col(raw, "attacker_Y"),
        "victim_x": _col(raw, "user_X"), "victim_y": _col(raw, "user_Y"),
        "attacker_cat": _col(raw, "attacker_inventory").map(equip_category),
        "victim_cat": _col(raw, "user_inventory").map(equip_category),
    })
    df["round_num"] = assign_round(df["tick"], rounds)
    df = df[df["round_num"].notna()].copy()
    df["round_num"] = df["round_num"].astype(int)
    df["t"] = _round_time(df, rounds, tick_rate)
    df = df[df["victim"].notna()]
    return df.sort_values("tick").reset_index(drop=True)


def _damages(raw: pd.DataFrame, rounds: pd.DataFrame, tick_rate: float) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_frame(DAMAGE_COLUMNS)
    df = pd.DataFrame({
        "tick": raw["tick"].astype(int),
        "attacker": _sid(_col(raw, "attacker_steamid")),
        "victim": _sid(raw["user_steamid"]),
        "dmg": raw["dmg_health"].astype(float),
        "weapon": _col(raw, "weapon", ""),
        "attacker_team": _col(raw, "attacker_team_num").map(TEAM_NUM),
        "victim_team": _col(raw, "user_team_num").map(TEAM_NUM),
    })
    df["round_num"] = assign_round(df["tick"], rounds)
    df = df[df["round_num"].notna() & df["attacker"].notna()].copy()
    df["round_num"] = df["round_num"].astype(int)
    df["t"] = _round_time(df, rounds, tick_rate)
    return df.sort_values("tick").reset_index(drop=True)


def _blinds(raw: pd.DataFrame | None, rounds: pd.DataFrame, tick_rate: float) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_frame(BLIND_COLUMNS)
    df = pd.DataFrame({
        "tick": raw["tick"].astype(int),
        "attacker": _sid(_col(raw, "attacker_steamid")),
        "victim": _sid(raw["user_steamid"]),
        "duration": pd.to_numeric(_col(raw, "blind_duration", 0.0), errors="coerce").fillna(0.0).astype(float),
    })
    df["round_num"] = assign_round(df["tick"], rounds)
    df = df[df["round_num"].notna()].copy()
    df["round_num"] = df["round_num"].astype(int)
    df["t"] = _round_time(df, rounds, tick_rate)
    df["attacker_team"] = None
    df["victim_team"] = None
    return df.sort_values("tick").reset_index(drop=True)


def _bomb(events: dict, rounds: pd.DataFrame, tick_rate: float) -> pd.DataFrame:
    rows = []
    for name, label in [("bomb_planted", "plant"), ("bomb_defused", "defuse"),
                        ("bomb_exploded", "explode"), ("bomb_begindefuse", "begin_defuse")]:
        df = events.get(name)
        if df is None or df.empty:
            continue
        for row in df.itertuples(index=False):
            rows.append({
                "tick": int(row.tick), "event": label,
                "player": getattr(row, "user_steamid", None),
                "site": getattr(row, "site", None),
            })
    if not rows:
        return empty_frame(BOMB_COLUMNS)
    out = pd.DataFrame(rows)
    out["player"] = _sid(out["player"])
    out["round_num"] = assign_round(out["tick"], rounds)
    out = out[out["round_num"].notna()].copy()
    out["round_num"] = out["round_num"].astype(int)
    out["t"] = _round_time(out, rounds, tick_rate)
    return out.sort_values("tick").reset_index(drop=True)


def _attach_plants(rounds: pd.DataFrame, bomb: pd.DataFrame) -> pd.DataFrame:
    plants = bomb[bomb["event"] == "plant"].drop_duplicates("round_num", keep="first") if not bomb.empty else bomb
    rounds = rounds.copy()
    rounds["plant_tick"] = rounds["round_num"].map(plants.set_index("round_num")["tick"]) if len(plants) else np.nan
    rounds["plant_site"] = rounds["round_num"].map(plants.set_index("round_num")["site"]) if len(plants) else None
    return rounds


# --------------------------------------------------------------------------- samples
def _sample_ticks(rounds: pd.DataFrame, kills: pd.DataFrame, bomb: pd.DataFrame, tick_rate: float, hz: float) -> list[int]:
    step = max(1, int(round(tick_rate / hz)))
    ticks: set[int] = set()
    for r in rounds.itertuples(index=False):
        ticks.update(range(int(r.freeze_end_tick), int(r.end_tick) + 1, step))
        ticks.add(int(r.end_tick))
    # state exactly before each event: the sample one tick earlier
    for ev_ticks in (kills["tick"], bomb["tick"]):
        for t in ev_ticks.tolist():
            ticks.add(int(t) - 1)
            ticks.add(int(t))
    return sorted(t for t in ticks if t >= 0)


def _samples(ticks_df: pd.DataFrame, rounds: pd.DataFrame, kills: pd.DataFrame, bomb: pd.DataFrame, tick_rate: float):
    df = ticks_df.rename(columns={"X": "x", "Y": "y", "Z": "z", "last_place_name": "place",
                                  "current_equip_value": "equip"})
    df["round_num"] = assign_round(df["tick"], rounds)
    df = df[df["round_num"].notna()].copy()
    df["round_num"] = df["round_num"].astype(int)
    df["team"] = df["team_num"].map(TEAM_NUM)
    df = df[df["team"].notna()].copy()
    df["t"] = _round_time(df, rounds, tick_rate)
    df["is_alive"] = df["is_alive"].fillna(False).astype(bool)
    df["has_defuser"] = df["has_defuser"].fillna(False).astype(bool)
    df["health"] = df["health"].fillna(0).astype(float)
    df["equip"] = df["equip"].fillna(0).astype(float)
    df["steamid"] = df["steamid"].astype("int64")
    event_ticks = set(kills["tick"].tolist()) | set(bomb["tick"].tolist())
    event_ticks |= {t - 1 for t in event_ticks}
    df["is_event"] = df["tick"].isin(event_ticks)

    players = df.groupby("steamid")["name"].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else "?").reset_index()

    # round-level economy and team names from the freeze-end snapshot
    meta_rows = []
    for r in rounds.itertuples(index=False):
        snap = df[(df["round_num"] == r.round_num)]
        if snap.empty:
            continue
        first_tick = snap["tick"].min()
        snap = snap[snap["tick"] == first_tick]
        t_side = snap[snap["team"] == "T"]
        ct_side = snap[snap["team"] == "CT"]
        t_equip = float(t_side["equip"].sum())
        ct_equip = float(ct_side["equip"].sum())
        meta_rows.append({
            "round_num": r.round_num,
            "t_team": _clan(t_side),
            "ct_team": _clan(ct_side),
            "t_roster": frozenset(t_side["steamid"].tolist()),
            "ct_roster": frozenset(ct_side["steamid"].tolist()),
            "t_equip": t_equip, "ct_equip": ct_equip,
            "t_buy": buy_type(t_equip), "ct_buy": buy_type(ct_equip),
            "ct_has_kit": bool(ct_side["has_defuser"].any()),
        })
    meta = _stabilise_team_names(pd.DataFrame(meta_rows))
    return df, players, meta.drop(columns=["t_roster", "ct_roster"])


def _clan(snap: pd.DataFrame) -> str | None:
    if "team_clan_name" not in snap:
        return None
    vals = snap["team_clan_name"].dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    return vals.mode().iloc[0] if len(vals) else None


def _stabilise_team_names(meta: pd.DataFrame) -> pd.DataFrame:
    """Follow rosters across half swaps; demos without clan tags get 'Team A'/'Team B'."""
    if meta.empty:
        return meta
    meta = meta.copy()
    roster_a = meta.at[0, "t_roster"]
    roster_b = meta.at[0, "ct_roster"]
    name_a = meta.at[0, "t_team"] or "Team A"
    name_b = meta.at[0, "ct_team"] or "Team B"
    if name_a == name_b:
        name_a, name_b = f"{name_a} (A)", f"{name_b} (B)"
    for i in range(len(meta)):
        t_r, ct_r = meta.at[i, "t_roster"], meta.at[i, "ct_roster"]
        t_is_a = len(t_r & roster_a) >= len(t_r & roster_b)
        meta.at[i, "t_team"] = name_a if t_is_a else name_b
        meta.at[i, "ct_team"] = name_b if t_is_a else name_a
        # rosters drift with stand-ins; follow the latest known line-up
        roster_a, roster_b = (t_r, ct_r) if t_is_a else (ct_r, t_r)
    return meta


def _team_names_on(df: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Fill missing attacker/victim side labels from the nearest position sample."""
    if df.empty:
        return df
    need = df["attacker_team"].isna() | df["victim_team"].isna()
    if not need.any():
        return df
    side = samples.groupby(["round_num", "steamid"])["team"].first()
    for col, key in (("attacker_team", "attacker"), ("victim_team", "victim")):
        missing = df[col].isna() & df[key].notna()
        if missing.any():
            keys = list(zip(df.loc[missing, "round_num"].astype(int), df.loc[missing, key].astype("int64")))
            df.loc[missing, col] = [side.get(k) for k in keys]
    return df
