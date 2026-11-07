"""Load matches from the ESTA dataset (Esports Trajectories and Actions, Xenopoulos &
Silva 2022: 1,558 professional CS:GO demos parsed to JSON at 2 Hz) into ``MatchData``.

ESTA frames carry raw coordinates but no nav-mesh place names, so places are looked
up in the CS:GO nav meshes shipped with awpy 1.3.1 (``navdata/csgo_nav_areas.csv.gz``)
and renamed to the CS2 callouts where the two games differ, so that a model trained
on ESTA speaks the same zone language as a parsed CS2 demo.
"""
from __future__ import annotations

import gzip
import json
import lzma
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    BLIND_COLUMNS, BOMB_COLUMNS, DAMAGE_COLUMNS, KILL_COLUMNS, ROUND_COLUMNS, SAMPLE_COLUMNS,
    MatchData, empty_frame,
)
from .economy import buy_type, equip_category

NAV_FILE = Path(__file__).parent / "navdata" / "csgo_nav_areas.csv.gz"

# CS:GO nav place -> CS2 last_place_name (derived by overlaying CS2 demo samples on the
# CS:GO nav mesh; coordinates of the two Mirage versions coincide)
PLACE_RENAMES = {
    "de_mirage": {"Tunnel": "Underpass", "TunnelStairs": "Underpass", "TicketBooth": "CTSpawn", "Balcony": "Apartments"},
}
WEAPON_RENAMES = {"M4A1": "M4A1-S"}
REASONS = {
    "TerroristsWin": "t_elimination", "CTWin": "ct_elimination", "TargetBombed": "bomb_exploded",
    "BombDefused": "bomb_defused", "TargetSaved": "time_expired", "Draw": "draw",
}
BOMB_ACTIONS = {"plant": "plant", "defuse": "defuse", "defuse_start": "begin_defuse"}


# ------------------------------------------------------------------- nav places
class NavPlaces:
    """Point -> nav place name using the axis-aligned nav areas of one map."""

    def __init__(self, map_name: str):
        self.map_name = map_name
        with gzip.open(NAV_FILE, "rt") as fh:
            nav = pd.read_csv(fh)
        nav = nav[(nav["mapName"] == map_name) & nav["areaName"].notna()]
        if nav.empty:
            raise ValueError(f"no nav areas for {map_name} in {NAV_FILE}")
        renames = PLACE_RENAMES.get(map_name, {})
        self.names = np.array([renames.get(n, n) for n in nav["areaName"]], dtype=object)
        pad = 8.0
        self.x0 = np.minimum(nav["northWestX"], nav["southEastX"]).to_numpy() - pad
        self.x1 = np.maximum(nav["northWestX"], nav["southEastX"]).to_numpy() + pad
        self.y0 = np.minimum(nav["northWestY"], nav["southEastY"]).to_numpy() - pad
        self.y1 = np.maximum(nav["northWestY"], nav["southEastY"]).to_numpy() + pad
        self.cx = (self.x0 + self.x1) / 2
        self.cy = (self.y0 + self.y1) / 2
        self.cz = ((nav["northWestZ"] + nav["southEastZ"]) / 2).to_numpy()

    def assign(self, x, y, z) -> np.ndarray:
        x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
        out = np.empty(len(x), dtype=object)
        for s in range(0, len(x), 4096):
            xs, ys, zs = x[s:s + 4096, None], y[s:s + 4096, None], z[s:s + 4096, None]
            inside = (xs >= self.x0) & (xs <= self.x1) & (ys >= self.y0) & (ys <= self.y1)
            dz = np.abs(zs - self.cz)
            score = np.where(inside, dz, np.inf)
            best = score.argmin(1)
            miss = ~np.isfinite(score[np.arange(len(best)), best])
            if miss.any():
                d2 = (xs[miss] - self.cx) ** 2 + (ys[miss] - self.cy) ** 2 + 0.25 * (zs[miss] - self.cz) ** 2
                best[miss] = d2.argmin(1)
            out[s:s + 4096] = self.names[best]
        return out


@lru_cache(maxsize=None)
def nav_places(map_name: str) -> NavPlaces:
    return NavPlaces(map_name)


# ------------------------------------------------------------------------ loader
def read_esta(path: str | Path) -> dict:
    path = Path(path)
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def esta_map(path: str | Path) -> str:
    """Peek at the map name without decoding the whole file."""
    path = Path(path)
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rb") as fh:
        head = fh.read(400).decode("utf-8", "ignore")
    i = head.find('"mapName"')
    return head[i:].split('"')[3] if i >= 0 else "unknown"


def load_esta(path: str | Path, sample_hz: float = 1.0) -> MatchData:
    """Convert one ESTA match (``.json`` or ``.json.xz``) into ``MatchData``.

    ESTA frames come at 2 Hz; ``sample_hz`` thins them (kill frames are always kept).
    """
    d = read_esta(path)
    map_name = d["mapName"]
    tick_rate = 128.0 if d.get("tickRate", 128) > 100 else 64.0
    places = nav_places(map_name)
    keep_every = max(1, int(round(2.0 / sample_hz)))

    rounds, kills, dmgs, blinds, bomb, samples, players = [], [], [], [], [], [], {}
    for rd in d["gameRounds"]:
        if rd.get("isWarmup") or rd.get("winningSide") not in ("CT", "T") or not rd.get("frames"):
            continue
        fe, end = int(rd["freezeTimeEndTick"]), int(rd["endTick"])
        if end <= fe:
            continue
        rnum = len(rounds) + 1
        reason = REASONS.get(rd.get("roundEndReason"), str(rd.get("roundEndReason", "")).lower())
        for side_key in ("ctSide", "tSide"):
            for p in rd.get(side_key, {}).get("players", []) or []:
                players[int(p["steamID"])] = p["playerName"]

        kill_ticks = sorted(int(k["tick"]) for k in rd["kills"])
        frames = [f for f in rd["frames"] if fe <= int(f["tick"]) <= end]
        frame_ticks = np.array([int(f["tick"]) for f in frames])
        # frames right before and at a kill are "event" samples
        event_idx: set[int] = set()
        for kt in kill_ticks:
            j = int(np.searchsorted(frame_ticks, kt, side="left"))
            if j - 1 >= 0:
                event_idx.add(j - 1)
            if j < len(frames) and frame_ticks[j] == kt:
                event_idx.add(j)
        inv_hist: dict[int, tuple[list[int], list[str]]] = {}
        ct_kit = False
        for i, f in enumerate(frames):
            is_event = i in event_idx or bool(f.get("isKillFrame"))
            if i % keep_every and not is_event:
                continue
            tick = int(f["tick"])
            for side in ("t", "ct"):
                block = f.get(side) or {}
                for p in block.get("players") or []:
                    sid = int(p["steamID"])
                    inv = [WEAPON_RENAMES.get(w["weaponName"], w["weaponName"]) for w in (p.get("inventory") or [])]
                    hist = inv_hist.setdefault(sid, ([], []))
                    hist[0].append(tick)
                    hist[1].append(equip_category(inv))
                    if side == "ct" and p.get("hasDefuse") and i < 3:
                        ct_kit = True
                    samples.append((tick, rnum, (tick - fe) / tick_rate, sid, side.upper(), bool(p.get("isAlive")),
                                    p["x"], p["y"], p["z"], None, float(p.get("hp") or 0),
                                    float(p.get("equipmentValue") or 0), bool(p.get("hasDefuse")), is_event))

        def cat_at(sid, tick, before: bool) -> str:
            hist = inv_hist.get(sid)
            if not hist:
                return "pistol"
            j = int(np.searchsorted(hist[0], tick, side="left" if before else "right")) - 1
            return hist[1][max(j, 0)]

        for k in rd["kills"]:
            att, vic = k.get("attackerSteamID"), k.get("victimSteamID")
            if vic is None:
                continue
            tick = int(k["tick"])
            assister, flash = k.get("assisterSteamID"), k.get("flashThrowerSteamID")
            kills.append({
                "tick": tick, "round_num": rnum, "t": (tick - fe) / tick_rate,
                "attacker": att, "victim": vic,
                "assister": assister if assister is not None else flash,
                "assistedflash": assister is None and flash is not None,
                "weapon": WEAPON_RENAMES.get(k.get("weapon"), k.get("weapon")), "headshot": bool(k.get("isHeadshot")),
                "attacker_team": k.get("attackerSide"), "victim_team": k.get("victimSide"),
                "attacker_x": k.get("attackerX"), "attacker_y": k.get("attackerY"), "attacker_z": k.get("attackerZ"),
                "victim_x": k.get("victimX"), "victim_y": k.get("victimY"), "victim_z": k.get("victimZ"),
                "attacker_cat": cat_at(att, tick, False) if att is not None else "pistol",
                "victim_cat": cat_at(vic, tick, True),
            })
        for dm in rd["damages"]:
            if dm.get("attackerSteamID") is None or dm.get("victimSteamID") is None:
                continue
            tick = int(dm["tick"])
            dmgs.append({"tick": tick, "round_num": rnum, "t": (tick - fe) / tick_rate,
                         "attacker": dm["attackerSteamID"], "victim": dm["victimSteamID"],
                         "dmg": float(dm.get("hpDamageTaken") or 0), "weapon": dm.get("weapon"),
                         "attacker_team": dm.get("attackerSide"), "victim_team": dm.get("victimSide")})
        for fl in rd.get("flashes") or []:
            if fl.get("attackerSteamID") is None or fl.get("playerSteamID") is None:
                continue
            tick = int(fl["tick"])
            blinds.append({"tick": tick, "round_num": rnum, "t": (tick - fe) / tick_rate,
                           "attacker": fl["attackerSteamID"], "victim": fl["playerSteamID"],
                           "duration": float(fl.get("flashDuration") or 0),
                           "attacker_team": fl.get("attackerSide"), "victim_team": fl.get("playerSide")})
        plant_tick, plant_site = None, None
        for b in rd.get("bombEvents") or []:
            ev = BOMB_ACTIONS.get(b.get("bombAction"))
            if ev is None:
                continue
            tick = int(b["tick"])
            if ev == "plant" and plant_tick is None:
                plant_tick, plant_site = tick, b.get("bombSite")
            bomb.append({"tick": tick, "round_num": rnum, "t": (tick - fe) / tick_rate, "event": ev,
                         "player": b.get("playerSteamID"), "site": b.get("bombSite")})
        if reason == "bomb_exploded":
            bomb.append({"tick": end, "round_num": rnum, "t": (end - fe) / tick_rate, "event": "explode",
                         "player": None, "site": plant_site})
        t_eq = float(rd.get("tFreezeTimeEndEqVal") or 0)
        ct_eq = float(rd.get("ctFreezeTimeEndEqVal") or 0)
        rounds.append({
            "round_num": rnum, "start_tick": int(rd["startTick"]), "freeze_end_tick": fe, "end_tick": end,
            "winner": rd["winningSide"], "reason": reason,
            "plant_tick": float(plant_tick) if plant_tick is not None else np.nan, "plant_site": plant_site,
            "t_team": rd.get("tTeam") or "T", "ct_team": rd.get("ctTeam") or "CT",
            "t_equip": t_eq, "ct_equip": ct_eq, "t_buy": buy_type(t_eq), "ct_buy": buy_type(ct_eq),
            "ct_has_kit": ct_kit,
        })
    if not rounds:
        raise ValueError(f"{path}: no playable rounds")

    smp = pd.DataFrame(samples, columns=SAMPLE_COLUMNS)
    smp["place"] = places.assign(smp["x"], smp["y"], smp["z"])
    smp["steamid"] = smp["steamid"].astype("int64")

    kdf = pd.DataFrame(kills) if kills else empty_frame(KILL_COLUMNS + ["attacker_z", "victim_z"])
    if len(kdf):
        kdf["attacker_place"] = _place_or_none(places, kdf, "attacker")
        kdf["victim_place"] = _place_or_none(places, kdf, "victim")
        for c in ("attacker", "victim", "assister"):
            kdf[c] = _sid(kdf[c])
    else:
        kdf["attacker_place"] = kdf["victim_place"] = None
    ddf = pd.DataFrame(dmgs) if dmgs else empty_frame(DAMAGE_COLUMNS)
    bdf = pd.DataFrame(blinds) if blinds else empty_frame(BLIND_COLUMNS)
    bombdf = pd.DataFrame(bomb) if bomb else empty_frame(BOMB_COLUMNS)
    for df, cols in ((ddf, ("attacker", "victim")), (bdf, ("attacker", "victim")), (bombdf, ("player",))):
        if len(df):
            for c in cols:
                df[c] = _sid(df[c])

    pdf = pd.DataFrame({"steamid": list(players), "name": list(players.values())}).astype({"steamid": "int64"})
    return MatchData(
        source=str(path), map_name=map_name, tick_rate=tick_rate, players=pdf,
        rounds=pd.DataFrame(rounds)[ROUND_COLUMNS], kills=kdf[KILL_COLUMNS], damages=ddf[DAMAGE_COLUMNS],
        blinds=bdf[BLIND_COLUMNS], bomb=bombdf[BOMB_COLUMNS], samples=smp[SAMPLE_COLUMNS],
        extra={"esta": {k: d.get(k) for k in ("matchId", "demoId", "competitionName", "hltvUrl", "matchDate", "matchName")}},
    )


def _place_or_none(places: NavPlaces, df: pd.DataFrame, who: str):
    x, y, z = df[f"{who}_x"], df[f"{who}_y"], df[f"{who}_z"]
    ok = x.notna() & y.notna()
    out = np.full(len(df), None, dtype=object)
    if ok.any():
        out[ok.to_numpy()] = places.assign(x[ok], y[ok], z[ok].fillna(0.0))
    return out


def _sid(s: pd.Series) -> pd.Series:
    return pd.Series([pd.NA if v is None or (isinstance(v, float) and np.isnan(v)) else int(v) for v in s],
                     index=s.index, dtype="Int64")


def convert_dir(src: str | Path, out: str | Path, maps: list[str] | None = None, sample_hz: float = 1.0,
                workers: int = 2) -> list[Path]:
    """Convert every ESTA file under ``src`` (optionally only ``maps``) to ``.pkl`` files under ``out``."""
    from concurrent.futures import ProcessPoolExecutor

    src, out = Path(src), Path(out)
    files = sorted([*src.glob("*.json.xz"), *src.glob("*.json")])
    if maps:
        files = [f for f in files if esta_map(f) in maps]
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(f, out / (f.name.split(".")[0] + ".pkl"), sample_hz) for f in files]
    jobs = [j for j in jobs if not j[1].exists()]
    done: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_convert_one, jobs):
            if res is not None:
                done.append(res)
    return done


def _convert_one(job) -> Path | None:
    f, target, hz = job
    try:
        load_esta(f, sample_hz=hz).save(target)
        return target
    except Exception as exc:  # one broken file must not stop a batch
        print(f"skip {f.name}: {exc}")
        return None
