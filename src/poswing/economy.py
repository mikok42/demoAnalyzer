"""Equipment classification and eco-adjustment (Rating 3.0 style)."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd

from .constants import (
    BUY_TYPES, DUEL_PRIOR, EQUIP_CATEGORIES, SMG_SHOTGUN, SNIPERS, STARTER_PISTOLS,
    TIER1_RIFLES, TIER2_RIFLES, UPGRADED_PISTOLS, WEAPON_PRICES,
)


def best_weapon(inventory: Iterable[str] | None) -> str | None:
    if inventory is None:
        return None
    if isinstance(inventory, str):
        inventory = [inventory]
    best, best_price = None, -1
    for item in inventory:
        price = WEAPON_PRICES.get(item)
        if price is not None and price > best_price:
            best, best_price = item, price
    return best


def equip_category(inventory: Iterable[str] | None) -> str:
    """Map a player's inventory to one of HLTV's six economy categories."""
    w = best_weapon(inventory)
    if w in SNIPERS:
        return "sniper"
    if w in TIER1_RIFLES:
        return "rifle1"
    if w in TIER2_RIFLES:
        return "rifle2"
    if w in SMG_SHOTGUN:
        return "smg"
    if w in UPGRADED_PISTOLS:
        return "pistol_up"
    if w in STARTER_PISTOLS or w is None:
        return "pistol"
    return "pistol"


def buy_type(team_equip_value: float) -> str:
    for lo, hi, label in BUY_TYPES:
        if lo <= team_equip_value < hi:
            return label
    return "full-buy"


class DuelTable:
    """Attacker-vs-victim equipment category duel win-rates with prior shrinkage.

    Kill value follows the published anchor points (rifle v rifle 48% -> 1.10 kill
    points, rifle v starter pistol 75% -> ~0.54): ``value = 2.1 - 2.07 * p_win``.
    """

    def __init__(self, prior_weight: float = 30.0):
        self.prior_weight = prior_weight
        self.wins: dict[tuple[str, str, str], float] = defaultdict(float)
        self.n: dict[tuple[str, str, str], float] = defaultdict(float)

    def fit(self, kills: pd.DataFrame) -> "DuelTable":
        if kills.empty:
            return self
        for row in kills.itertuples(index=False):
            if row.attacker_team == row.victim_team:
                continue
            a, v = row.attacker_cat, row.victim_cat
            if a not in EQUIP_CATEGORIES or v not in EQUIP_CATEGORIES:
                continue
            side = row.attacker_team
            self.wins[(side, a, v)] += 1
            self.n[(side, a, v)] += 1
            other = "CT" if side == "T" else "T"
            # the same duel seen from the victim's perspective is a loss
            self.n[(other, v, a)] += 1
        return self

    def p_win(self, side: str, attacker_cat: str, victim_cat: str) -> float:
        prior = DUEL_PRIOR.get(attacker_cat, {}).get(victim_cat, 0.5)
        key = (side, attacker_cat, victim_cat)
        n = self.n.get(key, 0.0)
        w = self.wins.get(key, 0.0)
        return (w + self.prior_weight * prior) / (n + self.prior_weight)

    def kill_value(self, side: str, attacker_cat: str, victim_cat: str) -> float:
        p = self.p_win(side, attacker_cat, victim_cat)
        return max(0.2, 2.1 - 2.07 * p)

    def death_value(self, side: str, victim_cat: str, attacker_cat: str) -> float:
        """Deaths are weighted by how expected they were: dying to a weaker loadout costs more."""
        p_survive = 1.0 - self.p_win("CT" if side == "T" else "T", attacker_cat, victim_cat)
        return max(0.2, 2.1 - 2.07 * (1.0 - p_survive))
