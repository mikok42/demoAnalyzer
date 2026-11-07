"""Game constants shared across the package."""

ROUND_TIME_S = 115.0
BOMB_TIME_S = 40.0
TRADE_WINDOW_S = 5.0
ASSIST_DAMAGE = 40  # Rating 3.0 reset assists to the CS:GO value of 40 damage

# HLTV "unstoppable bomb" rule (Oct 2026 Rating 3.0 update): kills with fewer than
# 10 s left on the bomb (5 s if the CTs have a kit) do not move win probability.
BOMB_DEAD_ZONE_S = 10.0
BOMB_DEAD_ZONE_KIT_S = 5.0

TEAM_T = "T"
TEAM_CT = "CT"
TEAM_NUM = {2: TEAM_T, 3: TEAM_CT}

# Team buy brackets used by HLTV for round economy labels (total team equipment).
BUY_TYPES = [
    (0, 5000, "eco"),
    (5000, 10000, "semi-eco"),
    (10000, 20000, "semi-buy"),
    (20000, float("inf"), "full-buy"),
]

# Rough CS2 weapon prices; used to find a player's most expensive weapon and to
# classify duels for eco-adjustment.  Names follow demoparser2's inventory strings.
WEAPON_PRICES = {
    # snipers
    "AWP": 4750, "SSG 08": 1700, "SCAR-20": 5000, "G3SG1": 5000,
    # tier-one rifles
    "AK-47": 2700, "M4A4": 2900, "M4A1-S": 2900, "SG 553": 3000, "AUG": 3300,
    # tier-two rifles
    "FAMAS": 2050, "Galil AR": 1800,
    # SMGs / shotguns / heavy
    "MP9": 1250, "MAC-10": 1050, "MP7": 1500, "MP5-SD": 1500, "UMP-45": 1200,
    "P90": 2350, "PP-Bizon": 1400, "Nova": 1050, "XM1014": 2000, "Sawed-Off": 1100,
    "MAG-7": 1300, "M249": 5200, "Negev": 1700,
    # pistols
    "Desert Eagle": 700, "R8 Revolver": 600, "Five-SeveN": 500, "Tec-9": 500,
    "CZ75-Auto": 500, "P250": 300, "Dual Berettas": 300, "USP-S": 200, "P2000": 200,
    "Glock-18": 200,
    # misc
    "Zeus x27": 200,
}

SNIPERS = {"AWP", "SSG 08", "SCAR-20", "G3SG1"}
TIER1_RIFLES = {"AK-47", "M4A4", "M4A1-S", "SG 553", "AUG"}
TIER2_RIFLES = {"FAMAS", "Galil AR"}
SMG_SHOTGUN = {
    "MP9", "MAC-10", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "Nova", "XM1014",
    "Sawed-Off", "MAG-7", "M249", "Negev",
}
UPGRADED_PISTOLS = {"Desert Eagle", "R8 Revolver", "Five-SeveN", "Tec-9", "CZ75-Auto", "P250", "Dual Berettas"}
STARTER_PISTOLS = {"USP-S", "P2000", "Glock-18"}

# Equipment categories in the order HLTV describes them (strongest first).
EQUIP_CATEGORIES = ["sniper", "rifle1", "rifle2", "smg", "pistol_up", "pistol"]

# Prior duel win-rates attacker-category vs victim-category, P(attacker wins).
# Rough figures in the spirit of the table HLTV published with Rating 3.0; the
# model shrinks observed duel outcomes towards these priors.
DUEL_PRIOR = {
    "sniper":    {"sniper": 0.50, "rifle1": 0.58, "rifle2": 0.60, "smg": 0.63, "pistol_up": 0.68, "pistol": 0.76},
    "rifle1":    {"sniper": 0.42, "rifle1": 0.50, "rifle2": 0.53, "smg": 0.57, "pistol_up": 0.64, "pistol": 0.75},
    "rifle2":    {"sniper": 0.40, "rifle1": 0.47, "rifle2": 0.50, "smg": 0.54, "pistol_up": 0.61, "pistol": 0.72},
    "smg":       {"sniper": 0.37, "rifle1": 0.43, "rifle2": 0.46, "smg": 0.50, "pistol_up": 0.58, "pistol": 0.69},
    "pistol_up": {"sniper": 0.32, "rifle1": 0.36, "rifle2": 0.39, "smg": 0.42, "pistol_up": 0.50, "pistol": 0.60},
    "pistol":    {"sniper": 0.24, "rifle1": 0.25, "rifle2": 0.28, "smg": 0.31, "pistol_up": 0.40, "pistol": 0.50},
}
