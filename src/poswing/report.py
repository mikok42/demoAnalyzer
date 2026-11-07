"""Plain-text / markdown rendering helpers (no extra dependencies)."""
from __future__ import annotations

import pandas as pd


def fmt(v) -> str:
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.0f}"
        return f"{v:.2f}"
    return str(v)


def table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    sub = df[cols] if max_rows is None else df[cols].head(max_rows)
    rows = [[fmt(v) for v in row] for row in sub.itertuples(index=False)]
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    head = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([head, sep, *body])


BOX_COLUMNS = ["name", "rounds", "kills", "deaths", "assists", "adr", "kast", "kpr_adj", "dpr_adj",
               "multikill_rating", "opening_kills", "opening_deaths", "clutches_won", "swing_pct",
               "pos_move_pct", "pos_death_pct", "rating_approx"]

SWING_COLUMNS = ["name", "rounds", "swing_pct", "added_pct", "deducted_pct", "kill_pct", "damage_pct",
                 "flash_pct", "traded_pct", "death_pct", "plant_pct", "defuse_pct", "round_end_pct",
                 "pos_move_pct", "pos_death_pct", "swing_pos_aware_pct"]

ZONE_COLUMNS = ["side", "phase", "zone", "rounds_weighted", "win_rate_when_held", "lift_pp", "model_pp_at_50"]

SETUP_COLUMNS = ["setup", "n", "win_rate", "p_model", "p_base", "edge_pp", "execution_gap"]
