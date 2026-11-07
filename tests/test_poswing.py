import json

import numpy as np
import pandas as pd
import pytest

from poswing.analysis import Dataset, parse_zone_spec, reactions, team_setups, zone_values
from poswing.engine import Engine, TransitionGraph, book, situation_at
from poswing.parse import _sid, _winner, assign_round
from poswing.rating import box_score
from poswing.state import GameState, build_states
from poswing.swing import compute_swing
from poswing.synth import synth_match
from poswing.winprob import WinProbModel
from poswing.zones import Zoner, humanize


@pytest.fixture(scope="module")
def match():
    return synth_match(n_rounds=80, seed=1)


@pytest.fixture(scope="module")
def fitted(match):
    zoner = Zoner()
    ds = Dataset.build([match], zoner)
    model = ds.fit_model("de_synth")
    return zoner, ds, model


# ----------------------------------------------------------------------- zones
def test_humanize():
    assert humanize("BombsiteA") == "A Site"
    assert humanize("TopofMid") == "Top of Mid"
    assert humanize("") == "Unknown"


def test_grid_and_custom_zoner(tmp_path):
    df = pd.DataFrame({"x": [10.0, 900.0], "y": [10.0, 900.0], "place": ["Mid", "Mid"]})
    assert Zoner(strategy="grid", cell=512).assign(df).tolist() == ["cell_0_0", "cell_1_1"]
    spec = {"zones": [{"name": "Box", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}]}
    f = tmp_path / "z.json"
    f.write_text(json.dumps(spec))
    z = Zoner.from_args(custom=f)
    assert z.assign(df).tolist() == ["Box", "Mid"]


def test_parse_zone_spec():
    assert parse_zone_spec("A Site x2, Connector, Mid=2") == ("A Site", "A Site", "Connector", "Mid", "Mid")


# ----------------------------------------------------------------------- parse
def test_parse_helpers():
    rounds = pd.DataFrame({"round_num": [1, 2], "freeze_end_tick": [100, 500], "end_tick": [300, 700]})
    out = assign_round(pd.Series([50, 100, 250, 400, 600, 800]), rounds)
    assert out.tolist()[1:3] == [1, 1] and out.tolist()[4] == 2
    assert np.isnan(out.tolist()[0]) and np.isnan(out.tolist()[3])
    assert _winner("CT") == "CT" and _winner(2) == "T" and _winner("TERRORIST") == "T" and _winner(None) is None
    s = _sid(pd.Series(["76561198014914403", None, "x"]))
    assert s.iloc[0] == 76561198014914403 and pd.isna(s.iloc[1]) and pd.isna(s.iloc[2])


# ---------------------------------------------------------------------- states
def test_terminal_rules():
    assert GameState("m", 0, 3, False, 50).terminal_t_win() == 0.0
    assert GameState("m", 2, 0, False, 50).terminal_t_win() == 1.0
    assert GameState("m", 1, 2, True, 7.0, ct_has_kit=False).terminal_t_win() == 1.0
    assert GameState("m", 1, 2, True, 7.0, ct_has_kit=True).terminal_t_win() is None
    assert GameState("m", 1, 2, True, 4.0, ct_has_kit=True).terminal_t_win() == 1.0
    assert GameState("m", 1, 2, True, 4.0, ct_has_kit=False, defusing=True).terminal_t_win() is None


def test_build_states(match):
    meta, states = build_states(match, Zoner())
    assert len(meta) == len(states) > 0
    assert set(meta["round_num"]) == set(match.rounds["round_num"])
    assert abs(meta.groupby("round_num")["weight"].sum() - 1.0).max() < 1e-9


# --------------------------------------------------------------------- winprob
def test_model_recovers_strong_zones(fitted):
    _, _, model = fitted
    coefs = {(r["side"], r["phase"], r["zone"]): r["logit"] for r in model.zone_coefficients()}
    assert coefs[("CT", "pre", "Window")] > coefs[("CT", "pre", "TSpawn")] if ("CT", "pre", "TSpawn") in coefs else True
    assert coefs[("CT", "pre", "Window")] > 0
    assert coefs[("T", "pre", "TRamp")] > coefs.get(("T", "pre", "CTSpawn"), -1)


def test_model_monotone_in_alive(fitted):
    _, _, model = fitted
    p_even = model.p_t(GameState("de_synth", 3, 3, False, 60.0))
    p_up = model.p_t(GameState("de_synth", 4, 3, False, 60.0))
    p_down = model.p_t(GameState("de_synth", 3, 4, False, 60.0))
    assert p_down < p_even < p_up
    assert model.p_t(GameState("de_synth", 3, 3, True, 30.0)) > p_even


def test_model_roundtrip(fitted, tmp_path):
    _, ds, model = fitted
    f = tmp_path / "model.json"
    model.save(f)
    m2 = WinProbModel.load(f)
    sts = ds.states[:50]
    assert np.allclose(model.predict(sts), m2.predict(sts))


# ----------------------------------------------------------------------- swing
def test_swing_is_zero_sum(match, fitted):
    zoner, _, model = fitted
    res = compute_swing(match, model, zoner)
    ev = res.events[res.events["component"] == "swing"]
    per_round = ev.groupby(["round_num", "team"])["delta"].sum().unstack()
    assert np.allclose(per_round["T"] + per_round["CT"], 0.0, atol=1e-9)
    expect = (res.rounds.set_index("round_num")["winner"] == "T").astype(float) - res.rounds.set_index("round_num")["p0_t"]
    assert np.allclose(per_round["T"], expect.reindex(per_round.index), atol=1e-9)
    assert (res.players["pos_death_pct"] <= 1e-9).all()
    assert "pos_move_pct" in res.players


def test_kills_never_negative_swing(match, fitted):
    zoner, _, model = fitted
    res = compute_swing(match, model, zoner)
    ev = res.events
    assert (ev[ev["kind"].isin(["kill", "damage", "flash", "traded"])]["delta"] >= -1e-12).all()
    assert (ev[ev["kind"] == "death"]["delta"] <= 1e-12).all()


# ---------------------------------------------------------------------- rating
def test_box_score(match, fitted):
    zoner, _, model = fitted
    res = compute_swing(match, model, zoner)
    box = box_score(match, res)
    assert box["kills"].sum() == len(match.kills[match.kills["attacker_team"] != match.kills["victim_team"]])
    assert box["deaths"].sum() == len(match.kills)
    assert ((box["kast"] >= 0) & (box["kast"] <= 100)).all()
    assert (box["rounds"] == match.n_rounds).all()
    assert box["rating_approx"].between(0.0, 3.0).all()


# -------------------------------------------------------------------- analysis
def test_zone_values_and_setups(fitted):
    _, ds, model = fitted
    zv = zone_values(ds, model, "de_synth")
    assert {"side", "zone", "model_pp_at_50", "win_rate_when_held"} <= set(zv.columns)
    st = team_setups(ds, model, "Alpha", "T", "de_synth", at=15.0, min_n=1)
    assert not st.empty and {"setup", "n", "win_rate", "edge_pp"} <= set(st.columns)
    emp, ranked = reactions(ds, model, "de_synth", "CT", ("Mid", "Mid", "TRamp", "TRamp", "TSpawn"), at=15.0)
    assert not ranked.empty


# ---------------------------------------------------------------------- engine
def test_engine_prefers_strong_zones(match, fitted):
    zoner, ds, model = fitted
    sit = situation_at(match, zoner, round_num=3, t=12.0)
    graph = TransitionGraph.from_matches([match], zoner, "de_synth")
    assert graph.reachable("Mid", 10.0).get("Connector") is not None
    eng = Engine(model, graph)
    lines, p_now = eng.suggest(sit, "CT", horizon=10.0, top=3)
    assert lines and lines[0].p_win >= p_now - 1e-9
    assert all(0.0 <= ln.p_win <= 1.0 for ln in lines)
    lines_pred, _ = eng.suggest(sit, "CT", horizon=10.0, top=3, enemy_mode="predict")
    assert lines_pred
    bk = book(ds, sit, "CT", horizon=10.0)
    assert isinstance(bk, pd.DataFrame)


def test_situation_describe(match, fitted):
    zoner, _, _ = fitted
    sit = situation_at(match, zoner, round_num=1, t=5.0)
    text = sit.describe()
    assert "round 1" in text and "T  (" in text and "CT (" in text


# ------------------------------------------------------------------------- cli
def test_cli_report_and_engine(match, tmp_path, capsys):
    from poswing.cli import main

    pkl = tmp_path / "synth.pkl"
    match.save(pkl)
    out = tmp_path / "report.md"
    main(["report", str(pkl), "-o", str(out), "--at", "15"])
    text = out.read_text()
    assert "Box score" in text and "Zone values" in text
    main(["engine", str(pkl), "--round", "2", "--time", "10", "--side", "T", "--top", "2"])
    captured = capsys.readouterr().out
    assert "P(win)" in captured and "Engine for T" in captured
    main(["fit", str(pkl), "-o", str(tmp_path / "m.json")])
    assert (tmp_path / "m.json").exists()
                "inventory": [{"weaponName": w, "weaponClass": "Rifle"} for w in inv]}
    ts = [(1, "t1"), (2, "t2")]
    cts = [(11, "c1"), (12, "c2")]
    a_site, ct_spawn, mid, tunnel = (-337.5, -2012.5, -171.0), (-1387.5, -2268.0, -222.0), (-837.5, -662.5, -264.0), (-1012.5, -387.5, -340.0)

    def frame(tick, t_alive=(True, True)):
        return {"tick": tick, "isKillFrame": False, "bombPlanted": False,
    # 2 Hz thinned to 1 Hz, kill frames kept and flagged
    per_round = m.samples.groupby("round_num")["tick"].nunique().iloc[0]
    assert 60 <= per_round <= 64 and m.samples["is_event"].any()
    assert nav_places("de_mirage").assign([-1387.5], [-2268.0], [-222.0])[0] == "CTSpawn"
    # downstream pipeline accepts it
    ds = Dataset.build([m], Zoner())
    assert ds.meta["map_name"].eq("de_mirage").all() and len(ds.states) > 100
