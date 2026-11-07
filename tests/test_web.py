"""HTTP API smoke tests against a synthetic match (no demoparser2 needed)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from poswing.session import MatchSession
from poswing.synth import synth_match
from poswing.web.app import create_app
from poswing.web.store import Store


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "data", sample_hz=1.0)
    m = synth_match(n_rounds=4, seed=3)
    pkl = s.cache_dir / "synth.pkl"
    m.save(pkl)
    s._register(pkl)
    return s


@pytest.fixture()
def client(store: Store) -> TestClient:
    return TestClient(create_app(store))


def test_list_and_summary(client: TestClient):
    r = client.get("/api/matches")
    assert r.status_code == 200
    body = r.json()
    assert body["matches"] and body["matches"][0]["id"] == "synth"
    s = client.get("/api/matches/synth").json()
    assert s["n_rounds"] == 4
    assert "box_score" in s and "rounds" in s
    assert all(isinstance(p["steamid"], str) for p in s["players"])


def test_round_frames_and_engine(client: TestClient):
    rd = client.get("/api/matches/synth/rounds/1").json()
    assert rd["frames"] and isinstance(rd["frames"][0]["players"][0][0], str)
    assert "roster" in rd and all(isinstance(k, str) for k in rd["roster"])
    eng = client.post("/api/matches/synth/engine",
                      json={"round_num": 1, "t": 5.0, "side": "CT", "horizon": 8}).json()
    assert "lines" in eng and eng["side"] == "CT"
    assert isinstance(eng["own"][0]["steamid"], str)


def test_zones_and_radar(client: TestClient):
    z = client.get("/api/matches/synth/zones").json()
    assert "zones" in z and "centroids" in z
    # synth map has no radar; Mirage does
    assert client.get("/api/radar/de_mirage.png").status_code == 200
    assert client.get("/").status_code == 200


def test_session_steamid_strings():
    s = MatchSession(synth_match(n_rounds=2, seed=1))
    sum_ = s.summary()
    assert isinstance(sum_["players"][0]["steamid"], str)
    assert isinstance(sum_["box_score"][0]["steamid"], str)
    rd = s.round_detail(1)
    assert isinstance(rd["frames"][0]["players"][0][0], str)
