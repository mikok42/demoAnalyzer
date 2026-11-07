"""Command line interface."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import __version__, resources
from .analysis import Dataset, parse_zone_spec, reactions, team_setups, zone_values
from .data import MatchData
from .economy import DuelTable
from .report import BOX_COLUMNS, SETUP_COLUMNS, SWING_COLUMNS, ZONE_COLUMNS, table
from .winprob import PositionalLayer, WinProbModel
from .zones import Zoner

log = logging.getLogger("poswing")


# ------------------------------------------------------------------ utilities
def _zoner(args) -> Zoner:
    return Zoner.from_args(strategy=args.zoning, cell=args.cell, custom=args.zones_file)


def _load_inputs(paths: list[str], cache: str | None, hz: float) -> list[MatchData]:
    from .parse import parse_demo

    out: list[MatchData] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files = sorted([*path.glob("*.dem"), *path.glob("*.pkl")])
        else:
            files = [path]
        for f in files:
            if f.suffix == ".pkl":
                out.append(MatchData.load(f))
                continue
            cached = Path(cache) / (f.stem + ".pkl") if cache else None
            if cached and cached.exists():
                out.append(MatchData.load(cached))
                continue
            log.info("parsing %s", f)
            m = parse_demo(f, sample_hz=hz)
            if cached:
                m.save(cached)
            out.append(m)
    if not out:
        sys.exit("no inputs found")
    return out


def _model_for(ds: Dataset, map_name: str, model_path: str | None) -> WinProbModel:
    """Model priority: --model FILE, then the bundled pro-data model for the map, then a fit on the inputs.

    ``--model fit`` forces the fit on the inputs (useful to compare against the bundled model).
    """
    if model_path and model_path != "fit":
        if not Path(model_path).exists():
            sys.exit(f"model file {model_path} not found")
        model = WinProbModel.load(model_path)
        if map_name not in model.maps:
            log.warning("model %s was not trained on %s; fitting from inputs instead", model_path, map_name)
        else:
            return model
    elif model_path != "fit":
        bundled = resources.bundled_model(map_name)
        if bundled is not None:
            log.info("using the bundled %s model (%d pro rounds); pass --model fit to fit on the inputs instead",
                     map_name, bundled.n_rounds)
            return bundled
    model = ds.fit_model(map_name)
    if model.n_rounds < 60:
        log.warning("only %d rounds of %s in the data: positional estimates will be noisy, feed more demos",
                    model.n_rounds, map_name)
    return model


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--zoning", choices=["place", "grid", "custom"], default="place", help="zone strategy (default: nav place names)")
    p.add_argument("--cell", type=float, default=512.0, help="grid cell size in units for --zoning grid")
    p.add_argument("--zones-file", help="JSON with custom polygons for --zoning custom")
    p.add_argument("--cache", help="directory to cache parsed demos as .pkl")
    p.add_argument("--hz", type=float, default=1.0, help="position sampling frequency (samples per second)")
    p.add_argument("-v", "--verbose", action="store_true")


# ------------------------------------------------------------------- commands
def cmd_parse(args) -> None:
    from .parse import parse_demo

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in args.demos:
        m = parse_demo(f, sample_hz=args.hz)
        target = out / (Path(f).stem + ".pkl")
        m.save(target)
        print(f"{f}: {m.map_name}, {m.n_rounds} rounds, {len(m.kills)} kills, teams {m.team_names()} -> {target}")


def cmd_esta(args) -> None:
    from .esta import convert_dir

    done = convert_dir(args.src, args.out, maps=args.map, sample_hz=args.hz, workers=args.workers)
    print(f"converted {len(done)} ESTA matches -> {args.out}")


def cmd_fit(args) -> None:
    matches = _load_inputs(args.inputs, args.cache, args.hz)
    ds = Dataset.build(matches, _zoner(args))
    maps = [args.map] if args.map else ds.maps()
    if len(maps) != 1:
        sys.exit(f"inputs contain several maps {maps}; choose one with --map")
    model = ds.fit_model(maps[0], positional=PositionalLayer(l2=args.l2, target=args.target, horizon=args.horizon,
                                                             shrink_rounds=args.shrink))
    model.save(args.out)
    print(f"fitted {maps[0]} on {model.n_rounds} rounds / {model.n_states} states, {len(model.positional.features)} positional features -> {args.out}")


def cmd_report(args) -> None:
    from .rating import box_score
    from .swing import compute_swing

    matches = _load_inputs(args.inputs, args.cache, args.hz)
    zoner = _zoner(args)
    ds = Dataset.build(matches, zoner)
    chunks: list[str] = []
    payload: dict = {"matches": []}
    duels = DuelTable().fit(pd.concat([m.kills for m in matches], ignore_index=True))
    for map_name in ds.maps():
        model = _model_for(ds, map_name, args.model)
        chunks.append(f"# {map_name}\n\nModel: {model.n_rounds} rounds, {model.n_states} states, "
                      f"{len(model.positional.features)} positional features.\n")
        for mi, m in enumerate(matches):
            if m.map_name != map_name:
                continue
            idx = ds.meta.index[ds.meta["match"] == mi]
            meta = ds.meta.loc[idx].reset_index(drop=True)
            states = [ds.states[i] for i in idx]
            swing = compute_swing(m, model, zoner, meta_states=(meta, states))
            box = box_score(m, swing, duels)
            score = m.rounds["winner"].value_counts()
            chunks.append(f"## {Path(m.source).name}\n\n{' vs '.join(m.team_names())}, {m.n_rounds} rounds "
                          f"(T {int(score.get('T', 0))} - CT {int(score.get('CT', 0))} by side).\n")
            chunks.append("### Box score (Rating 3.0 ingredients, rating is an approximation)\n\n" + table(box, BOX_COLUMNS))
            chunks.append("\n### Round swing breakdown (percentage points per round)\n\n" + table(swing.players, SWING_COLUMNS))
            payload["matches"].append({"source": m.source, "map": m.map_name, "teams": m.team_names(),
                                       "box_score": json.loads(box.to_json(orient="records")),
                                       "swing": json.loads(swing.players.to_json(orient="records"))})
        zv = zone_values(ds, model, map_name)
        chunks.append(f"\n## Zone values on {map_name}\n\nPositive `model_pp_at_50` = holding that zone helps the side "
                      f"(percentage points at a 50/50 state, one player).\n\n" + table(zv, ZONE_COLUMNS, max_rows=args.max_rows))
        teams = sorted({t for m in matches if m.map_name == map_name for t in m.team_names()})
        for team in teams:
            for side in ("T", "CT"):
                st = team_setups(ds, model, team, side, map_name, at=args.at, min_n=args.min_n)
                if st.empty:
                    continue
                chunks.append(f"\n## {team} on {side} - setups at {args.at}s\n\nSorted worst edge first: common setups with a "
                              f"negative edge are structural flaws, a negative `execution_gap` means the setup loses more "
                              f"than it should.\n\n" + table(st, SETUP_COLUMNS, max_rows=args.max_rows))
        payload.setdefault("zones", {})[map_name] = json.loads(zv.to_json(orient="records"))
    text = "\n".join(chunks)
    if args.out:
        Path(args.out).write_text(text)
        print(f"report written to {args.out}")
    else:
        print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1, default=str))


def cmd_zones(args) -> None:
    matches = _load_inputs(args.inputs, args.cache, args.hz)
    ds = Dataset.build(matches, _zoner(args))
    for map_name in ([args.map] if args.map else ds.maps()):
        model = _model_for(ds, map_name, args.model)
        zv = zone_values(ds, model, map_name, min_rounds=args.min_n)
        if args.side:
            zv = zv[zv["side"] == args.side]
        print(f"# {map_name}\n" + table(zv, ZONE_COLUMNS))


def cmd_setups(args) -> None:
    matches = _load_inputs(args.inputs, args.cache, args.hz)
    ds = Dataset.build(matches, _zoner(args))
    map_name = args.map or (ds.maps()[0] if len(ds.maps()) == 1 else sys.exit("choose --map"))
    model = _model_for(ds, map_name, args.model)
    team = _resolve_team(args.team, matches)
    at = "contact" if args.at == "contact" else float(args.at)
    for side in ([args.side] if args.side else ["T", "CT"]):
        st = team_setups(ds, model, team, side, map_name, at=at, min_n=args.min_n)
        print(f"# {team} on {side} ({map_name}) at {args.at}\n" + table(st, SETUP_COLUMNS) + "\n")


def cmd_react(args) -> None:
    matches = _load_inputs(args.inputs, args.cache, args.hz)
    ds = Dataset.build(matches, _zoner(args))
    map_name = args.map or (ds.maps()[0] if len(ds.maps()) == 1 else sys.exit("choose --map"))
    model = _model_for(ds, map_name, args.model)
    enemy = parse_zone_spec(args.enemy)
    at = "contact" if args.at == "contact" else float(args.at)
    emp, ranked = reactions(ds, model, map_name, args.side, enemy, at=at, top=args.top)
    print(f"# {args.side} reactions on {map_name} when enemy is: {', '.join(enemy)} (at {args.at})\n")
    print("## Played against similar enemy setups\n" + table(emp) + "\n")
    print("## Model ranking of observed own setups against exactly this enemy setup\n" + table(ranked))


def cmd_engine(args) -> None:
    from .engine import Engine, TransitionGraph, book, situation_at

    zoner = _zoner(args)
    demo = _load_inputs([args.demo], args.cache, args.hz)[0]
    data = _load_inputs(args.data, args.cache, args.hz) if args.data else []
    matches = [demo] + [m for m in data if m.source != demo.source and m.map_name == demo.map_name]
    ds = Dataset.build(matches, zoner)
    model = _model_for(ds, demo.map_name, args.model)
    t = args.time
    if args.clock:
        mm, ss = args.clock.split(":")
        t = 115.0 - (int(mm) * 60 + int(ss))
    sit = situation_at(demo, zoner, args.round, t=t, tick=args.tick)
    side = args.side or _side_for_team(sit, args.team)
    graph = TransitionGraph.from_matches(matches, zoner, demo.map_name, horizon=args.horizon)
    bundled_graph = resources.bundled_graph(demo.map_name)
    if bundled_graph is not None and args.zoning == "place":
        graph = bundled_graph.merge(graph)
    eng = Engine(model, graph, move_cost=args.move_cost)
    lines, p_now = eng.suggest(sit, side, horizon=args.horizon, top=args.top, beam=args.beam,
                               enemy_mode=args.enemy, max_movers=args.max_movers)
    print(sit.describe())
    print(f"\nEngine for {side} ({sit.teams.get(side)}), horizon {args.horizon:.0f}s, enemy {args.enemy}, "
          f"model {model.n_rounds} rounds of {demo.map_name}")
    print(f"Current P(win) for {side}: {100 * p_now:.1f}%  (baseline without positions: "
          f"{100 * model.p_side(sit.state, side, False):.1f}%)\n")
    for i, ln in enumerate(lines, 1):
        print(f"{i}. {ln.describe()}")
    if args.book:
        bk = book(ds, sit, side, horizon=args.horizon)
        print(f"\nBook - similar situations in the data, what {side} did {args.horizon:.0f}s later:\n" + table(bk))
    if args.json:
        Path(args.json).write_text(json.dumps({
            "situation": sit.describe(), "side": side, "p_now": p_now,
            "lines": [{"p_win": l.p_win, "delta": l.delta, "zones": l.zones, "moves": l.moves} for l in lines],
        }, indent=1))


def cmd_serve(args) -> None:
    try:
        from .web.app import serve
    except ImportError:
        sys.exit("the web UI needs extra packages: pip install 'poswing[web]'  (fastapi, uvicorn, python-multipart)")
    serve(args.data_dir, host=args.host, port=args.port, sample_hz=args.hz, model_path=args.model,
          demos=args.demos, open_browser=not args.no_browser)


def _resolve_team(name: str, matches: list[MatchData]) -> str:
    teams = sorted({t for m in matches for t in m.team_names()})
    for t in teams:
        if t.lower() == name.lower():
            return t
    hits = [t for t in teams if name.lower() in t.lower()]
    if len(hits) == 1:
        return hits[0]
    sys.exit(f"team '{name}' not found; available: {teams}")


def _side_for_team(sit, team: str | None) -> str:
    if not team:
        sys.exit("give --side T/CT or --team NAME")
    for side, name in sit.teams.items():
        if team.lower() in str(name).lower():
            return side
    sys.exit(f"team '{team}' not in this round; teams: {sit.teams}")


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="poswing", description="Position-based round swing and Rating 3.0-style analytics for CS2 demos")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("parse", help="parse demos into cached .pkl files")
    s.add_argument("demos", nargs="+")
    s.add_argument("-o", "--out", required=True)
    _add_common(s)
    s.set_defaults(func=cmd_parse)

    s = sub.add_parser("esta", help="convert ESTA dataset files (.json.xz) into cached .pkl files")
    s.add_argument("src", help="directory with ESTA .json.xz files")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--map", nargs="*", help="only these maps, e.g. de_mirage")
    s.add_argument("--hz", type=float, default=1.0, help="thin the 2 Hz ESTA frames to this rate")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_esta)

    s = sub.add_parser("fit", help="fit the win-probability model on demos/.pkl files")
    s.add_argument("inputs", nargs="+")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--map")
    s.add_argument("--l2", type=float, default=2.0, help="L2 strength of the positional layer")
    s.add_argument("--target", choices=["horizon", "outcome"], default="horizon",
                   help="horizon: zone coefficients explain the change of the baseline over the next --horizon seconds "
                        "(causal-ish); outcome: classic round-outcome regression (correlational)")
    s.add_argument("--horizon", type=float, default=15.0, help="seconds ahead for --target horizon")
    s.add_argument("--shrink", type=float, default=5.0, help="extra shrinkage for rare zones (weighted rounds)")
    _add_common(s)
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("report", help="box score, swing breakdown, zone values and team setups")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--model")
    s.add_argument("-o", "--out")
    s.add_argument("--json")
    s.add_argument("--at", default="20", help="reference moment for setups: seconds or 'contact'")
    s.add_argument("--min-n", type=int, default=2)
    s.add_argument("--max-rows", type=int, default=40)
    _add_common(s)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("zones", help="zone value table")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--model")
    s.add_argument("--map")
    s.add_argument("--side", choices=["T", "CT"])
    s.add_argument("--min-n", type=float, default=3.0)
    _add_common(s)
    s.set_defaults(func=cmd_zones)

    s = sub.add_parser("setups", help="a team's setups at a moment with expected vs actual win rate")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--team", required=True)
    s.add_argument("--side", choices=["T", "CT"])
    s.add_argument("--map")
    s.add_argument("--model")
    s.add_argument("--at", default="20")
    s.add_argument("--min-n", type=int, default=2)
    _add_common(s)
    s.set_defaults(func=cmd_setups)

    s = sub.add_parser("react", help="best own setups against a given enemy setup")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--side", required=True, choices=["T", "CT"], help="side you play")
    s.add_argument("--enemy", required=True, help="enemy zones, e.g. 'Apartments x3, Underpass x2'")
    s.add_argument("--map")
    s.add_argument("--model")
    s.add_argument("--at", default="20")
    s.add_argument("--top", type=int, default=10)
    _add_common(s)
    s.set_defaults(func=cmd_react)

    s = sub.add_parser("engine", help="demo + moment -> best plays for a team")
    s.add_argument("demo")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--time", type=float, help="seconds since freeze end")
    s.add_argument("--clock", help="in-game round clock instead of --time, e.g. 1:20 (pre-plant only)")
    s.add_argument("--tick", type=int)
    s.add_argument("--team")
    s.add_argument("--side", choices=["T", "CT"])
    s.add_argument("--data", nargs="*", help="extra demos/.pkl/dirs of the same map to learn from")
    s.add_argument("--model")
    s.add_argument("--horizon", type=float, default=10.0, help="seconds the team has to reposition")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--beam", type=int, default=40)
    s.add_argument("--enemy", choices=["static", "predict"], default="static")
    s.add_argument("--max-movers", type=int)
    s.add_argument("--move-cost", type=float, default=0.003, help="probability cost per moving player (prefers simple lines)")
    s.add_argument("--book", action="store_true", help="also show similar historical situations")
    s.add_argument("--json")
    _add_common(s)
    s.set_defaults(func=cmd_engine)

    s = sub.add_parser("serve", help="local web UI: upload demos, replay rounds on the radar, ask the engine")
    s.add_argument("demos", nargs="*", help="demos / .pkl files / folders to import on start")
    s.add_argument("--data-dir", default="~/.poswing", help="where uploads and parsed demos are kept")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--hz", type=float, default=2.0, help="position samples per second for newly parsed demos")
    s.add_argument("--model", help="win-probability model file (default: bundled model for the map)")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200)
    args.func(args)


if __name__ == "__main__":
    main()
