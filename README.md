# poswing

Position-based round swing for Counter-Strike 2 demos, plus the Rating 3.0
ingredients HLTV uses, plus an **engine mode**: give it a demo and a moment and it
tells a team the best repositioning it could make.

```
python3 -m pip install -U pip setuptools wheel
pip install -e ".[web]"          # demoparser2 + FastAPI; omit [web] for the CLI only
poswing serve                    # open http://127.0.0.1:8765, drop a .dem on the page
poswing report demos/*.dem --cache cache/
poswing engine demos/match.dem --round 7 --clock 1:20 --team NAVI --book
```

Demos are the `.dem` files HLTV offers for download on every match page (GOTV
recordings, Source 2 format). They are parsed with
[demoparser2](https://github.com/LaihoE/demoparser). On Mirage the CLI and the web
UI use a bundled model trained on 7,393 professional rounds (no extra data needed);
other maps fall back to a fit on the demo itself.

## Quick start on a Mac

```
git clone https://github.com/mikok42/demoAnalyzer.git && cd demoAnalyzer
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel   # macOS ships an old pip; this is required
pip install -e ".[web]"
./run.sh                         # or: poswing serve
# open http://127.0.0.1:8765, drop a .dem (or pass demos on the command line)
poswing serve ~/Downloads/match.dem --port 8765
```

Parsed demos and uploads live in `~/.poswing/` (override with `--data-dir`). The UI
replays each round as CT/T dots on the radar with the live win probability, kill
feed, swing attribution and an engine panel that draws arrows to the suggested zones.

## What it does

| Command  | Output |
|----------|--------|
| `serve`  | Local web UI: upload a demo, replay rounds as dots on the radar, ask the engine. |
| `parse`  | Normalises demos into cached `.pkl` files (rounds, kills, damage, flashes, bomb, 1 Hz position samples). |
| `esta`   | Converts matches from the public [ESTA](https://github.com/pnxenopoulos/esta) dataset (1,558 pro CS:GO demos as JSON) into the same cached format, so the model can be trained on thousands of professional rounds. |
| `fit`    | Fits the win-probability model for one map and saves it as JSON. |
| `report` | Per match: Rating 3.0 ingredients + approximate rating, round-swing breakdown (kill / damage / flash / traded / death / plant / defuse / round-end) and the new positional swing. Per map: zone values and each team's recurring setups with expected vs actual win rate. |
| `zones`  | Which map sections are worth holding, per side and phase (pre/post-plant). |
| `setups` | A team's setups at a reference moment (seconds into the round or first contact), worst edge first: structural flaws and execution gaps. |
| `react`  | Given what the enemy is doing, which own setups did best (empirically and by the model). |
| `engine` | Demo + round + moment -> top lines (who should move where within the horizon) with the win probability of each, optionally against a predicted enemy and with a "book" of similar historical situations. |

## How it works

1. **Zones.** By default the game's own nav-mesh place names (`BombsiteA`, `Connector`,
   `Apartments`...) read from each player, so no map drawing is needed. A regular grid
   or custom polygons (JSON) are also supported.
2. **Game state** every second and at every kill/plant: players alive per side, bomb,
   clock, team buys (HLTV brackets), defuse kit, and the multiset of zones each side occupies.
3. **Win probability** `P(T wins | state)` = baseline + positional layer.
   The baseline is what HLTV describes for Round Swing (map, side, alive, bomb, economy)
   as a prior corrected by an empirical-Bayes table. The positional layer is an
   L2-regularised logistic regression on the baseline logit with features: side present
   in zone, extra players stacked in zone, how spread the side is, separately pre/post-plant.
   `P_pos - P_base` is the value of the positions themselves.
4. **Round swing** follows the Rating 3.0 write-ups: probability before/after each kill
   with credit split finisher / damage share / flash assist / traded teammate, deaths
   deducted, plants and round-end residual shared by the team, no negative swing for a
   kill, kills in the unstoppable-bomb window ignored (<10 s, <5 s with kit, unless
   defusing). Every round is exactly zero-sum and sums to `outcome - P0`.
5. **Positional swing** (new):
   `pos_move` credits changes of the team's positional edge to the players who moved;
   `pos_death = -|contribution of the victim's zone|` so dying in a good spot (wasted) and
   dying in a bad spot (you chose it) both cost.
6. **Engine.** Legal moves = zones reachable within the horizon according to how players
   actually travel between zones in the data (Dijkstra on median transition times).
   Beam search over the team's assignment, scored by the model against the enemy's
   current zones or a Monte-Carlo prediction of where the enemy will be. A small cost per
   mover prefers simple lines.

## Training on the ESTA dataset

HLTV demos cannot be downloaded by scripts, but the ESTA dataset ships 1,558 parsed
professional demos (2 Hz player snapshots, kills, damage, flashes, bomb events).
ESTA frames have coordinates but no place names, so `poswing esta` looks the place up
in the CS:GO nav mesh (bundled from awpy 1.3.1) and renames the few callouts that
changed in CS2 (`Tunnel` -> `Underpass`, `TicketBooth` -> `CTSpawn`, ...), so a model
trained on ESTA evaluates a CS2 demo directly.

```
poswing esta esta/data/lan -o cache/esta --map de_mirage --workers 3
poswing fit cache/esta -o models/mirage.json --map de_mirage          # 278 matches, 7,393 rounds
poswing engine match.dem --round 7 --clock 1:25 --team NAVI --model models/mirage.json --data cache/esta --book
```

A model trained this way on all 278 ESTA Mirage matches ships in
`models/de_mirage_esta.json` (`--model models/de_mirage_esta.json`).
poswing engine match.dem --round 7 --clock 1:25 --team NAVI --model models/mirage.json --data cache/esta --book
```

A model trained this way on all 278 ESTA Mirage matches ships inside the package at
`poswing/assets/models/de_mirage.json` (plus a transition graph for the engine and
awpy radar images). Pass `--model fit` to force a fit on the inputs instead; pass
`--model PATH` to use your own.

On Mirage this gives 7,393 rounds / 655k states and 287 positional features instead
of the ~90 noisy ones a single demo yields; the baseline table (alive counts, bomb,
* HLTV has not published the Rating 3.0 formula; `rating_approx` combines the eco-adjusted
  ingredients using the public Rating 2.0 regression and a swing term. Treat it as an
  ordering, not as HLTV's number. Eco categories and duel win-rates follow the published
  description with priors that the data refines.
* Positional estimates need data. One map with 23 rounds gives ~90 noisy features; the
  tool warns below 60 rounds. Feed a season of demos per map before trusting the engine.
* Zone-based reasoning ignores utility, angles and timing inside a zone; the model is a
  positional prior, not a tactics oracle.

## Development

```
pip install -e .[dev]
pytest
python docs/build_pdf.py   # regenerates docs/poswing.pdf
```

Without demos, `poswing.synth.synth_match()` builds a synthetic match with known strong zones.
