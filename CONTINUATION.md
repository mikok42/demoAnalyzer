# Continuation checkpoint (2026-11-03, Europe/Warsaw)

Resume from this file. The user asked for an easy Mac app with demo upload and
radar dots, then to keep iterating (code review → model → multi-map) until
evening. **There is no git remote** — work lives only in `/agent/poswing` on this
VM. Do not try to push; offer a `git bundle` or ask for a remote when handing off.

## Done this session (web track)

- Bundled assets under `src/poswing/assets/`:
  - `models/de_mirage.json` (horizon target, bomb site, 7,393 rounds)
  - `graphs/de_mirage.json` (persisted `TransitionGraph`)
  - `radar/*.png` + `map_data.json` (awpy MIT; 9 maps)
- `resources.py`, `TransitionGraph.save/load/merge` (non-mutating merge; per-edge
  median union with `_loaded_medians`)
- CLI defaults to bundled model; merges bundled graph into engine; `--model fit`
  forces a local fit
- `session.py` (`MatchSession`): summary / round frames / engine / zones; steamids
  as **strings** (JS safe-integer)
- Web UI: `poswing serve` → FastAPI + vanilla JS
  - upload `.dem`, background parse, `~/.poswing/{demos,cache}`
  - radar dots (CT blue / T orange), names, trails, bomb, kill Xs / lines
  - P(win) overlay + timeline sparkline, round strip, Players / Engine / Feed
  - Stats (box, swing, key moments) + Zones (model coeffs on radar)
  - URL hash `#m=&r=&t=`
- `run.sh`, README Mac quickstart, `[web]` extra in `pyproject.toml`
- Tests: 19 core + 4 web (`tests/test_web.py`) — all green
- Manual E2E: Mirage (bundled) + Ancient upload (fit-on-demo); screenshots in
  `/tmp/shot{1..5}.png`

## How to run (verified)

```
cd /agent/poswing
pip install -e ".[web]"
./run.sh --data-dir /tmp/poswing-data --no-browser
# or: python3 -m poswing.cli serve /tmp/demos/mirage.pkl --data-dir /tmp/poswing-data
# open http://127.0.0.1:8765
```

Useful local data (outside the repo):

| Path | What |
|------|------|
| `/tmp/demos/mirage.pkl` + `.dem` | NAVI Javelins vs 9Pandas Fearless Mirage |
| `/tmp/demos/ancient.pkl` + VP vs NAVI dem | Ancient |
| `/tmp/demos/vertigo.pkl` | Vertigo |
| `/tmp/esta/pkl/mirage` | 278 ESTA Mirage pkl (~662 MB) |
| `/tmp/esta/files` | 1,558 ESTA `.json.xz` |
| `/tmp/cmp/` | holdout scripts, v1/v2 models, `graph.pkl` |
| `/tmp/poswing-data/` | running serve cache (mirage + ancient already imported) |

Canvases (earlier): compare `8e99e283-…`, round-4 `2ad4bc7f-…`.

## Next iteration order (do these)

1. **Code review pass** over `session.py`, `web/`, `engine.py` persistence,
   `cli.py` bundled defaults. Fix dead code / race in `Store.info` ready flag;
   confirm `cached_property` thread-safety under concurrent warm + request.
2. **Kill serve tmux** if still up: session `poswing-serve`.
3. **Model: contest features** — feature when both sides occupy the same zone
   (site contest / plant block); held-out eval on `/tmp/esta/pkl/mirage` (40
   matches); retrain and replace `assets/models/de_mirage.json`.
4. **Multi-map ESTA** — convert + fit + graph for `de_ancient`, `de_vertigo`
   (have CS2 demos to verify), then `de_inferno`, `de_nuke`, `de_overpass`,
   `de_dust2` (place-rename caveats like Mirage). Ship under `assets/models/` and
   `assets/graphs/`.
5. **PDF** — regenerate `docs/poswing.pdf` via `docs/build_pdf.py` (serve + assets).
6. **Polish UI** if time: mid-round seek via hash on first paint is done; engine
   panel on mobile; Vertigo lower-radar (`de_vertigo_lower.png` already in assets).
7. **Handoff** — user pulls evening; no remote → `git bundle create poswing.bundle
   --all` or they attach origin. Author for commits:
   `git -c user.name="Ferdek Dżipitek" -c user.email="ferdek@example.invalid"`.

## Known limits (unchanged)

- Engine ignores enemy response to our move; no interaction feature yet (→ contest).
- T pre-plant “execute” advice ignores entry-fight cost.
- ESTA is CS:GO 2021–22; place renames only wired for Mirage so far.
- `rating_approx` is not HLTV’s unpublished Rating 3.0 formula.

## Git tip

```
cd /agent/poswing && git log --oneline | head
# latest at checkpoint should include the web UI commit
```
