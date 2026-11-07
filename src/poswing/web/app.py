"""HTTP API + static front-end.  Run with ``poswing serve``."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__, resources
from .store import Store

log = logging.getLogger("poswing.web")
STATIC = Path(__file__).parent / "static"


class EngineRequest(BaseModel):
    round_num: int
    t: float
    side: str = Field(pattern="^(T|CT)$")
    horizon: float = Field(10.0, ge=3.0, le=40.0)
    top: int = Field(5, ge=1, le=12)
    enemy_mode: str = Field("static", pattern="^(static|predict)$")
    max_movers: int | None = Field(None, ge=1, le=5)
    beam: int = Field(40, ge=5, le=200)


def create_app(store: Store) -> FastAPI:
    app = FastAPI(title="poswing", version=__version__)
    app.state.store = store

    def session_or_404(match_id: str):
        try:
            return store.get(match_id)
        except KeyError:
            raise HTTPException(404, f"match {match_id} not found") from None

    # ------------------------------------------------------------- matches
    @app.get("/api/matches")
    def list_matches():
        return {"matches": store.list(), "jobs": [j.to_dict() for j in store.jobs.values()],
                "bundled_maps": resources.bundled_maps(), "version": __version__}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "demo.dem").name
        if not name.lower().endswith((".dem", ".pkl")):
            raise HTTPException(400, "upload a .dem (CS2 demo) file")
        target = store.demos_dir / name
        i = 2
        while target.exists():
            target = store.demos_dir / f"{Path(name).stem}-{i}{Path(name).suffix}"
            i += 1
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh, length=1 << 20)
        job = store.submit(target, display_name=name)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = store.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        return j.to_dict()

    @app.get("/api/matches/{match_id}")
    def match_summary(match_id: str):
        return session_or_404(match_id).summary()

    @app.delete("/api/matches/{match_id}")
    def delete_match(match_id: str):
        try:
            store.delete(match_id)
        except KeyError:
            raise HTTPException(404, "match not found") from None
        return {"ok": True}

    @app.get("/api/matches/{match_id}/rounds/{round_num}")
    def round_detail(match_id: str, round_num: int):
        s = session_or_404(match_id)
        try:
            return s.round_detail(round_num)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/api/matches/{match_id}/zones")
    def zones(match_id: str):
        s = session_or_404(match_id)
        return {"zones": s.zone_table(), "centroids": {z: list(xy) for z, xy in s.zone_centroids.items()}}

    @app.post("/api/matches/{match_id}/engine")
    def engine(match_id: str, req: EngineRequest):
        s = session_or_404(match_id)
        try:
            return s.suggest(req.round_num, req.t, req.side, horizon=req.horizon, top=req.top,
                             enemy_mode=req.enemy_mode, max_movers=req.max_movers, beam=req.beam)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    # --------------------------------------------------------------- assets
    @app.get("/api/radar/{map_name}.png")
    def radar(map_name: str):
        p = resources.radar_path(map_name)
        if p is None:
            raise HTTPException(404, f"no radar for {map_name}")
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/maps")
    def maps():
        return {m: resources.radar_info(m) for m in resources.radar_meta()}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        log.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    return app


def serve(data_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, sample_hz: float = 1.0,
          model_path: str | None = None, demos: list[str] | None = None, open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import uvicorn

    store = Store(Path(data_dir), sample_hz=sample_hz, model_path=model_path)
    if demos:
        jobs = store.import_paths([Path(d) for d in demos])
        if jobs:
            log.info("queued %d demo(s) for parsing", len(jobs))
    store.warm_all()
    app = create_app(store)
    url = f"http://{host}:{port}"
    print(f"poswing {__version__} - open {url}  (data in {store.data_dir})")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
