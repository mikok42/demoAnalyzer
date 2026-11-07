"""Registry of demos on disk plus background parse / analysis jobs.

Layout of the data directory (default ``~/.poswing``)::

    demos/   uploaded .dem files
    cache/   parsed .pkl files (one per demo; re-used across restarts)
"""
from __future__ import annotations

import logging
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..data import MatchData
from ..session import MatchSession
from ..zones import Zoner

log = logging.getLogger("poswing.web")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:80] or "demo"


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued | parsing | analysing | done | error
    progress: float = 0.0
    message: str = ""
    match_id: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def set(self, status: str | None = None, progress: float | None = None, message: str | None = None) -> None:
        if status is not None:
            self.status = status
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        self.updated = time.time()

    def to_dict(self) -> dict:
        return {"id": self.id, "filename": self.filename, "status": self.status, "progress": self.progress,
                "message": self.message, "match_id": self.match_id, "created": self.created, "updated": self.updated}


@dataclass
class Entry:
    id: str
    pkl: Path
    name: str
    _session: MatchSession | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _info: dict | None = None

    def session(self, zoner: Zoner, model_path: str | None) -> MatchSession:
        with self._lock:
            if self._session is None:
                self._session = MatchSession(MatchData.load(self.pkl), zoner=zoner, model_path=model_path)
            return self._session

    def info(self, zoner: Zoner, model_path: str | None) -> dict:
        if self._info is None:
            s = self.session(zoner, model_path)
            m = s.match
            score: dict[str, int] = {}
            for r in m.rounds.itertuples(index=False):
                w = r.t_team if r.winner == "T" else r.ct_team
                score[str(w)] = score.get(str(w), 0) + 1
            self._info = {"id": self.id, "name": self.name, "map_name": m.map_name, "teams": m.team_names(),
                          "n_rounds": m.n_rounds, "score": score, "source": m.source,
                          "ready": self._session is not None and "swing" in self._session.__dict__}
        self._info["ready"] = self._session is not None and "swing" in self._session.__dict__
        return self._info


class Store:
    def __init__(self, data_dir: Path, sample_hz: float = 1.0, model_path: str | None = None, workers: int = 1):
        self.data_dir = Path(data_dir).expanduser()
        self.demos_dir = self.data_dir / "demos"
        self.cache_dir = self.data_dir / "cache"
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_hz = sample_hz
        self.model_path = model_path
        self.zoner = Zoner()
        self.entries: dict[str, Entry] = {}
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="poswing-parse")
        self._warm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="poswing-warm")
        self.scan()

    # ------------------------------------------------------------- registry
    def scan(self) -> None:
        for pkl in sorted(self.cache_dir.glob("*.pkl")):
            self._register(pkl)

    def _register(self, pkl: Path) -> Entry:
        with self._lock:
            mid = pkl.stem
            if mid in self.entries:
                return self.entries[mid]
            e = Entry(id=mid, pkl=pkl, name=pkl.stem)
            self.entries[mid] = e
            return e

    def _unique_id(self, stem: str) -> str:
        base = slugify(stem)
        mid, i = base, 2
        while mid in self.entries or (self.cache_dir / f"{mid}.pkl").exists():
            mid = f"{base}-{i}"
            i += 1
        return mid

    def list(self) -> list[dict]:
        out = []
        for e in list(self.entries.values()):
            try:
                out.append(e.info(self.zoner, self.model_path))
            except Exception as exc:  # noqa: BLE001 - a broken cache file should not hide the others
                out.append({"id": e.id, "name": e.name, "error": str(exc)})
        return sorted(out, key=lambda d: d["name"])

    def get(self, match_id: str) -> MatchSession:
        e = self.entries.get(match_id)
        if e is None:
            raise KeyError(match_id)
        return e.session(self.zoner, self.model_path)

    def delete(self, match_id: str) -> None:
        with self._lock:
            e = self.entries.pop(match_id, None)
        if e is None:
            raise KeyError(match_id)
        e.pkl.unlink(missing_ok=True)
        for dem in self.demos_dir.glob(f"{match_id}.*"):
            dem.unlink(missing_ok=True)

    # ----------------------------------------------------------------- jobs
    def submit(self, path: Path, display_name: str | None = None) -> Job:
        """Parse a .dem (or import a .pkl) in the background."""
        job = Job(id=uuid.uuid4().hex[:10], filename=display_name or path.name)
        self.jobs[job.id] = job
        self._pool.submit(self._run_job, job, Path(path))
        return job

    def import_paths(self, paths: list[Path]) -> list[Job]:
        jobs = []
        for p in paths:
            p = Path(p)
            files = sorted([*p.glob("*.dem"), *p.glob("*.pkl")]) if p.is_dir() else [p]
            for f in files:
                if f.suffix == ".pkl" and f.parent.resolve() == self.cache_dir.resolve():
                    continue
                if (self.cache_dir / f"{slugify(f.stem)}.pkl").exists():
                    self._register(self.cache_dir / f"{slugify(f.stem)}.pkl")
                    continue
                jobs.append(self.submit(f))
        return jobs

    def _run_job(self, job: Job, path: Path) -> None:
        try:
            mid = self._unique_id(path.stem)
            target = self.cache_dir / f"{mid}.pkl"
            if path.suffix == ".pkl":
                job.set("parsing", 0.3, "importing cached match")
                m = MatchData.load(path)
                m.save(target)
            else:
                from ..parse import parse_demo

                job.set("parsing", 0.1, "parsing demo (this takes 10-60 s)")
                t0 = time.time()
                m = parse_demo(path, sample_hz=self.sample_hz)
                m.save(target)
                log.info("parsed %s in %.1fs: %s, %d rounds", path.name, time.time() - t0, m.map_name, m.n_rounds)
            e = self._register(target)
            e.name = mid
            job.match_id = mid
            job.set("analysing", 0.7, f"{m.map_name}, {m.n_rounds} rounds - computing win probabilities and swing")
            s = e.session(self.zoner, self.model_path)
            s.warm()
            job.set("done", 1.0, "ready")
        except Exception as exc:  # noqa: BLE001 - reported to the client
            log.error("job %s failed: %s\n%s", job.id, exc, traceback.format_exc())
            job.set("error", 1.0, f"{type(exc).__name__}: {exc}")

    def warm_all(self) -> None:
        """Analyse every registered match in the background (so the first click is instant)."""
        for e in list(self.entries.values()):
            self._warm_pool.submit(lambda e=e: e.session(self.zoner, self.model_path).warm())
