"""Win-probability model: P(T wins round | state).

Two layers, in the spirit of HLTV's Round Swing tables but extended with positions:

1. **Baseline** – what HLTV uses: map, side, players alive, bomb state, economy
   (plus the round clock).  A parametric prior is corrected by an empirical-Bayes
   lookup table so that well-populated situations follow the data while rare ones
   fall back on the prior.
2. **Positional layer** – an L2-regularised logistic regression on top of the
   baseline logit (used as an offset) whose features are "how many alive players of
   each side stand in each zone", separately for pre- and post-plant.

``p_pos - p_base`` is therefore exactly "how much the positions are worth" on top of
the man-count/eco/bomb situation.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .state import BUY_RANK, GameState


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------- baseline
def prior_logit(st: GameState) -> float:
    """Hand-set prior for P(T win). The data corrects it through the lookup table."""
    l = -0.08 + 0.9 * (st.t_alive - st.ct_alive)
    if st.bomb_planted:
        l += 1.4 - 0.15 * st.ct_alive
        l += 0.8 * (1.0 - min(st.time_left, 40.0) / 40.0)
        if st.defusing:
            l -= 0.6
    else:
        if st.time_left < 25.0:
            l -= 1.2 * (25.0 - st.time_left) / 25.0
    l += 0.35 * (BUY_RANK.get(st.t_buy, 3) - BUY_RANK.get(st.ct_buy, 3))
    return l


def bucket_key(st: GameState) -> str:
    eco = int(np.clip(BUY_RANK.get(st.t_buy, 3) - BUY_RANK.get(st.ct_buy, 3), -1, 1))
    return f"{st.map_name}|{st.t_alive}v{st.ct_alive}|{'post' if st.bomb_planted else 'pre'}|eco{eco:+d}"


@dataclass
class BaselineTable:
    pseudo_count: float = 20.0
    adjustments: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)

    def fit(self, states: list[GameState], y: np.ndarray, w: np.ndarray) -> "BaselineTable":
        sw: dict[str, float] = defaultdict(float)
        swy: dict[str, float] = defaultdict(float)
        swp: dict[str, float] = defaultdict(float)
        for st, yi, wi in zip(states, y, w):
            if st.terminal_t_win() is not None:
                continue
            k = bucket_key(st)
            sw[k] += wi
            swy[k] += wi * yi
            swp[k] += wi * _sigmoid(prior_logit(st))
        self.adjustments, self.counts = {}, {}
        for k in sw:
            p_prior = swp[k] / sw[k]
            shrunk = (swy[k] + self.pseudo_count * p_prior) / (sw[k] + self.pseudo_count)
            self.adjustments[k] = _logit(shrunk) - _logit(p_prior)
            self.counts[k] = sw[k]
        return self

    def logit(self, st: GameState) -> float:
        return prior_logit(st) + self.adjustments.get(bucket_key(st), 0.0)


# -------------------------------------------------------------- positional layer
@dataclass
class PositionalLayer:
    """L2 logistic layer on top of the baseline logit.

    ``target`` controls what the zone coefficients mean:

    * ``"horizon"`` (default) – the label of a state is the *baseline* win probability
      ``horizon`` seconds later in the same round (the round outcome if it ends sooner).
      Because the baseline at ``t`` is the offset, a coefficient measures how a position
      changes the man-count/bomb/clock situation over the next seconds, i.e. what the
      position *produces*, not what it correlates with.  A CT standing in T spawn
      because the round is already won gets no credit for the win that was already
      in the baseline.  Since ``E[P_base(t+h) | state] = P(win | state)`` up to the
      information that outlives the horizon, the sum is still a win probability.
    * ``"outcome"`` – classic: the label is the round outcome.  Larger, partly
      correlational coefficients.

    ``shrink_rounds`` makes the L2 penalty exposure-aware: a feature seen in ``n``
    weighted rounds gets ``l2 * (1 + shrink_rounds / n)``, so rare zones are pulled to
    zero harder than common ones.
    """

    l2: float = 2.0
    features: list[str] = field(default_factory=list)
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    index: dict[str, int] = field(default_factory=dict, repr=False)
    target: str = "horizon"
    horizon: float = 15.0
    shrink_rounds: float = 5.0

    @staticmethod
    def feature_names(st: GameState) -> list[tuple[str, float]]:
        """Presence of a side in a zone, extra players stacked there, and how spread the side is.

        Splitting presence from stacking gives diminishing returns (a second player in a
        zone is not worth the same as the first) and the spread/stack features are shared
        across zones, so they are well estimated even from few rounds.
        """
        ph = st.phase
        feats: list[tuple[str, float]] = []
        for side, zones in (("T", st.t_zones), ("CT", st.ct_zones)):
            if not zones:
                continue
            counts = Counter(zones)
            for z, n in counts.items():
                feats.append((f"{side}|{ph}|{z}", 1.0))
                if n > 1:
                    feats.append((f"{side}|{ph}|{z}|extra", float(n - 1)))
            feats.append((f"{side}|{ph}|spread", len(counts) / len(zones) - 0.6))
            feats.append((f"{side}|{ph}|stack", float(max(counts.values()) - 1)))
        if st.bomb_planted:
            feats.append(("time|post", (st.time_left - 20.0) / 20.0))
        else:
            feats.append(("time|pre", (st.time_left - 60.0) / 60.0))
        return feats

    def _rebuild_index(self):
        self.index = {f: i for i, f in enumerate(self.features)}

    def design(self, states: list[GameState], grow: bool = False) -> np.ndarray:
        if grow:
            for st in states:
                for f, _ in self.feature_names(st):
                    if f not in self.index:
                        self.index[f] = len(self.features)
                        self.features.append(f)
        X = np.zeros((len(states), len(self.features)))
        for i, st in enumerate(states):
            for f, v in self.feature_names(st):
                j = self.index.get(f)
                if j is not None:
                    X[i, j] += v
        return X

    def _sparse_design(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, cols, vals) triplets; grows the feature list.  Keeps memory flat for
        hundreds of thousands of states where a dense matrix would not fit."""
        rows, cols, vals = [], [], []
        for i, st in enumerate(states):
            for f, v in self.feature_names(st):
                j = self.index.get(f)
                if j is None:
                    j = self.index[f] = len(self.features)
                    self.features.append(f)
                rows.append(i)
                cols.append(j)
                vals.append(v)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=float)

    def fit(self, states: list[GameState], y: np.ndarray, w: np.ndarray, offset: np.ndarray,
            iters: int = 25, chunk: int = 50_000) -> "PositionalLayer":
        self.features, self.index = [], {}
        rows, cols, vals = self._sparse_design(states)
        n, d = len(states), len(self.features)
        beta = np.zeros(d)
        exposure = np.bincount(cols, weights=w[rows], minlength=d)  # weighted rounds per feature
        lam = self.l2 * (1.0 + self.shrink_rounds / np.maximum(exposure, 1e-6))
        # row boundaries of the triplets for chunked dense Hessian blocks
        bounds = np.searchsorted(rows, np.arange(0, n + chunk, chunk))
        for _ in range(iters):
            xb = np.bincount(rows, weights=vals * beta[cols], minlength=n)
            p = _sigmoid(offset + xb)
            grad = np.bincount(cols, weights=vals * (w * (p - y))[rows], minlength=d) + lam * beta
            H = np.diag(lam)
            s = w * p * (1 - p)
            for k in range(len(bounds) - 1):
                lo, hi = bounds[k], bounds[k + 1]
                if hi <= lo:
                    continue
                r0 = rows[lo]
                Xc = np.zeros((rows[hi - 1] - r0 + 1, d))
                np.add.at(Xc, (rows[lo:hi] - r0, cols[lo:hi]), vals[lo:hi])
                H += (Xc * s[r0:r0 + Xc.shape[0], None]).T @ Xc
            step = np.linalg.solve(H, grad)
            beta -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        self.coef = beta
        return self

    def logit_delta(self, states: list[GameState]) -> np.ndarray:
        if not self.features:
            return np.zeros(len(states))
        return self.design(states) @ self.coef


# ---------------------------------------------------------------------- facade
@dataclass
class WinProbModel:
    baseline: BaselineTable = field(default_factory=BaselineTable)
    positional: PositionalLayer = field(default_factory=PositionalLayer)
    n_states: int = 0
    n_rounds: int = 0
    maps: list[str] = field(default_factory=list)

    def fit(self, states: list[GameState], y: np.ndarray, w: np.ndarray, n_rounds: int = 0,
            round_ids: np.ndarray | None = None, times: np.ndarray | None = None) -> "WinProbModel":
        """``round_ids``/``times`` (one per state, states ordered by time within a round)
        are needed for the horizon target; without them the outcome target is used."""
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        self.baseline.fit(states, y, w)
        open_idx = np.array([i for i, st in enumerate(states) if st.terminal_t_win() is None], dtype=int)
        sub = [states[i] for i in open_idx]
        offset = np.array([self.baseline.logit(st) for st in sub])
        label = y[open_idx]
        if self.positional.target == "horizon" and round_ids is not None and times is not None and len(open_idx):
            label = self._horizon_labels(states, y, np.asarray(round_ids), np.asarray(times, dtype=float), open_idx)
        self.positional.fit(sub, label, w[open_idx], offset)
        self.n_states = len(states)
        self.n_rounds = n_rounds
        self.maps = sorted({st.map_name for st in states})
        return self

    def _horizon_labels(self, states, y, round_ids, times, open_idx) -> np.ndarray:
        """Baseline P(T) ``horizon`` seconds later in the same round (outcome if the round ends first)."""
        h = self.positional.horizon
        p_base = np.array([self.baseline.logit(st) for st in states])
        p_base = _sigmoid(p_base)
        for i, st in enumerate(states):
            term = st.terminal_t_win()
            if term is not None:
                p_base[i] = term
        label = y[open_idx].copy()
        pos_in_open = {int(i): k for k, i in enumerate(open_idx)}
        order = np.lexsort((times, round_ids))
        rid_sorted = round_ids[order]
        bounds = np.flatnonzero(np.r_[True, rid_sorted[1:] != rid_sorted[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = order[a:b]
            t = times[idx]
            j = np.searchsorted(t, t + h, side="left")
            for k, i in enumerate(idx):
                k_open = pos_in_open.get(int(i))
                if k_open is None:
                    continue
                if j[k] < len(idx):
                    label[k_open] = p_base[idx[j[k]]]
        return label

    # -- prediction ---------------------------------------------------------
    def predict(self, states: list[GameState], positional: bool = True) -> np.ndarray:
        base = np.array([self.baseline.logit(st) for st in states])
        if positional:
            base = base + self.positional.logit_delta(states)
        p = _sigmoid(base)
        for i, st in enumerate(states):
            term = st.terminal_t_win()
            if term is not None:
                p[i] = term
        return p

    def p_t(self, st: GameState, positional: bool = True) -> float:
        return float(self.predict([st], positional)[0])

    def p_side(self, st: GameState, side: str, positional: bool = True) -> float:
        p = self.p_t(st, positional)
        return p if side == "T" else 1.0 - p

    def positional_edge(self, st: GameState, side: str) -> float:
        """How much the positions add to the side's win probability vs. the baseline."""
        return self.p_side(st, side, True) - self.p_side(st, side, False)

    def zone_coefficients(self) -> list[dict]:
        rows = []
        for f, b in zip(self.positional.features, self.positional.coef):
            parts = f.split("|")
            if len(parts) != 3 or parts[2] in ("spread", "stack"):
                continue
            side, phase, zone = parts
            # effect for the side that stands there, in percentage points at a coin-flip state
            signed = b if side == "T" else -b
            rows.append({"side": side, "phase": phase, "zone": zone, "logit": float(signed),
                         "pp_at_50": float(signed) / 4.0 * 100.0})
        return rows

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "baseline": {"pseudo_count": self.baseline.pseudo_count,
                          "adjustments": self.baseline.adjustments, "counts": self.baseline.counts},
            "positional": {"l2": self.positional.l2, "features": self.positional.features,
                            "coef": self.positional.coef.tolist()},
            "n_states": self.n_states, "n_rounds": self.n_rounds, "maps": self.maps,
        }
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "WinProbModel":
        d = json.loads(Path(path).read_text())
        m = cls()
        m.baseline = BaselineTable(d["baseline"]["pseudo_count"], d["baseline"]["adjustments"], d["baseline"]["counts"])
        m.positional = PositionalLayer(d["positional"]["l2"], d["positional"]["features"], np.array(d["positional"]["coef"]))
        m.positional._rebuild_index()
        m.n_states, m.n_rounds, m.maps = d["n_states"], d["n_rounds"], d["maps"]
        return m
