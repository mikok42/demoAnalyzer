/* poswing web UI - vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const COLORS = { CT: "#4aa3ff", T: "#f5a524" };
  const ROUND_SECONDS = 115, BOMB_SECONDS = 40;

  const S = {
    matches: [], jobs: [], matchId: null, summary: null, round: null, roundNum: null,
    t: 0, playing: false, speed: 2, lastFrame: null, radarImg: null, radarMeta: null,
    engine: null, engineLine: null, engineSide: "CT", engineEnemy: "static", engineMovers: "",
    zones: null, zoneSide: "CT", zonePhase: "pre", view: "replay", hoverSid: null,
  };

  // ------------------------------------------------------------------ helpers
  const fmt1 = (v) => (v == null ? "-" : (Math.round(v * 10) / 10).toFixed(1));
  const fmt2 = (v) => (v == null ? "-" : (Math.round(v * 100) / 100).toFixed(2));
  const pp = (v, digits = 1) => (v == null ? "-" : (v >= 0 ? "+" : "") + v.toFixed(digits) + " pp");
  const clock = (sec) => { sec = Math.max(0, sec); const m = Math.floor(sec / 60), s = Math.floor(sec % 60); return `${m}:${String(s).padStart(2, "0")}`; };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function toast(msg, isError = false) {
    const el = $("#toast");
    el.textContent = msg; el.classList.toggle("error", isError); el.classList.remove("hidden");
    clearTimeout(toast._h); toast._h = setTimeout(() => el.classList.add("hidden"), isError ? 6000 : 2500);
  }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (e) { /* ignore */ }
      throw new Error(detail);
    }
    return r.json();
  }

  // -------------------------------------------------------------- match list
  async function refreshMatches() {
    try {
      const d = await api("/api/matches");
      S.matches = d.matches; S.jobs = d.jobs;
      $("#foot").textContent = `poswing ${d.version} - bundled models: ${d.bundled_maps.join(", ") || "none"}`;
      renderMatchList(); renderJobs();
      if (!S.matchId && d.matches.length && !S._openedOnce) {
        S._openedOnce = true;
        const h = parseHash();
        const want = d.matches.find((m) => m.id === h.m) || d.matches[0];
        if (want && !want.error) openMatch(want.id, h.m === want.id ? +h.r || null : null);
      }
      const active = d.jobs.some((j) => j.status !== "done" && j.status !== "error");
      if (active) setTimeout(refreshMatches, 1000);
    } catch (e) { toast("cannot reach the server: " + e.message, true); }
  }

  function renderMatchList() {
    const ul = $("#match-list"); ul.innerHTML = "";
    if (!S.matches.length) { ul.innerHTML = '<li class="muted small">no demos yet</li>'; return; }
    for (const m of S.matches) {
      const li = document.createElement("li");
      li.className = m.id === S.matchId ? "active" : "";
      if (m.error) { li.innerHTML = `<div class="m-teams">${esc(m.name)}</div><div class="m-sub neg">${esc(m.error)}</div>`; ul.appendChild(li); continue; }
      const teams = m.teams || [];
      const score = teams.map((t) => m.score[t] || 0);
      li.innerHTML = `<button class="del" title="delete">&times;</button>
        <div class="m-teams">${esc(teams[0] || "?")} <span class="muted">${score[0] ?? ""}:${score[1] ?? ""}</span> ${esc(teams[1] || "?")}</div>
        <div class="m-sub"><span>${esc(m.map_name)} - ${m.n_rounds} rounds</span><span>${m.ready ? "" : "analysing..."}</span></div>`;
      li.onclick = (ev) => { if (ev.target.classList.contains("del")) return; openMatch(m.id); };
      $(".del", li).onclick = async (ev) => {
        ev.stopPropagation();
        if (!confirm(`Delete ${m.name}?`)) return;
        await api(`/api/matches/${m.id}`, { method: "DELETE" });
        if (S.matchId === m.id) { S.matchId = null; $("#match-view").classList.add("hidden"); $("#empty-state").classList.remove("hidden"); }
        refreshMatches();
      };
      ul.appendChild(li);
    }
  }

  function renderJobs() {
    const box = $("#jobs"); box.innerHTML = "";
    const recent = S.jobs.filter((j) => j.status !== "done" || Date.now() / 1000 - j.updated < 20);
    for (const j of recent) {
      const div = document.createElement("div");
      div.className = "job " + j.status;
      div.innerHTML = `<div><b>${esc(j.filename)}</b> <span class="muted">${j.status}</span></div>
        <div class="msg">${esc(j.message)}</div><div class="bar"><i style="width:${Math.round(j.progress * 100)}%"></i></div>`;
      box.appendChild(div);
      if (j.status === "done" && j.match_id && S._autoOpen === j.id) { S._autoOpen = null; openMatch(j.match_id); }
    }
  }

  async function uploadFiles(files) {
    for (const f of files) {
      const fd = new FormData(); fd.append("file", f);
      toast(`uploading ${f.name} (${(f.size / 1e6).toFixed(0)} MB)...`);
      try {
        const job = await api("/api/upload", { method: "POST", body: fd });
        S._autoOpen = job.id;
        refreshMatches();
      } catch (e) { toast("upload failed: " + e.message, true); }
    }
  }

  // ------------------------------------------------------------------- match
  function parseHash() {
    const out = {};
    for (const kv of location.hash.replace(/^#/, "").split("&")) { const [k, v] = kv.split("="); if (k) out[k] = decodeURIComponent(v || ""); }
    return out;
  }
  function writeHash() {
    if (!S.matchId) return;
    const h = `#m=${encodeURIComponent(S.matchId)}${S.roundNum != null ? `&r=${S.roundNum}` : ""}`;
    if (location.hash !== h) history.replaceState(null, "", h);
  }

  async function openMatch(id, roundNum = null) {
    try {
      S.matchId = id;
      const sum = await api(`/api/matches/${id}`);
      S.summary = sum; S.round = null; S.engine = null; S.engineLine = null; S.zones = null;
      renderMatchList();
      $("#empty-state").classList.add("hidden"); $("#match-view").classList.remove("hidden");
      const [a, b] = sum.teams;
      $("#team-a").textContent = a; $("#team-b").textContent = b || "";
      $("#score-a").textContent = sum.final_score[a] || 0; $("#score-b").textContent = sum.final_score[b] || 0;
      $("#map-name").textContent = sum.map_name;
      $("#model-badge").textContent = sum.model.source === "bundled" ? `model: ${sum.model.n_rounds.toLocaleString()} pro rounds` : `model: ${sum.model.source} (${sum.model.n_rounds} rounds)`;
      $("#match-title").textContent = `${a} vs ${b} - ${sum.map_name}`;
      await loadRadar(sum.map_name, sum.radar);
      renderRoundStrip(); renderStats();
      const wanted = sum.rounds.find((r) => r.round_num === roundNum) || sum.rounds.find((r) => r.planted) || sum.rounds[0];
      openRound(wanted.round_num);
      if (S.view === "zones") loadZones();
    } catch (e) { toast(e.message, true); }
  }

  function loadRadar(mapName, meta) {
    return new Promise((resolve) => {
      S.radarMeta = meta; S.radarImg = null;
      if (!meta) { resolve(); return; }
      const img = new Image();
      img.onload = () => { S.radarImg = img; drawRadar(); drawZoneRadar(); resolve(); };
      img.onerror = () => resolve();
      img.src = `/api/radar/${mapName}.png`;
    });
  }

  function renderRoundStrip() {
    const strip = $("#round-strip"); strip.innerHTML = "";
    const rounds = S.summary.rounds;
    rounds.forEach((r, i) => {
      if (i === 12 || (i > 24 && (i - 24) % 6 === 0 && i < rounds.length)) { const h = document.createElement("span"); h.className = "half"; strip.appendChild(h); }
      const c = document.createElement("div");
      c.className = `chip ${r.winner.toLowerCase()} ${r.planted ? "planted" : ""} ${r.round_num === S.roundNum ? "active" : ""}`;
      c.textContent = r.round_num;
      const sa = r.score_after;
      c.title = `R${r.round_num}: ${r.winner_team} (${r.reason}) - ${Object.entries(sa).map(([k, v]) => `${k} ${v}`).join(", ")}\nT ${r.t_buy} vs CT ${r.ct_buy}`;
      c.onclick = () => openRound(r.round_num);
      strip.appendChild(c);
    });
  }

  // ------------------------------------------------------------------- round
  async function openRound(n, seekTo = 0) {
    if (!S.matchId) return;
    try {
      S.playing = false; updatePlayBtn();
      const rd = await api(`/api/matches/${S.matchId}/rounds/${n}`);
      S.round = rd; S.roundNum = n; S.t = seekTo; S.engine = null; S.engineLine = null;
      $("#seek").max = rd.duration;
      $("#eng-result").innerHTML = "";
      renderRoundStrip(); renderFeed(); renderRoundInfo();
      writeHash();
      drawAll();
    } catch (e) { toast(e.message, true); }
  }

  function renderRoundInfo() {
    const r = S.round, sum = S.summary;
    const meta = sum.rounds.find((x) => x.round_num === r.round_num);
    const sb = meta.score_before;
    $("#round-info").innerHTML = `Round ${r.round_num} - ${Object.entries(sb).map(([k, v]) => `${esc(k)} ${v}`).join(" : ")}<br>` +
      `<span class="t">T</span> ${esc(r.t_team)} (${esc(r.t_buy)}) vs <span class="ct">CT</span> ${esc(r.ct_team)} (${esc(r.ct_buy)})<br>` +
      `won by <b>${esc(meta.winner_team)}</b> - ${esc(r.reason.replace(/_/g, " "))}`;
  }

  // frame interpolation ------------------------------------------------------
  function frameAt(t) {
    const fr = S.round.frames;
    if (!fr.length) return null;
    let lo = 0, hi = fr.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (fr[mid].t <= t) lo = mid; else hi = mid - 1; }
    const a = fr[lo], b = fr[Math.min(lo + 1, fr.length - 1)];
    const span = b.t - a.t;
    const u = span > 0 && span < 3 ? Math.min(1, Math.max(0, (t - a.t) / span)) : 0;
    const nextBy = new Map(b.players.map((p) => [p[0], p]));
    const players = a.players.map((p) => {
      const q = nextBy.get(p[0]);
      if (!q || !p[1] || !q[1] || u === 0) return { sid: p[0], alive: p[1], x: p[2], y: p[3], z: p[4], hp: p[5], zone: p[6] };
      return { sid: p[0], alive: p[1], x: p[2] + (q[2] - p[2]) * u, y: p[3] + (q[3] - p[3]) * u, z: p[4], hp: p[5], zone: u > 0.5 ? q[6] : p[6] };
    });
    const pT = a.p_t != null && b.p_t != null ? a.p_t + (b.p_t - a.p_t) * u : a.p_t;
    const pBase = a.p_base != null && b.p_base != null ? a.p_base + (b.p_base - a.p_base) * u : a.p_base;
    return { t, players, p_t: pT, p_base: pBase, frame: a };
  }

  // radar drawing ------------------------------------------------------------
  function radarCtx(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth || 600;
    if (canvas.width !== Math.round(size * dpr)) { canvas.width = Math.round(size * dpr); canvas.height = Math.round(size * dpr); }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, size };
  }

  function worldToPx(x, y, size) {
    const m = S.radarMeta;
    if (!m) return [size / 2, size / 2];
    const k = size / m.size;
    return [((x - m.pos_x) / m.scale) * k, ((m.pos_y - y) / m.scale) * k];
  }

  function drawBase(ctx, size) {
    ctx.clearRect(0, 0, size, size);
    if (S.radarImg) { ctx.globalAlpha = 0.85; ctx.drawImage(S.radarImg, 0, 0, size, size); ctx.globalAlpha = 1; }
    else { ctx.fillStyle = "#13171d"; ctx.fillRect(0, 0, size, size); ctx.fillStyle = "#555"; ctx.font = "13px sans-serif"; ctx.fillText("no radar image for this map - positions drawn on a blank canvas", 16, 26); }
  }

  function drawRadar() {
    const canvas = $("#radar"); if (!canvas) return;
    const { ctx, size } = radarCtx(canvas);
    drawBase(ctx, size);
    if (!S.round) return;
    const f = frameAt(S.t); if (!f) return;
    S.lastFrame = f;
    const r = S.round, roster = r.roster;
    const showNames = $("#chk-names").checked, showZones = $("#chk-zones").checked, showTrails = $("#chk-trails").checked;
    const lower = S.radarMeta && S.radarMeta.z_cutoff != null;

    if (showZones && S.zones) {
      ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillStyle = "rgba(255,255,255,.55)";
      for (const [z, xy] of Object.entries(S.zones.centroids)) { const [px, py] = worldToPx(xy[0], xy[1], size); ctx.fillText(z, px, py); }
    }

    // trails: last 6 s of movement
    if (showTrails) {
      const fr = r.frames, from = S.t - 6;
      const paths = new Map();
      for (const fa of fr) {
        if (fa.t < from || fa.t > S.t) continue;
        for (const p of fa.players) { if (!p[1]) continue; if (!paths.has(p[0])) paths.set(p[0], []); paths.get(p[0]).push(p); }
      }
      ctx.lineWidth = 2;
      for (const [sid, pts] of paths) {
        if (pts.length < 2) continue;
        ctx.strokeStyle = COLORS[roster[sid].team] + "66"; ctx.beginPath();
        pts.forEach((p, i) => { const [px, py] = worldToPx(p[2], p[3], size); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
        ctx.stroke();
      }
    }

    // bomb
    const plant = r.bomb.find((b) => b.event === "plant" && b.t <= S.t);
    const defused = r.bomb.find((b) => b.event === "defuse" && b.t <= S.t);
    if (plant && plant.pos) {
      const [px, py] = worldToPx(plant.pos[0], plant.pos[1], size);
      const blink = !defused && Math.floor(S.t * 2) % 2 === 0;
      ctx.fillStyle = defused ? "#3ecf8e" : blink ? "#ff5d5d" : "#aa2b2b";
      ctx.beginPath(); ctx.moveTo(px, py - 8); ctx.lineTo(px + 8, py); ctx.lineTo(px, py + 8); ctx.lineTo(px - 8, py); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
    }

    // deaths so far
    for (const k of r.kills) {
      if (k.t > S.t || k.vx == null) continue;
      const [px, py] = worldToPx(k.vx, k.vy, size);
      ctx.strokeStyle = COLORS[k.victim_team] + "99"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px - 5, py - 5); ctx.lineTo(px + 5, py + 5); ctx.moveTo(px + 5, py - 5); ctx.lineTo(px - 5, py + 5); ctx.stroke();
    }
    // recent kill lines (2 s)
    for (const k of r.kills) {
      const age = S.t - k.t;
      if (age < 0 || age > 2 || k.ax == null || k.vx == null) continue;
      const [ax, ay] = worldToPx(k.ax, k.ay, size), [vx, vy] = worldToPx(k.vx, k.vy, size);
      ctx.strokeStyle = `rgba(255,255,255,${0.9 * (1 - age / 2)})`; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(vx, vy); ctx.stroke(); ctx.setLineDash([]);
    }

    // engine arrows
    if (S.engine && S.engineLine != null && S.engine.lines[S.engineLine]) {
      const line = S.engine.lines[S.engineLine];
      for (const mv of line.moves) {
        const me = f.players.find((p) => p.sid === mv.steamid);
        if (!me || mv.to_xy[0] == null) continue;
        const [x0, y0] = worldToPx(me.x, me.y, size), [x1, y1] = worldToPx(mv.to_xy[0], mv.to_xy[1], size);
        arrow(ctx, x0, y0, x1, y1, "#ffffff");
        ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center"; ctx.fillText(mv.to, x1, y1 - 10);
      }
    }

    // players
    const rad = Math.max(5, size / 110);
    ctx.font = `${Math.max(10, size / 60)}px sans-serif`; ctx.textAlign = "left";
    for (const p of f.players) {
      if (!p.alive) continue;
      const team = roster[p.sid]?.team || "T";
      const [px, py] = worldToPx(p.x, p.y, size);
      const isLower = lower && p.z < S.radarMeta.z_cutoff;
      ctx.beginPath(); ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[team]; ctx.globalAlpha = isLower ? 0.55 : 1; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = S.hoverSid === p.sid ? 3 : 1.5; ctx.strokeStyle = S.hoverSid === p.sid ? "#fff" : "#0008"; ctx.stroke();
      if (p.hp < 100) { // health ring
        ctx.beginPath(); ctx.arc(px, py, rad + 2.5, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * p.hp / 100);
        ctx.strokeStyle = p.hp > 50 ? "#3ecf8e" : p.hp > 25 ? "#f5a524" : "#ff5d5d"; ctx.lineWidth = 2; ctx.stroke();
      }
      if (showNames) {
        const name = roster[p.sid]?.name || p.sid;
        ctx.fillStyle = "#000"; ctx.fillText(name, px + rad + 3, py + 4);
        ctx.fillStyle = "#fff"; ctx.fillText(name, px + rad + 2, py + 3);
      }
    }
    if (f.p_t != null) {
      const pct = Math.round(100 * (1 - f.p_t));
      $("#pwin").innerHTML = `<div class="big"><span class="ct">CT ${pct}%</span> <span class="muted">|</span> <span class="t">T ${100 - pct}%</span></div>
        <div class="bar"><i style="width:${pct}%"></i></div>
        <div class="sub">baseline without positions: CT ${Math.round(100 * (1 - f.p_base))}%</div>`;
    }
    // clock
    const planted = r.plant_t != null && S.t >= r.plant_t;
    const clk = $("#clock");
    clk.textContent = planted ? clock(BOMB_SECONDS - (S.t - r.plant_t)) : clock(ROUND_SECONDS - S.t);
    clk.classList.toggle("bomb", planted);
    renderPlayers(f);
  }

  function arrow(ctx, x0, y0, x1, y1, color) {
    const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy); if (len < 2) return;
    const ux = dx / len, uy = dy / len;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * 8, y1 - uy * 8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - ux * 12 - uy * 6, y1 - uy * 12 + ux * 6); ctx.lineTo(x1 - ux * 12 + uy * 6, y1 - uy * 12 - ux * 6); ctx.closePath(); ctx.fill();
  }

  // timeline -----------------------------------------------------------------
  function drawTimeline() {
    const canvas = $("#timeline"); if (!canvas || !S.round) return;
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const r = S.round, dur = r.duration || 1, padL = 34, padR = 8, padT = 8, padB = 18;
    const X = (t) => padL + (t / dur) * (W - padL - padR);
    const Y = (p) => padT + (1 - p) * (H - padT - padB);  // p = P(CT)
    // axes
    ctx.fillStyle = "#8b95a5"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
    for (const p of [0, 0.5, 1]) { ctx.fillText(`${Math.round(p * 100)}%`, padL - 4, Y(p) + 3); ctx.strokeStyle = p === 0.5 ? "#3a4555" : "#232b36"; ctx.beginPath(); ctx.moveTo(padL, Y(p)); ctx.lineTo(W - padR, Y(p)); ctx.stroke(); }
    ctx.textAlign = "center";
    for (let t = 0; t <= dur; t += 15) { ctx.fillText(clock(Math.max(0, ROUND_SECONDS - t)), X(t), H - 5); }
    // plant marker
    if (r.plant_t != null) { ctx.strokeStyle = "#ff5d5d"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(r.plant_t), padT); ctx.lineTo(X(r.plant_t), H - padB); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = "#ff5d5d"; ctx.fillText("plant " + (r.plant_site || ""), X(r.plant_t), padT + 8); }
    // fills: CT advantage above .5 -> blue, below -> orange
    const fr = r.frames.filter((f) => f.p_t != null);
    if (fr.length) {
      const fill = (color, cond) => {
        ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(X(fr[0].t), Y(0.5));
        for (const f of fr) { const p = 1 - f.p_t; ctx.lineTo(X(f.t), Y(cond(p) ? p : 0.5)); }
        ctx.lineTo(X(fr[fr.length - 1].t), Y(0.5)); ctx.closePath(); ctx.fill();
      };
      fill("rgba(74,163,255,.3)", (p) => p > 0.5); fill("rgba(245,165,36,.3)", (p) => p < 0.5);
      ctx.strokeStyle = "#9aa7bb"; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.beginPath();
      fr.forEach((f, i) => { const y = Y(1 - f.p_base); if (i === 0) ctx.moveTo(X(f.t), y); else ctx.lineTo(X(f.t), y); }); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = "#e6e9ef"; ctx.lineWidth = 1.8; ctx.beginPath();
      fr.forEach((f, i) => { const y = Y(1 - f.p_t); if (i === 0) ctx.moveTo(X(f.t), y); else ctx.lineTo(X(f.t), y); }); ctx.stroke();
    }
    // kills
    for (const k of r.kills) { ctx.fillStyle = COLORS[k.attacker_team] || "#fff"; ctx.beginPath(); ctx.arc(X(k.t), H - padB - 4, 3, 0, Math.PI * 2); ctx.fill(); }
    // engine horizon
    if (S.engine && S.engine.round_num === r.round_num) { ctx.fillStyle = "rgba(124,156,255,.18)"; ctx.fillRect(X(S.engine.t), padT, X(Math.min(dur, S.engine.t + S.engine.horizon)) - X(S.engine.t), H - padT - padB); }
    // cursor
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(S.t), padT); ctx.lineTo(X(S.t), H - padB); ctx.stroke();
    $("#seek").value = S.t; $("#time-label").textContent = `${S.t.toFixed(1)} s`;
  }

  function drawAll() { drawRadar(); drawTimeline(); }

  // players panel ------------------------------------------------------------
  function renderPlayers(f) {
    const box = $("#tab-players"); if (!box.classList.contains("active")) return;
    const r = S.round, roster = r.roster;
    const swingSoFar = new Map();
    for (const e of r.events) { if (e.t <= S.t + 1e-6) swingSoFar.set(e.steamid, (swingSoFar.get(e.steamid) || 0) + e.delta); }
    const total = new Map((S.summary.swing || []).map((p) => [p.steamid, p]));
    let html = "";
    for (const side of ["CT", "T"]) {
      const team = side === "CT" ? r.ct_team : r.t_team;
      const ps = f.players.filter((p) => roster[p.sid]?.team === side).sort((a, b) => (b.alive - a.alive) || (roster[a.sid].name.localeCompare(roster[b.sid].name)));
      const alive = ps.filter((p) => p.alive).length;
      html += `<div class="team-block"><h4><span class="${side.toLowerCase()}">${side} - ${esc(team)}</span><span class="muted">${alive} alive</span></h4>`;
      for (const p of ps) {
        const sw = swingSoFar.get(p.sid) || 0, tot = total.get(p.sid);
        html += `<div class="player ${p.alive ? "" : "dead"}" data-sid="${p.sid}">
          <span class="swatch" style="background:${COLORS[side]}"></span>
          <div><div class="nm">${esc(roster[p.sid].name)}</div><div class="zn">${p.alive ? esc(p.zone) : "dead"}</div>${p.alive ? `<div class="hp"><i style="width:${p.hp}%"></i></div>` : ""}</div>
          <div class="sw ${sw >= 0 ? "pos" : "neg"}">${pp(100 * sw)}<small>${tot ? "match " + pp(tot.swing_pct) + "/r" : ""}</small></div></div>`;
      }
      html += "</div>";
    }
    html += `<div class="muted small">Swing = change of the team's win probability credited to the player so far in this round (kills, damage, flashes, trades, plants, deaths and positional terms).</div>`;
    box.innerHTML = html;
    $$(".player", box).forEach((el) => { el.onmouseenter = () => { S.hoverSid = +el.dataset.sid; drawRadar(); }; el.onmouseleave = () => { S.hoverSid = null; drawRadar(); }; });
  }

  // feed -----------------------------------------------------------------------
  function renderFeed() {
    const r = S.round, box = $("#tab-feed");
    const byTick = new Map();
    for (const e of r.events) { if (e.component !== "swing") continue; const k = `${e.tick}:${e.kind}:${e.steamid}`; byTick.set(k, e); }
    const items = [];
    for (const k of r.kills) {
      const credit = r.events.filter((e) => e.tick === k.tick && e.steamid === k.attacker && e.component === "swing" && e.kind === "kill").reduce((a, e) => a + e.delta, 0);
      const tags = [k.headshot ? "HS" : "", k.is_trade ? "trade" : "", k.failed_trade ? "failed trade" : "", k.trade_denial ? "denial" : ""].filter(Boolean);
      items.push({ t: k.t, html: `<span class="tm">${clock(ROUND_SECONDS - k.t)}</span><span><b class="${(k.attacker_team || "").toLowerCase()}">${esc(k.attacker_name || "?")}</b> <span class="muted">${esc(k.weapon)}</span> <b class="${k.victim_team.toLowerCase()}">${esc(k.victim_name)}</b>${tags.map((t) => `<span class="tag">${t}</span>`).join("")}<span class="delta ${credit >= 0 ? "pos" : "neg"}">${pp(100 * credit)}</span></span>` });
    }
    for (const b of r.bomb) {
      items.push({ t: b.t, cls: "bomb", html: `<span class="tm">${clock(ROUND_SECONDS - b.t)}</span><span>${esc(b.event.replace("_", " "))}${b.site ? " " + b.site : ""} <span class="muted">${esc(b.player_name || "")}</span></span>` });
    }
    items.sort((a, b) => a.t - b.t);
    box.innerHTML = items.map((it) => `<div class="feed-item ${it.cls || ""}" data-t="${it.t}">${it.html}</div>`).join("") || '<div class="muted">no events</div>';
    $$(".feed-item", box).forEach((el) => { el.onclick = () => { S.t = Math.max(0, +el.dataset.t - 2); S.playing = false; updatePlayBtn(); drawAll(); }; });
  }

  // engine ---------------------------------------------------------------------
  async function runEngine() {
    if (!S.round) return;
    const btn = $("#eng-run"); btn.disabled = true; btn.textContent = "thinking...";
    try {
      const body = { round_num: S.roundNum, t: S.t, side: S.engineSide, horizon: +$("#eng-h").value, enemy_mode: S.engineEnemy, top: 6 };
      if (S.engineMovers) body.max_movers = +S.engineMovers;
      S.engine = await api(`/api/matches/${S.matchId}/engine`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      S.engineLine = S.engine.lines.findIndex((l) => l.moves.length > 0);
      if (S.engineLine < 0) S.engineLine = 0;
      renderEngine(); drawAll();
    } catch (e) { toast(e.message, true); }
    finally { btn.disabled = false; btn.textContent = "Analyse this moment"; }
  }

  function renderEngine() {
    const e = S.engine, box = $("#eng-result");
    if (!e) { box.innerHTML = ""; return; }
    const side = e.side.toLowerCase();
    let html = `<div class="now"><span>P(win) <b class="${side}">${esc(e.team)}</b> now</span><span><b>${(100 * e.p_now).toFixed(1)}%</b> <span class="muted small">base ${(100 * e.p_base).toFixed(1)}%</span></span></div>`;
    e.lines.forEach((l, i) => {
      const mv = l.moves.length ? l.moves.map((m) => `<b>${esc(m.name)}</b> ${esc(m.from)} &rarr; ${esc(m.to)}`).join("<br>") : "hold current positions";
      html += `<div class="line ${i === S.engineLine ? "active" : ""}" data-i="${i}"><div class="hd"><span>${(100 * l.p_win).toFixed(1)}%</span><span class="${l.delta_pp >= 0 ? "pos" : "neg"}">${pp(l.delta_pp)}</span></div><div class="mv">${mv}</div></div>`;
    });
    html += `<div class="sit">${esc(e.situation)}\nhorizon ${e.horizon}s, enemy ${e.enemy_mode}; the shaded band on the timeline is the horizon.</div>`;
    box.innerHTML = html;
    $$(".line", box).forEach((el) => { el.onclick = () => { S.engineLine = +el.dataset.i; renderEngine(); drawRadar(); }; });
  }

  // stats view -------------------------------------------------------------------
  const BOX_COLS = [["name", "player"], ["team", "team"], ["rounds", "R"], ["kills", "K"], ["deaths", "D"], ["assists", "A"], ["adr", "ADR"], ["kast", "KAST%"], ["kpr_adj", "KPR adj"], ["dpr_adj", "DPR adj"], ["multikill_rating", "multikill"], ["opening_kills", "OK"], ["opening_deaths", "OD"], ["clutches_won", "clutch"], ["trade_kills", "trade K"], ["swing_pct", "swing pp/r"], ["pos_move_pct", "pos move"], ["pos_death_pct", "pos death"], ["rating_approx", "rating~"]];
  const SWING_COLS = [["name", "player"], ["team", "team"], ["swing_pct", "swing"], ["added_pct", "added"], ["deducted_pct", "deducted"], ["kill_pct", "kills"], ["damage_pct", "damage"], ["flash_pct", "flash"], ["traded_pct", "traded"], ["death_pct", "deaths"], ["plant_pct", "plant"], ["defuse_pct", "defuse"], ["round_end_pct", "round end"], ["pos_move_pct", "pos move"], ["pos_death_pct", "pos death"], ["swing_pos_aware_pct", "pos-aware swing"]];

  function renderTable(table, rows, cols, opts = {}) {
    const state = table._sort || { key: opts.defaultSort || cols[0][0], dir: -1 };
    table._sort = state;
    const sorted = rows.slice().sort((a, b) => { const x = a[state.key], y = b[state.key]; if (x == null) return 1; if (y == null) return -1; return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * state.dir; });
    let html = "<thead><tr>" + cols.map(([k, label]) => `<th data-k="${k}" class="${k === state.key ? "sorted" : ""}">${label}${k === state.key ? (state.dir < 0 ? " &#9662;" : " &#9652;") : ""}</th>`).join("") + "</tr></thead><tbody>";
    for (const row of sorted) {
      html += `<tr class="${opts.onRow ? "clickable" : ""}" data-id="${row[opts.idKey || "round_num"] ?? ""}">` + cols.map(([k]) => {
        let v = row[k];
        if (typeof v === "number") v = Number.isInteger(v) && !k.endsWith("pct") ? v : (k.endsWith("pct") ? pp(v) : fmt2(v));
        const cls = k === "name" ? "name" : (typeof row[k] === "number" && (k.endsWith("pct")) ? (row[k] >= 0 ? "pos" : "neg") : "");
        return `<td class="${cls}">${esc(v ?? "-")}</td>`;
      }).join("") + "</tr>";
    }
    table.innerHTML = html + "</tbody>";
    $$("th", table).forEach((th) => { th.onclick = () => { if (state.key === th.dataset.k) state.dir *= -1; else { state.key = th.dataset.k; state.dir = -1; } renderTable(table, rows, cols, opts); }; });
    if (opts.onRow) $$("tbody tr", table).forEach((tr) => { tr.onclick = () => opts.onRow(tr.dataset.id); });
  }

  function renderStats() {
    const sum = S.summary;
    renderTable($("#box-table"), sum.box_score, BOX_COLS, { defaultSort: "rating_approx" });
    renderTable($("#swing-table"), sum.swing, SWING_COLS, { defaultSort: "swing_pct" });
    const roundRows = sum.rounds.map((r) => ({ ...r, plant: r.planted ? `${r.plant_site || "?"} @ ${clock(ROUND_SECONDS - r.plant_t)}` : "", score: Object.entries(r.score_after).map(([k, v]) => `${k} ${v}`).join(" - "), p0_t_pct: r.p0_t != null ? 100 * r.p0_t : null }));
    const RCOLS = [["round_num", "#"], ["winner_team", "winner"], ["reason", "reason"], ["t_team", "T"], ["t_buy", "T buy"], ["ct_team", "CT"], ["ct_buy", "CT buy"], ["plant", "plant"], ["kills", "kills"], ["p0_t_pct", "P(T) at start"], ["score", "score"]];
    const rt = $("#rounds-table"); rt._sort = { key: "round_num", dir: 1 };
    renderTable(rt, roundRows, RCOLS, { onRow: (id) => { switchView("replay"); openRound(+id); } });
    $("#key-moments").innerHTML = sum.key_moments.map((k) => `<div class="moment" data-r="${k.round_num}" data-t="${k.t}"><b class="${k.delta_pp >= 0 ? "pos" : "neg"}">${pp(k.delta_pp)}</b> <span class="muted">R${k.round_num} ${clock(ROUND_SECONDS - k.t)}</span><div class="who"><span class="${k.team.toLowerCase()}">${esc(k.name)}</span> <span class="muted">${esc(k.kind.replace("_", " "))}</span></div></div>`).join("");
    $$(".moment").forEach((el) => { el.onclick = () => { switchView("replay"); openRound(+el.dataset.r, Math.max(0, +el.dataset.t - 4)); }; });
  }

  // zones view -------------------------------------------------------------------
  async function loadZones() {
    if (!S.matchId) return;
    if (!S.zones) { try { S.zones = await api(`/api/matches/${S.matchId}/zones`); } catch (e) { toast(e.message, true); return; } }
    const rows = S.zones.zones.filter((z) => z.side === S.zoneSide && z.phase === S.zonePhase);
    const ZCOLS = [["zone", "zone"], ["rounds_weighted", "rounds"], ["win_rate_when_held", "win rate held"], ["lift_pp", "lift pp"], ["model_pp_at_50", "model pp @50"]];
    const zt = $("#zone-table"); zt._sort = zt._sort || { key: "model_pp_at_50", dir: -1 };
    renderTable(zt, rows.map((r) => ({ ...r, win_rate_when_held: r.win_rate_when_held != null ? 100 * r.win_rate_when_held : null })), ZCOLS);
    drawZoneRadar();
  }

  function drawZoneRadar() {
    const canvas = $("#zone-radar"); if (!canvas || !S.zones) return;
    const { ctx, size } = radarCtx(canvas);
    drawBase(ctx, size);
    const rows = S.zones.zones.filter((z) => z.side === S.zoneSide && z.phase === S.zonePhase);
    const maxAbs = Math.max(1, ...rows.map((r) => Math.abs(r.model_pp_at_50 || 0)));
    ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    for (const r of rows) {
      const xy = S.zones.centroids[r.zone]; if (!xy) continue;
      const [px, py] = worldToPx(xy[0], xy[1], size);
      const v = r.model_pp_at_50 || 0, rad = 6 + 22 * Math.abs(v) / maxAbs;
      ctx.beginPath(); ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fillStyle = v >= 0 ? "rgba(62,207,142,.55)" : "rgba(255,93,93,.55)"; ctx.fill();
      ctx.fillStyle = "#fff"; ctx.fillText(`${r.zone} ${v >= 0 ? "+" : ""}${v.toFixed(1)}`, px, py - rad - 3);
    }
  }

  // playback -----------------------------------------------------------------------
  let rafLast = null;
  function tick(ts) {
    if (!S.playing) { rafLast = null; return; }
    if (rafLast != null && S.round) {
      S.t += ((ts - rafLast) / 1000) * S.speed;
      if (S.t >= S.round.duration) { S.t = S.round.duration; S.playing = false; updatePlayBtn(); }
      drawAll();
    }
    rafLast = ts;
    requestAnimationFrame(tick);
  }
  function play(on) { S.playing = on == null ? !S.playing : on; updatePlayBtn(); if (S.playing) { if (S.round && S.t >= S.round.duration - 0.05) S.t = 0; rafLast = null; requestAnimationFrame(tick); } }
  function updatePlayBtn() { $("#btn-play").innerHTML = S.playing ? "&#10074;&#10074;" : "&#9654;"; }
  function seek(dt) { if (!S.round) return; S.t = Math.min(S.round.duration, Math.max(0, S.t + dt)); drawAll(); }
  function stepRound(d) { if (!S.summary) return; const rs = S.summary.rounds, i = rs.findIndex((r) => r.round_num === S.roundNum); const n = rs[i + d]; if (n) openRound(n.round_num); }

  function switchView(v) {
    S.view = v;
    $$(".views button").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
    for (const k of ["replay", "stats", "zones"]) $(`#view-${k}`).classList.toggle("hidden", k !== v);
    if (v === "zones") loadZones();
    if (v === "replay") requestAnimationFrame(drawAll);
  }

  // wiring -------------------------------------------------------------------------
  function wire() {
    $$(".views button").forEach((b) => (b.onclick = () => switchView(b.dataset.view)));
    $("#toggle-sidebar").onclick = () => $("#sidebar").classList.toggle("open");
    const dz = $("#dropzone"), fi = $("#file-input");
    dz.onclick = () => fi.click(); $("#browse").onclick = (e) => { e.stopPropagation(); fi.click(); };
    fi.onchange = () => { uploadFiles(Array.from(fi.files)); fi.value = ""; };
    ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); if (ev === "drop" || e.target === document) dz.classList.remove("drag"); }));
    document.addEventListener("drop", (e) => { if (e.dataTransfer?.files?.length) uploadFiles(Array.from(e.dataTransfer.files)); });

    $("#btn-play").onclick = () => play();
    $("#btn-back").onclick = () => seek(-5); $("#btn-fwd").onclick = () => seek(5);
    $("#btn-prev-round").onclick = () => stepRound(-1); $("#btn-next-round").onclick = () => stepRound(1);
    $("#speed").onchange = (e) => (S.speed = +e.target.value);
    $("#seek").oninput = (e) => { S.t = +e.target.value; drawAll(); };
    $("#timeline").onclick = (e) => { if (!S.round) return; const rect = e.target.getBoundingClientRect(); const x = e.clientX - rect.left; const padL = 34, padR = 8; S.t = Math.min(S.round.duration, Math.max(0, ((x - padL) / (rect.width - padL - padR)) * S.round.duration)); drawAll(); };
    ["chk-names", "chk-zones", "chk-trails"].forEach((id) => ($(`#${id}`).onchange = async () => { if (id === "chk-zones" && !S.zones && $("#chk-zones").checked) { try { S.zones = await api(`/api/matches/${S.matchId}/zones`); } catch (e) { /* ignore */ } } drawRadar(); }));

    $$(".tabs button").forEach((b) => (b.onclick = () => { $$(".tabs button").forEach((x) => x.classList.toggle("active", x === b)); $$(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`)); drawRadar(); }));
    const seg = (id, key, after) => $$(`#${id} button`).forEach((b) => (b.onclick = () => { $$(`#${id} button`).forEach((x) => x.classList.toggle("active", x === b)); S[key] = b.dataset.v; if (after) after(); }));
    seg("eng-side", "engineSide"); seg("eng-enemy", "engineEnemy"); seg("eng-movers", "engineMovers");
    seg("zone-side", "zoneSide", loadZones); seg("zone-phase", "zonePhase", loadZones);
    $("#eng-h").oninput = (e) => ($("#eng-h-val").textContent = e.target.value);
    $("#eng-run").onclick = runEngine;

    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.code === "Space") { e.preventDefault(); play(); }
      else if (e.key === "ArrowLeft") seek(e.shiftKey ? -5 : -1);
      else if (e.key === "ArrowRight") seek(e.shiftKey ? 5 : 1);
      else if (e.key === "[") stepRound(-1); else if (e.key === "]") stepRound(1);
      else if (e.key === "e") runEngine();
    });
    window.addEventListener("resize", () => { drawAll(); drawZoneRadar(); });

    // hover on radar -> highlight player
    $("#radar").addEventListener("mousemove", (e) => {
      if (!S.lastFrame) return;
      const rect = e.target.getBoundingClientRect(), size = rect.width, mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let best = null, bd = 14;
      for (const p of S.lastFrame.players) { if (!p.alive) continue; const [px, py] = worldToPx(p.x, p.y, size); const d = Math.hypot(px - mx, py - my); if (d < bd) { bd = d; best = p.sid; } }
      if (best !== S.hoverSid) { S.hoverSid = best; drawRadar(); }
    });
  }

  wire();
  refreshMatches();
})();
