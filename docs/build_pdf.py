"""Build docs/poswing.pdf (concise documentation of the tool and the work behind it).

Run:  python docs/build_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = Path(__file__).with_name("poswing.pdf")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8, spaceBefore=12)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceAfter=5, spaceBefore=10)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.8, leading=13.2, alignment=TA_LEFT, spaceAfter=5)
BUL = ParagraphStyle("Bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.2, leading=10.2, leftIndent=6, backColor=colors.whitesmoke,
                      borderPadding=4, spaceAfter=6)
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=BODY, textColor=colors.grey, fontSize=10)


def p(text: str) -> Paragraph:
    return Paragraph(text, BODY)


def bullets(items: list[str]) -> list:
    return [Paragraph(f"• {it}", BUL) for it in items]


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE)


def table(rows: list[list[str]], widths: list[float]) -> Table:
    data = [[Paragraph(f"<b>{c}</b>" if i == 0 else c, BODY) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def build() -> None:
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="poswing - position-based round swing for CS2",
                            author="poswing")
    s: list = []
    s.append(Paragraph("poswing", TITLE))
    s.append(Paragraph("Position-based round swing, Rating 3.0 ingredients and an engine mode for CS2 demos", SUB))
    s.append(Spacer(1, 10))

    # 1 ---------------------------------------------------------------------
    s.append(Paragraph("1. Purpose", H1))
    s.append(p("HLTV's Round Swing measures how much each kill changes a team's probability of winning the round, "
               "given the map, side, players alive, bomb state and economy. <b>poswing</b> extends that idea with "
               "<i>where</i> the players are. The map is divided into sections, the win probability is modelled as a "
               "function of which sections each side occupies, and from that the tool derives: the value of each "
               "position, a team's recurring setups and their flaws, the best reactions to an enemy setup, "
               "a positional swing for every player, and an engine that answers: <i>given this demo and this moment, "
               "what should the team do?</i>"))
    s.append(p("Input is the standard HLTV demo download (a GOTV <font face='Courier'>.dem</font> recording in the "
               "Source 2 format). Parsing is done by the open-source library demoparser2; everything else is in this package."))

    # 2 ---------------------------------------------------------------------
    s.append(Paragraph("2. Quick start", H1))
    s.append(code("""
pip install -e .
poswing parse demos/*.dem -o cache/                       # once per demo, seconds each
poswing report cache/ -o report.md                        # box score, swing, zones, setups
poswing zones cache/ --map de_mirage --side CT
poswing setups cache/ --team "NAVI" --side T --at contact
poswing react cache/ --map de_mirage --side CT --enemy "Apartments x3, Underpass x2" --at 25
poswing engine cache/match.pkl --round 7 --clock 1:20 --team NAVI --data cache/ --book
"""))
    s.append(p("Every command accepts <font face='Courier'>.dem</font> files, cached <font face='Courier'>.pkl</font> "
               "files or directories. The model is fitted per map from whatever inputs are given, or loaded with "
               "<font face='Courier'>--model</font> after <font face='Courier'>poswing fit</font>."))

    # 3 ---------------------------------------------------------------------
    s.append(Paragraph("3. Pipeline", H1))
    s.append(table([
        ["Stage", "Module", "What it produces"],
        ["Parse", "parse.py", "Rounds (freeze end, end, winner, reason, plant, team buys, kit), kills with equipment "
                              "categories, damage, flashes, bomb events, and position samples once per second plus the "
                              "tick before and at every kill/plant. Team names follow rosters across half swaps."],
        ["Zone", "zones.py", "A section label per sample. Default: the nav-mesh place name the game stores on the player "
                             "(BombsiteA, Connector, Apartments...). Alternatives: a square grid, or custom polygons from JSON."],
        ["State", "state.py", "GameState per sample: alive per side, bomb, clock, buys, kit, defusing, and the multiset "
                              "of zones each side occupies. Rule-fixed terminal states (side eliminated, unstoppable bomb)."],
        ["Model", "winprob.py", "P(T wins | state) = baseline + positional layer (section 4)."],
        ["Swing", "swing.py", "HLTV-style round swing and the new positional swing (section 5)."],
        ["Rating", "rating.py", "Rating 3.0 ingredients and an approximate composite (section 6)."],
        ["Analysis", "analysis.py", "Zone values, team setups, reactions (section 7)."],
        ["Engine", "engine.py", "Moment -> best lines, movement graph, book of similar situations (section 8)."],
    ], [2.0 * cm, 2.4 * cm, 12.6 * cm]))

    # 4 ---------------------------------------------------------------------
    s.append(Paragraph("4. Win-probability model", H1))
    s.append(p("<b>Baseline.</b> The variables HLTV describes for Round Swing (map, players alive per side, bomb "
               "planted, economy bracket of both teams) plus the round clock. A hand-set parametric prior is corrected "
               "by an empirical-Bayes lookup table keyed on map, alive counts, bomb state and economy difference, so "
               "well-populated situations follow the data and rare ones fall back on the prior (pseudo-count 20)."))
    s.append(p("<b>Positional layer.</b> An L2-regularised logistic regression fitted on top of the baseline logit "
               "(used as an offset). Features, separately for pre- and post-plant: side present in a zone, extra players "
               "stacked in the same zone (so the second player is not worth the same as the first), how spread the side "
               "is, and the clock. Rounds are weighted equally regardless of length. The difference "
               "<font face='Courier'>P_pos - P_base</font> is the value of the positions on top of the man-count situation."))
    s.append(p("<b>Rules override the model.</b> A side with nobody alive has lost (or has a 3% chance if the bomb is "
               "down); when the bomb cannot be stopped (under 10 s, under 5 s with a kit, nobody defusing) the Ts have "
               "won, which also implements HLTV's October 2026 rule that kills in that window carry no swing."))

    # 5 ---------------------------------------------------------------------
    s.append(Paragraph("5. Round swing and positional swing", H1))
    s.append(p("The HLTV-style part uses the baseline model so that it stays comparable with HLTV's numbers. "
               "For each kill the team's probability before and after is computed; the gain is split between the "
               "finisher (50%), the damage dealers on the victim in proportion to damage (30%), the flash assister (10%) "
               "and, when the kill is a trade, the traded teammate (10%); unused shares go to the finisher. The victim is "
               "deducted the same amount. A kill never yields negative swing. Bomb plants are shared by the alive Ts and "
               "deducted from the alive CTs. The round-end residual goes to the winners in proportion to their positive "
               "contributions (post-plant contributions only when the bomb was down, with half of a defuse to the "
               "defuser) and is deducted equally from the losers who survived. Every round is exactly zero-sum and sums "
               "to <i>outcome - P0</i>; this is checked by a unit test."))
    s.append(p("<b>Positional swing</b> is new and reported separately so the HLTV-comparable figure stays clean:"))
    s.extend(bullets([
        "<b>pos_move</b>: between two consecutive samples with no event, the change of the team's positional edge "
        "(<font face='Courier'>P_pos - P_base</font>) is credited to the players who changed zone, split equally.",
        "<b>pos_death</b>: when a player dies, minus the absolute value of their zone's contribution at that moment. "
        "Dying in a good position wastes it; dying in a bad position means the player chose that position. Both cost, "
        "as requested; the sign of the contribution does not rescue anyone.",
        "<b>swing_pos_aware</b> = swing + pos_move + pos_death.",
    ]))

    # 6 ---------------------------------------------------------------------
    s.append(Paragraph("6. Rating 3.0 ingredients", H1))
    s.append(p("HLTV publishes the ingredients of Rating 3.0 (Kills, Damage, Survival, KAST, Multi-kills, Round Swing, "
               "all economy-adjusted) but not the formula, and there is no Rating 3.1: the October 2026 release was an "
               "accuracy update of 3.0. The box score contains every ingredient: kills, deaths, assists (40 damage), "
               "flash assists, headshot %, KPR, DPR, ADR, KAST, eco-adjusted KPR/DPR, opening kills/deaths, "
               "multi-kill rounds and rating, clutches, trade kills, traded deaths, failed trades, trade denials, "
               "utility damage, survival, and round swing with its added/deducted halves."))
    s.append(p("Economy adjustment follows the published description: each player's equipment is classed as sniper, "
               "tier-one rifle, tier-two rifle, SMG/shotgun, upgraded pistol or starter pistol by armour plus most "
               "expensive weapon; duel win-rates per category pair start from priors in the spirit of HLTV's table and "
               "are refined by the data; kill value = 2.1 - 2.07 x P(win duel), which reproduces the published anchors "
               "(rifle v rifle 48% = 1.10 points, rifle v starter pistol 75% = 0.54). The composite "
               "<font face='Courier'>rating_approx</font> plugs the adjusted ingredients, multi-kill rating and round "
               "swing into the public Rating 2.0 regression; it is an ordering, not HLTV's number."))

    # 7 ---------------------------------------------------------------------
    s.append(Paragraph("7. Position analytics", H1))
    s.extend(bullets([
        "<b>Zone values</b> (<font face='Courier'>zones</font>): per side and phase, the model's effect of holding a zone "
        "(percentage points at a 50/50 state) next to the raw win rate when the zone was held and its lift over the side's "
        "average, with the weighted number of rounds behind it.",
        "<b>Team setups</b> (<font face='Courier'>setups</font>): the multiset of zones a team occupies at a reference "
        "moment (N seconds into the round, or one second before first contact), grouped, with frequency, actual win "
        "rate, model probability, baseline probability, edge and execution gap. Common setups with a negative edge are "
        "structural flaws; setups that lose more than the model expects are execution problems.",
        "<b>Reactions</b> (<font face='Courier'>react</font>): given an enemy setup, what own setups were played against "
        "similar enemy setups and how they went, plus every observed own setup ranked by the model against exactly that "
        "enemy setup.",
    ]))

    # 8 ---------------------------------------------------------------------
    s.append(Paragraph("8. Engine mode", H1))
    s.append(p("<font face='Courier'>poswing engine DEMO --round R (--time S | --clock M:SS | --tick T) --team NAME</font> "
               "reconstructs the situation (who is where, alive counts, bomb, clock, buys) and searches for the best "
               "repositioning of the chosen team within a horizon (default 10 s):"))
    s.extend(bullets([
        "<b>Legal moves</b>: zones reachable within the horizon according to how players actually move between zones in "
        "the data (shortest path over median transition times).",
        "<b>Search</b>: beam search over the team's assignment, one player at a time, scored by the model with the enemy "
        "either static or predicted (Monte-Carlo over the empirical distribution of where a player in that zone is "
        "N seconds later). A small cost per mover prefers simple lines.",
        "<b>Output</b>: the top lines with P(win) and the gain over holding, the current and baseline probabilities, and "
        "optionally a book of similar historical situations (same phase and alive counts, similar enemy setup) showing "
        "what the side did next and whether it won.",
    ]))
    s.append(code("""
de_vertigo round 5 @ 25.0s (tick 42598) - 90s left, 5v5 (T v CT), T buy full-buy, CT buy full-buy
  T  (MOUZ NXT): PR@Tunnels, sirah@Bridge, Burmylov@Bridge, Neityu@Mid, Chr1zN@Mid
  CT (Space): TruNiQ@B Site, Vert@B Site, danistzz@Elevator, fozil@A Site, X5G7V@Scaffolding

Engine for T (MOUZ NXT), horizon 10s, enemy static, model 23 rounds of de_vertigo
Current P(win) for T: 36.8%  (baseline without positions: 48.1%)
1. P(win)  40.5%  (+3.6 pp)  PR: Tunnels -> Pit
2. P(win)  41.0%  (+4.2 pp)  PR: Tunnels -> Pit; Neityu: Mid -> Elevator; Chr1zN: Mid -> Elevator
"""))
    s.append(p("The example above was produced from a single 23-round demo and is shown for the format only; the "
               "numbers are not meaningful at that sample size."))

    # 9 ---------------------------------------------------------------------
    s.append(Paragraph("9. Training on the ESTA dataset", H1))
    s.append(p("HLTV blocks scripted demo downloads, so <font face='Courier'>poswing esta</font> imports the public ESTA "
               "dataset (Xenopoulos and Silva, 2022): 1,558 professional CS:GO demos parsed to JSON with 2 Hz player "
               "snapshots, kills, damage, flashes and bomb events. ESTA frames carry coordinates but no place names, so each "
               "position is looked up in the CS:GO nav mesh bundled from awpy 1.3.1 and the few callouts that changed in "
               "CS2 are renamed (Tunnel and TunnelStairs to Underpass, TicketBooth to CTSpawn, Balcony to Apartments); "
               "the mapping was derived by overlaying CS2 samples on the CS:GO mesh, whose Mirage coordinates coincide. "
               "A model trained on ESTA therefore evaluates a CS2 demo directly."))
    s.append(p("On Mirage: 278 matches, 7,393 rounds, 654,954 states, 194 positional features, fitted in about four "
               "minutes on four cores (the logistic layer accumulates its Hessian in chunks, so memory stays flat)."))
    s.append(p("Held-out check on 40 ESTA matches (1,008 rounds, 83k open states) not used for training, weighted log-loss "
               "of P(T wins):"))
    s.append(table([
        ["Model", "Baseline only", "Baseline + positions"],
        ["23-round model (one CS2 demo)", "0.471", "0.517 (positions hurt: overfit)"],
        ["ESTA model (7,393 rounds)", "0.447", "0.442 (positions help)"],
    ], [7.0 * cm, 4.0 * cm, 6.0 * cm]))
    s.append(p("The single-demo positional layer memorises its 23 rounds (in-sample log-loss 0.337) and generalises worse "
               "than no positions at all; the ESTA layer generalises, but its gain over the man-count/bomb/economy baseline "
               "is modest (about 1% of log-loss), which is the honest measure of how much coarse zone occupancy adds."))
    s.append(p("Same round, both models (NAVI Javelins on CT, Mirage round 7, clock 1:25, 4v2 after losing three duels, "
               "three Ts on the A site): both put the CTs at about 4% and both find no move worth more than one "
               "percentage point, but the small model's favourite (Middle to Underpass) rested on three weighted rounds in "
               "which a CT in Underpass happened to win, while the ESTA model prefers holding or pushing Middle and Top of "
               "Mid. Its baseline for the 5v5 semi-eco versus full-buy start is 21% instead of 34%, in line with published "
               "economy tables."))

    # 10 --------------------------------------------------------------------
    s.append(Paragraph("10. Validation", H1))
    s.extend(bullets([
        "Parsed three HLTV demos (Mirage, Ancient, Vertigo; 22-23 rounds each) end to end in about two seconds per demo; "
        "rounds, winners, reasons, plants, team names across halves and economies matched the scoreboard.",
        "Unit tests (16) cover zoning, parsing helpers, terminal rules, state building, model monotonicity and "
        "persistence, the zero-sum identity of the swing, non-negative kill swing, box-score consistency, analytics, "
        "the engine and the CLI. A synthetic match generator with known strong zones checks that the model recovers them.",
        "On the real Mirage demo the swing breakdown is in the expected range for a single map (about -6 to +7 "
        "percentage points per round) and every round sums to outcome - P0 to machine precision.",
    ]))

    # 10 --------------------------------------------------------------------
    s.append(Paragraph("10. Limitations and next steps", H1))
    s.extend(bullets([
        "Data volume is the limiting factor. One map with 23 rounds yields about 90 noisy positional features; the tool "
        "warns below 60 rounds. A season of demos per map is needed before the engine's advice is trustworthy.",
        "Zones are coarse: utility, angles, timing and information inside a zone are not modelled. The model is a "
        "positional prior, not a tactics oracle.",
        "HLTV's exact credit shares, duel tables and rating weights are not public; the implementation follows their "
        "published descriptions and anchors and exposes the shares as parameters.",
        "Natural extensions: nav-mesh adjacency for the movement graph instead of observed transitions, utility state in "
        "the game state, a two-ply search with an opponent policy, and a web UI over the JSON outputs.",
    ]))

    # 11 --------------------------------------------------------------------
    s.append(Paragraph("11. Command reference", H1))
    s.append(table([
        ["Command", "Key options"],
        ["parse DEMOS -o DIR", "--hz sampling rate"],
        ["fit INPUTS -o model.json", "--map, --l2"],
        ["report INPUTS", "-o report.md, --json out.json, --at 20|contact, --min-n, --model"],
        ["zones INPUTS", "--map, --side, --min-n"],
        ["setups INPUTS --team NAME", "--side, --map, --at, --min-n"],
        ["react INPUTS --side S --enemy SPEC", "--map, --at, --top; SPEC like 'A Site x2, Connector'"],
        ["engine DEMO --round R", "--time | --clock | --tick, --team | --side, --data, --horizon, --top, --beam, "
                                  "--enemy static|predict, --max-movers, --move-cost, --book, --json"],
        ["all commands", "--zoning place|grid|custom, --cell, --zones-file, --cache DIR, -v"],
    ], [6.0 * cm, 11.0 * cm]))

    doc.build(s)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
