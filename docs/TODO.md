# TODO

Stand: 2026-08-08 (Session 2)

## Status

**Targeted Escape integriert**: Q≤3 Trigger + exhaustive 625-Combo-Quench + early_escape_done-Flag.
**Durchbruch Session 1**: Q=1-Wand durchbrochen via Basin-Hopping (10/10 escapable bei n=44).
**Erkenntnis Session 2**: Kein Pre-Filter ersetzt den Quench. Kein Lean-Kick ohne Quench hilft.
**Neuer Flaschenhals**: Nur ~20% der Runs erreichen Q≤3 bei n≥48.

---

## A. Quench-Beschleunigung

Aktuell: 625 Combos × 10k-Step-ILS im Quench. Akzeptabler Tradeoff (3.5% bei n=50 vs 0.08% ohne).

- [x] **A1 — 2-stufiger Quench (Turnier)** — Getestet, tot. Top-5 Greedy-End-Q fallen auf Q=1 zurück. Escape auf Rang 483/625. Greedy-Q korreliert nicht mit Escape-Erfolg.
- [ ] **A2 — Adaptives Quench-Budget pro Q-Level** — 10k fix getestet, funktioniert. Per-Q-Tuning nicht probiert.
- [x] **A3 — Quench-Early-Abort** — Getestet, tot. 200-Step und 2-Step Greedy-Drop sind kein Prädiktor. Alle Combos zeigen ähnliche Drops.
- [x] **A4 — `||sum d_i||²` als Combo-Filter** — Getestet, tot. Escape auf Rang 483. Keine Korrelation.
- [x] **A5 — Gray-Code Combo-Traversal** — Getestet, schnell aber findet Q=0 nicht ohne Quench. Kein Ersatz für exhaustive Suche.

---

## B. Mehr Trajektorien zu Q≤3 (Flaschenhals)

- [x] **B1 — Q-Trajektorie-Tracking** — Decision-Matrix erstellt. Q≥6: Tabu 100%. Q=4-5: Tabu 20-74%. Q≤3: Nur Escape.
- [ ] **B2 — Geo-Gewicht adaptiv pro Q-Level** — Ungetestet. Vielversprechend: starkes Geo bei hohem Q bremst Abstieg in schlechte Basins.
- [ ] **B3 — Tabu FRÜHER triggern** — Wenn Q-Drop-Rate abfällt, Tabu statt weiterem Greedy. Ungetestet.
- [x] **B4 — Targeted Escape bei Q=2-5 (Lean, ohne Quench)** — Getestet, tot. 524 Versuche, 0 erreichen Q≤3. Ohne Quench wirkungslos.
- [x] **B5 — Basin-aware Seed-Filter** — FFT-Screening getestet (Screen-1/5/10/20). Ohne Escape: Screen-1 SCHLECHTER als Random. Mit Escape: kein Gewinn.

---

## C. Mathematisches Verständnis

- [x] **C1 — sum_d-Struktur** — 10 gute vs 10 schlechte Kicks verglichen. Kein statischer Unterschied. Reine Basin-Dynamik.
- [ ] **C2 — D^T D vor und nach Kick** — Eigenwertspektrum. Ungetestet.
- [x] **C3 — Greedy-Pfad-Vergleich** — Erfolgreich: 25 verbessernde Flips nach Kick. Erfolglos: 19. Kleiner Unterschied, kein zuverlässiger Filter.
- [ ] **C4 — Theorem-Formulierung** — "Q=1 immer escapable". Empirisch 10/10. Gegenbeispiele bei n≥52 suchen.
- [x] **C5 — Kanten-Raum-Analyse** — `||d_i||² = #{t: P_t(c)=P_t(c-t)}`. 95% der Kanten bleiben nach Kick unverändert.

---

## D. Architektur / Solver-Design

- [ ] **D1 — Targeted-Kick auslagern und parameterisieren** — `max_flips_per_seq`, `quench_budget` als Config.
- [ ] **D2 — Budget-Tracking pro Phase** — Greedy/Tabu/Kick/TargetedEscape.
- [x] **D3 — Cooldown nach Targeted Escape** — `early_escape_done`-Flag verhindert Wiederholung. Done.
- [x] **D4 — Greedy-Descent als exportierte Funktion** — `_greedy_descent()` in Phasen aufgeteilt. Done.

---

## E. PZinn-Adaption

- [x] **E1 — Improve-Phases für negazyklischen Raum** — Getestet. FFT/Rundung erreicht Q=31, nicht 0. Andere Mathematik (zirkulant vs negazyklisch).
- [ ] **E2 — Parallel Tempering mit 4 Repliken** — Ungetestet. 4× Kosten, PZinn bewiesen effektiv.
- [x] **E3 — Smooth Score im Greedy** — Getestet. Spearman 0.86 mit Q, aber Top-5 nur 2/5. 10-100× teurer via FFT.

---

## F. Benchmarks & Messungen

- [x] **F1 — n=48 mit optimiertem Quench** — 10% Solve-Rate mit 50k-200k Steps. Steps-Budget irrelevant, Escape triggert früh.
- [x] **F2 — n=50 Durchsatz** — 35/1000 mit Escape (3.5%) vs 4/5000 ohne (0.08%). 44× mehr Lösungen.
- [x] **F3 — n=52** — 0/40. Q≤3 wird zu selten erreicht, Escape triggert nie.
- [x] **F4 — Steps-Sweep n=48** — 50k=100k=150k=200k → alle ~10%. 50k optimal.
- [x] **F5 — Tabu-Steps-Sweep n=48** — 200/400/600/800 → alle ~10%. Tabu-Steps irrelevant.
- [x] **F6 — Segment-Summen Start→Lösung** — delta median 30, nur 1% mit delta<10. Kein Filter.
- [x] **F7 — Greedy-Descent-Entscheidungsmatrix** — Greedy-Q-Drops: Stuck~816. Tabu: Q≥6 100%, Q=4-5 20-74%, Q≤3 0-3%.

---

## G. n=167 Roadmap

- [ ] **G1 — Skalierungsgesetz für Escape-Rate vs n** — Prognose aus n=36-52 Daten.
- [ ] **G2 — Subexponentielle Beschleunigung** — Kann Escape + smarter Seed exponentiellen Abfall kompensieren?

---

## Verworfen (Session 2, 08.08.2026)

- [x] Soft Greedy (ΔQ≥0 erlaubt) — zerstört Trajektorie, 0/20 vs 14/20.
- [x] Neutrale Flips vor Tabu — reiner Random Walk, 0 Löser.
- [x] Targeted-Kick bei Q=4-5 (während Greedy, ohne Quench) — 0/524 erreichen Q≤3.
- [x] Lean-Kick bei Q≤10 + End-Quench — 63% vs 68% Current, schlechter.
- [x] Nested Parallel (6 Seeds × 2 Quench-Threads) — marginal besser, Overhead zu hoch.

## Erledigt (Session 1, 08.08.2026)

- [x] Q=1 mathematisch als striktes lokales Minimum bewiesen
- [x] Targeted Escape (1 neg-Flip/Seq + Quench) implementiert
- [x] 10/10 Q=1 Zustände escapable bei n=44 (tiefe Suche)
- [x] FFT-Prescreening getestet (marginal, Screen-1 schlechter als Random)
- [x] Pure-Greedy Quench getestet (endet bei Q=3 statt 0)
- [x] `geo_weight` in SolverConfig integriert
- [x] Solver in Phasen-Funktionen aufgeteilt

## Erledigt oder verworfen (vor 08.08.2026)

- [x] Pair-Rescue gepaart ablariert und vollständig entfernt.
- [x] Statischen und Online-Gray-Code bis n=52 geprüft; monotone Varianten verworfen.
- [x] `combo_energy()` unabhängig gegen direkte Neuberechnung geprüft.
- [x] Alle einzelnen Intervallflips auf n=52-Low-Q-Zuständen geprüft.
- [x] Blockintegrabilität des Flip-Operators charakterisiert und verifiziert.
