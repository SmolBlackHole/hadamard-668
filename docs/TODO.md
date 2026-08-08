# TODO

Stand: 2026-08-08

## Status

**Durchbruch am 08.08.2026**: Targeted Escape durchbricht die Q=1-Wand via
Basin-Hopping. 1 neg-Flip pro Sequenz am dominanten Lag + Quench.
100% der tief getesteten Q=1-Zustände escapen (10/10 bei n=44).

Neuer Flaschenhals: Nur ~20% der Runs erreichen Q≤3 bei n≥48.

---

## A. Quench-Beschleunigung (höchste Priorität)

Aktuell: 600+ Kombos × 20k-Step-ILS = ~8s pro Escape.
Ziel: Gleiche Escape-Rate in <500ms.

- [ ] **A1 — 2-stufiger Quench (Turnier)**
  - Stufe 1: 600× Pure Greedy (200 Steps) → sortiere nach Greedy-End-Q
  - Stufe 2: Top-5 Kandidaten → Full Quench (20k)
  - Erwartung: 99% der Quench-Zeit gespart

- [ ] **A2 — Adaptives Quench-Budget pro Q-Level**
  - Q=1: 20k, Q=2: 15k, Q=3: 10k, Q=4-5: 8k
  - Weniger Combos für höhere Q, proportional zum erwarteten Aufwand

- [ ] **A3 — Quench-Early-Abort nach Q-Verbesserung**
  - Wenn Quench um ≥10 Q-Stufen in ersten 2k Steps fällt → weiter
  - Sonst abbrechen. 200-Step-Test war zu kurz, 2k könnte reichen

- [ ] **A4 — `||sum d_i||²` als Combo-Filter**
  - Je kleiner, desto weniger Lag-Kopplung nach Kick
  - 1 dot product (ns) vs 200 Greedy-Steps (µs)

- [ ] **A5 — Gray-Code Combo-Traversal**
  - 2^k Kombos mit O(1) Delta-Cache-Update pro Schritt
  - 4096 States in 40ms statt 81 Builds in 11ms

---

## B. Mehr Trajektorien zu Q≤3 (neuer Flaschenhals)

- [ ] **B1 — Q-Trajektorie-Tracking**
  - Logge Q nach jedem Greedy+Tabu+Kick
  - Identifiziere wo Runs plateauen (aktuell nur End-Q bekannt)

- [ ] **B2 — Geo-Gewicht adaptiv pro Q-Level**
  - Stärkeres Geo bei hohem Q (bremst zu schnellen Abstieg)
  - Schwächer bei Q≤5
  - Analog PZinn's Temperatur-Anpassung

- [ ] **B3 — Tabu FRÜHER triggern**
  - Nicht erst wenn Greedy stuck, sondern wenn Q-Drop-Rate unter
    Schwellwert fällt
  - Verhindert dass Greedy in Sackgassen läuft

- [ ] **B4 — Targeted Escape bei Q=2,3,4,5 (nicht nur ≤3)**
  - Bei n=48 erreichen Runs Q=4-5 regelmäßig
  - Random-Kick zerstört diese sofort (Q=4→47→5→53)
  - Test mit größerem Kandidatenpool (10 statt 5 pro Seq)

- [ ] **B5 — Basin-aware Seed-Filter**
  - FFT-Screening bringt nichts (marginal bei n=44)
  - Neue Metrik: `F(x) = Q(Greedy(x))` statt `Q(x)`
  - Oder: `||d||²`-Verteilung, Kanten-Run-Längen, NAF-Spärlichkeit

---

## C. Mathematisches Verständnis

- [ ] **C1 — sum_d-Struktur vergleichen**
  - 10 erfolgreiche vs 10 erfolglose Kicks
  - Per-Lag, Per-Sequenz, Kopplungs-Matrix
  - Ergebnis bisher: Kein statischer Unterschied — reine Basin-Dynamik

- [ ] **C2 — D^T D vor und nach Kick**
  - Eigenwertspektrum, Konditionszahl
  - Wird D^T D näher an 2nI?
  - Tight-Frame-Charakterisierung (docs/TIGHT_FRAME_CHARACTERIZATION.md §4)

- [ ] **C3 — Greedy-Pfad-Vergleich**
  - Welche Flips werden in den ersten 20 Greedy-Steps akzeptiert?
  - Erfolgreich: 25 verbessernde Flips nach Kick
  - Erfolglos: 19 verbessernde Flips

- [ ] **C4 — Theorem-Formulierung**
  - "Für Q=1 existiert Targeted-Kick dessen Quench Q=0 erreicht"
  - Empirisch: 10/10 bei n=44 (inkl. Tieffindung jenseits Top-5)
  - Gegenbeispiele bei n=48,52 suchen oder Beweis skizzieren

- [ ] **C5 — Kanten-Raum-Analyse**
  - `||d_i||² = #{t: P_t(c) = P_t(c-t)}`
  - Bietet Kantenstruktur bessere Combo-Auswahl als Q?
  - Targeted-Kick = 8 Kanten-Flips (4×2 benachbarte Kanten)

---

## D. Architektur / Solver-Design

- [ ] **D1 — Targeted-Kick auslagern und parameterisieren**
  - `max_flips_per_seq`, `quench_budget`, `max_combos` als Config
  - Aktuell hart kodiert in `_targeted_kick()`

- [ ] **D2 — Budget-Tracking pro Phase**
  - Wieviel Steps in Greedy / Tabu / Kick / TargetedEscape?
  - Erst dann sichtbar ob Targeted Escape sich lohnt

- [ ] **D3 — Cooldown nach Targeted Escape**
  - Mindestens N Greedy-Steps vor nächstem Targeted-Kick
  - Verhindert Oszillation zwischen Kicks

- [ ] **D4 — Greedy-Descent als eigene exportierte Funktion**
  - Mit optionalem Callback für Trajektorie-Analyse
  - Nutzbar als Pre-Filter für Quench

- [ ] **D5 — Experiment-Template**
  - Einheitliches Benchmark-Script: `(n, seeds, workers, config_dict)`
  - Statt ad-hoc -c Scripts

---

## E. PZinn-Adaption

- [ ] **E1 — Improve-Phases für negazyklischen Raum**
  - Fixiere 3 Sequenzen, löse 4. via NAF-Rekonstruktion
  - Andere Mathematik als FFT (zirkulant vs negazyklisch)
  - 20ms für 30 Phasenversuche, Ergebnis: Q=31 (nicht 0)

- [ ] **E2 — Parallel Tempering mit 4 Repliken**
  - Verschiedene `tabu_noise`-Levels
  - Swaps alle ~100 Steps mit Autotuning
  - 4× Kosten, bewiesen effektiv in PZinn

- [ ] **E3 — Smooth Score im Greedy**
  - `f(u) = u - log(1+u)` statt diskretem Q
  - Spearman 0.86 mit Q, aber Top-5 Überlappung nur 2/5
  - 10-100× teurer via FFT — lohnt nicht für Standard-Greedy

---

## F. Nächstes n-Level

- [ ] **F1 — n=48 mit optimiertem Quench**
  - 2-stufig, adaptives Budget, Targeted Escape bei Q≤5
  - Ziel: >10% Solve-Rate (aktuell ~0%)

- [ ] **F2 — n=52 Profil**
  - Wo scheitert der Solver? Start-Q, Abstiegsrate, Q-End-Verteilung
  - Aktuell völlig unbekannt

- [ ] **F3 — n=32-44 Sweep mit finaler Konfiguration**
  - Baseline für alle zukünftigen Tests
  - Mit und ohne Targeted Escape

---

## G. n=167 Roadmap

- [ ] **G1 — Skalierungsgesetz für Q=1 Escape-Rate vs n**
  - Ab wann bricht Targeted Escape zusammen?
  - Prognose aus n=36-52 Daten

- [ ] **G2 — Subexponentielle Beschleunigung**
  - Kann Targeted Escape + smarter Seed exponentiellen Abfall
    der Solve-Rate kompensieren?

---

## Erledigt (08.08.2026)

- [x] Q=1 mathematisch als striktes lokales Minimum bewiesen
- [x] Targeted Escape (1 neg-Flip/Seq + Quench) implementiert
- [x] 10/10 Q=1 Zustände escapable bei n=44 (tiefe Suche)
- [x] PZinn Smooth Score, Phase-Reconstruction, Gray-Code getestet
- [x] FFT-Prescreening getestet (marginal, kein Gewinn)
- [x] Early-Abort 200-Step Greedy getestet (filtert nichts)
- [x] Pure-Greedy Quench getestet (endet bei Q=3 statt 0)
- [x] `geo_weight` in SolverConfig integriert
- [x] Solver in Phasen-Funktionen aufgeteilt
- [x] Experimente aufgeräumt (10 tote Dateien gelöscht)

---

## Erledigt oder verworfen (vor 08.08.2026)

- [x] Pair-Rescue gepaart ablariert und vollständig entfernt.
- [x] K-Tuning gestrichen; es gehörte nur zum Pair-Pfad.
- [x] Statischen und Online-Gray-Code bis n=52 geprüft; monotone Varianten verworfen.
- [x] `combo_energy()` unabhängig gegen direkte Neuberechnung geprüft.
- [x] Alle einzelnen Intervallflips auf n=52-Low-Q-Zuständen geprüft.
- [x] Kantenbalancierten `r_1=0`-Start geprüft und verworfen.
- [x] Blockintegrabilität des Flip-Operators charakterisiert und verifiziert.
- [x] Trace-Infrastruktur und abgeschlossene Wegwerfprototypen entfernt.

Details und Zahlen stehen in [`research/2026-08-07-search-ablations.md`](research/2026-08-07-search-ablations.md).
