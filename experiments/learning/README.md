# Learning-Experimente: Muster für schnellere Q-Level-Senkung und Lösungen

Forschungsauftrag: Lassen sich Muster/Modelle erlernen, mit denen wir schneller
auf niedrigere Q-Levels kommen oder Lösungen (Q=0) finden? Analyse-Skripte nur,
der Solver in `src/` bleibt unangetastet.

## Daten & Werkzeuge

- `data/benchmark-n52.json`: 500 Seeds bei n=52, 6 gelöst (Seeds 60, 297, 355,
  371, 437, 457). Generiert mit der Solver-Version von Git-Commit `dcfacc5`
  (v1: exhaustiver 625-Combo-Escape, KEIN kbit). Verifiziert: Replay stimmt
  exakt (Energie, Kicks, Stats zu 10/10).
- `data/benchmark-kbit-on/off.json`: 120 Runs bei n=44/48/50, generiert mit
  v1-Escape + kbit im Descent (21:55-21:57).
- `data/benchmark-parallel-quench.json` (22:19): n=44: 32/50, n=48: 5/50,
  n=50: 2/50 gelöst — derzeit bester Ansatz (aktive Arbeit des Users).
- `data/solutions.json`: gelöste Zustände für n=44/48/50/52 (alle in
  verschiedenen Äquivalenzklassen).
- `solver_snapshot/`: eingefrorene Kopie der dcfacc5-Solver-Version
  (solver.py, tracker.py, benchmark_stats.py). Nötig, weil die Live-Datei
  `src/solver.py` während der Analyse aktiv editiert wird (zwischenzeitlich
  unparsbar). Alle Replays laufen gegen den Snapshot.
- `common.py`: Laden, Base64-Decoding, NAF/u-Berechnung, Sequenz-Features.
- `instr_solver.py`: eigener instrumentierter Solver (v1-Pipeline, exakt
  repliziert + Event-Log mit Escape-Details).
- `v2_solver.py`: eigene Transkription des kbit/top-5-Escapes (v2, Zustand
  22:12) — für Experimente; gegen die kbit-Benchmarks nicht validierbar,
  weil diese mit v1-Escape+kbit generiert wurden.
- `verify_replay.py`: Replay-Validierung gegen benchmark-n52 (10/10 Match).
- `q1_escapes.py`, `q1b_all_seeds.py`, `q1c_q1states.py`: Escape-Analysen.
- `q2_initial_q.py`, `q2b_final_q_levels.py`: Initial-Q-Analysen.
- `q3_stats_analysis.py`: Stat-Korrelationen.
- `q4_solved_structure.py`: Struktur gelöster Zustände + Datenübersicht.

## Zentrale Befunde

### Q1: Escape-Mechanik und Flip-Muster

1. **Der v1-Escape ist im Kern eine Basin-Hopping-Suche, kein exakter
   4-Flip-Killer.** Der Screening-Code `t2.accept(cand, ...)` erzeugt durch
   Doppel-Flips (Direct-Flip + Accept-Flip) einen korrupten Delta-Cache; das
   `q_after_flips=0` der Sieger-Kombos war ein Artefakt. Die echte Q der
   Sieger-Kombos: 44-83 (NIE 0). Die 4 Flips lenken nur in ein anderes Becken,
   der Quench (10.000-Schritte-ILS, bis zu 625-mal) findet die Lösung.
2. **Die Sieger-Kombos haben Q_grid-Ränge 103-606/625** — sie sind NICHT unter
   den besten 5. Ein Escape, der nur die top-5 Kombos nach Q_grid quencht
   (v2-Design), würde alle 6 gefundenen Lösungen verpassen (kontrolliertes
   Experiment läuft).
3. **Jeder der 500 Runs erreicht einen {0,±1}-Residualzustand (u_t ∈ {0,±1},
   Q = Anzahl Nicht-Null-Lags, Q ∈ {3,4,5,6})** und bleibt dort hängen
   (Rekordtief). Der Escape feuert bei 435/500 Runs; 65 Runs (13%) verbrennen
   das Budget im Descent-Grinding, bevor sie 3-mal am Rekordtief hängen.
4. **Escape-Q-Verteilung**: Q=3: 19 Runs (0 gelöst), Q=4: 156 (3), Q=5: 191
   (2), Q=6: 69 (1). Solve-Raten 0%, 1.9%, 1.0%, 1.4%.
5. **Q≤2 ist die Mauer.** Bei den 6 gelösten Runs erreichte genau EIN
   innerer Quench-Search Q=0 (min_Q bei Iterationsgrenzen: 2→0-Sprung oder
   Tabu-Walk 8→0). Bei keinem der 494 ungelösten Runs erreichte irgendeiner
   der bis zu 625 Quenches je Q≤1 (bestes Niveau: Q=2). Q=1-Zustände wurden
   an Iterationsgrenzen nie beobachtet — die finale Transition läuft intern
   im Tabu-Walk (Uphill-Moves nötig, Greedy bleibt an Q=2-Dead-Ends ohne
   Einzel-Flip-Kanzelation hängen).
6. **Die Sieger-Kombos sind gleichverteilt über die 625** (Ränge 85-585) —
   kein Flip-Positionsmuster erkennbar; alle Kombos haben ähnliche
   Erfolgswahrscheinlichkeit (~1/500 pro Combo in guten Becken, 0 in
   schlechten).

### Q1e/Q1f: Feintrajektorie der Sieger-Quenches

- Die Sieger-Quenches starten bei Q=19-83 (4-Flip-Kombos) und erreichen Q=0 in
  nur 25-137 Outer-Iterationen. Das Muster (Seed 60, gewinnender Combo):
  Descents Q 10→9→6, random-kick 6→40, Descents 40→32→...→8, **Tabu-Walk
  8→0**. Der finalen Durchbruch Q≤8→0 macht der Tabu-Walk (200 interne
  Schritte), kein einzelner Flip.
- Der random-kick innerhalb des Quenchs (Basin-Hopping) ist essenziell: Er
  verschlechtert Q absichtlich (6→40), die frischen Descents erreichen dann
  ein neues Niveau (8), aus dem der Tabu durchbricht.
- Q=2-Zustände (in gelösten wie ungelösten Runs): u = {0,±1} mit GENAU 2
  Nicht-Null-Lags, Positionen scheinen zufällig verteilt (kein Clustering).
- Die Bestätigung der User-Hypothese: "Greedy+Tabu+Kick kommen in einen
  Bereich, wo der Quench einen Startzustand erzeugt, von dem der Greedy
  wieder sinnvoll iteriert" — exakt das beobachten wir; der finale Schritt
  ist der Tabu-Walk von Q≤8 auf Q=0.

### Q2: Initialzustand-Qualität

- Greedy-only-Descent-Q (ms-schnell berechenbar) korreliert NICHT mit dem
  Erfolg (corr(q0, solved) = +0.11; alle 6 gelösten Seeds haben q0 im
  mittleren Bereich 12-17; das unterste Quartil hat Solve-Rate 0.0).
- Seed-Vorfilterung über initiale Descent-Q ist nicht machbar.
- Finales Q (nach vollem Run) korreliert mit solved (-0.35) — tautologisch.

### Q3: Trajektorie-Stats

- **e_kicks/kick ist der stärkste Diskriminator**: gelöste Runs haben viel
  kleinere Energieänderung pro Kick (-4.400 vs -24.000 bei n=52; -3.200 vs
  -21.200 bei n=48). Gelöste Runs verbringen die meiste Zeit in niedrigen
  Q-Bereichen (Escapes), ungelöste hüpfen mit random Kicks.
- kick_evals/kicks/time negativ korreliert mit solved (teils tautologisch:
  Run stoppt bei Lösung).
- tabu_hits_q1=0 in fast allen Runs — Q=1 wird außerhalb des Outer-Loops
  (nur in Quenchen) erreicht. Das Outer-Loop-Trigger-System sieht Q=1 fast nie.

### Q4: Struktur gelöster Zustände

- 6/6 Lösungen bei n=52, 44+ bei n=44, 8 bei n=48, 1 bei n=50 — alle in
  verschiedenen Äquivalenzklassen.
- Keine auffällige gemeinsame Struktur (bal 0-9.5, runs ~21-26, sym/nsym ~1).
  Lösungen sind "typische" gut-balanceierte Folgen.

### Q7/Q9: Bitmuster- und Spiegelpunkt-Analyse (mathematischer Blickwinkel)

- **Frequenzbereich**: Die 4 Sequenzen jeder Lösung haben negativ korrelierte
  Skew-DFT-Magnituden (-0.33 bis -0.35 vs ~0 zufällig) — die
  Komplementärstruktur der Tight-Frame-Bedingung (Σ|Â|² = 4n konstant,
  verifiziert auf 1e-12). ALLE niedrig-Q-Zustände (auch Escape-Zustände der
  ungelösten Runs, corr ≈ -0.322) haben diese Struktur — sie trennt also
  nicht, ist aber die Signatur der Konstruktion.
- **Spiegelpunkt-Lemma** (siehe `mirror_point_lemma.md`): Q=1→0 durch einen
  Flip ⟺ "Spiegelpunkt mit Exception k" (x_{c-t} = -x_{c+t} für t≠k,
  x_{c-k} = x_{c+k} = x_c). Extrem seltenes Motiv (~2^{-(n+2)/2} zufällig);
  empirisch haben alle 10 untersuchten Q=1-Sackgassen-Zustände (n=44) 0
  Spiegelpunkte. Bei n=52 hat kein Escape-Zustand (0/500) eine
  Einzelflip-Kanzelation von -u.
- **n=44-Landschaft**: Alle Runs erreichen Q=1, aber nur ein Teil schließt
  zu 0 — die Q=1→0-Stufe ist der Flaschenhals (anders als bei n=52, wo
  ungelöste Runs nie Q≤1 erreichten).
- Datensammlung für das Spiegelpunkt-Signal läuft (collect_n44.jsonl:
  Escape-u + Spiegel-Vektor + Q=1-Lag pro Run).

### Q5: PZinn-Transformer-Bewertung

- `reference_pzinn/transformer.py`: kleines GPT (4 Layer, 128 Embedding,
  causal attention, myActiv = SiLU-ähnlich), tokenisiert Matrix-Zeilen mit
  `stacking` Bits pro Token (vocab 2^stacking), block_size = Zeilenlänge ×
  (n/stacking). Optional Score-Embedding (`transformer_uses_score`).
- PZinn lernt die Verteilung EINER bekannten Hadamard-Matrix aus Beispielen
  (mit Symmetrie-Randomisierung als Data Augmentation). Für unsere 4
  negazyklischen Sequenzen bräuchten wir analog: (a) viele gelöste
  Zustände als Trainingsbeispiele, (b) Tokenisierung der 4×n-Sequenzen.
- Datenlage: 6 gelöste n=52-Zustände (alle verschiedene Klassen) reichen
  NICHT. Bei n=44 ist die Lösungsrate ~55-64% (kbit-on/parallel-quench) —
  mit tausenden Runs sammelbar. Danach: Transformer als Seed-Generator für
  den ILS (statt rng.choice) oder als "repair"-Modell (niedriges Q in der
  Nähe einer Lösung vervollständigen).
- Empfehlung: Q5 erst nach Datensammlung; die Lernaufgabe "welche
  Escape-Zustände haben Q=0 in ihrem Becken" ist mit den n=52-Daten (6
  Positiv) nicht trainierbar.

### Q6: ML-Bewertung

- Siehe Abschnitt "ML-Evaluation (Q6)" unten.

## Kontrolliertes Escape-Design-Experiment (eigene Benchmarks, 500 Seeds n=52)

| Variante | Escape-Design | Gelöst |
| --- | --- | --- |
| A (Baseline) | v1-Trigger + exhaustive 625-Combo-Quench | **6/500** |
| B | v1-Trigger + top-5 nach Q_grid + kbit-Quench | **0/500** |
| C | v2-Trigger (q≤5, Cooldown, max 5 Escapes) + top-5 + kbit | **0/500** |
| E (Teilstichprobe) | v1-Trigger + top-5 nach Q_grid + VOLLER 10000-Schritte-Quench | **0/6** auf den Seeds, die A löst |

**Ergebnis: Die top-5-Selektion nach Q_grid ist der Killer, nicht der
Quench-Typ.** Die Sieger-Kombos liegen bei Q_grid-Rängen 103-606 (wahre
Post-Flip-Q 44-83); die top-5-Kombos (Q 15-28) liegen in Sackgassen-Becken,
die auch ein voller Quench nicht verlässt (inner_best_q 2-5). Niedrigeres
Start-Q heißt auf dieser Landschaft NICHT näher an der Lösung. Auch 5
Escape-Gelegenheiten (C) helfen nicht: Die Selektion bestimmt alles.

## Empfehlungen für den Solver (basierend auf den Befunden)

1. **Nicht nur top-5 Kombos nach Q_grid quenchen.** Die Sieger-Kombos
   liegen bei Q_grid-Rängen 103-606 (Q=44-83). Exhaustive Quenchs (~12s)
   finden die Lösungen; jede Selektion nach Start-Q verliert sie. Bessere
   Selektionen: (a) mehrere Score-Funktionen (u-Nähe, Segment-Struktur),
   (b) stichprobenartige Quenchs über den Combo-Raum mit kleinem Budget,
   dann Re-Quench der besten Stichprobe, (c) Diversität statt Gier.
2. **Niedrig-Q-Plateaus gezielt bearbeiten.** Die Q≤8→0-Transition macht
   der Tabu-Walk; kein ungelöster Run erreicht Q≤1 aus irgendeinem Combo.
   Escape-Designs, die das Tabu-Walk-Plateau bei Q≤8 verbreitern (mehr
   Tabu-Schritte bei niedrigem Q, mehrmalige Tabu-Walks vom selben
   Q≤2-Zustand), sind der vielversprechendste Hebel.
3. **Der {0,±1}-Zustand ist der natürliche Startpunkt für Muster-Lernen:**
   Jeder Run endet in einem solchen Zustand; die Frage "welche u-Muster sind
   nahe an Q=0-Becken" ist die erlernbare Struktur (aktuell 6 Beispiele;
   bei n≤48 mit höherer Lösungsrate viel mehr sammelbar).
4. **13% der Runs feuern nie einen Escape** (Budget-End im Grinding) — für
   diese wäre ein früherer Escape-Trigger (weniger als 3-mal am Rekordtief)
   billig zu testen.

## ML-Evaluation (Q6/Q8) auf den Escape-Features

- n=52 (6 Positive/494 Negative): keine Ein-Feature-Regel selektiv; kein
  Training möglich.
- n=44 (eigene Sammlung, 379 Runs, 231 gelöst = 61%): RandomForest 5-fold
  AUC ≈ 0.60-0.63, acc ≈ 0.61 (Baseline 0.60) — **schwaches, kaum nutzbares
  Signal**. Dezilen: unterste 10% ~18-24% Solve-Rate vs ~70% obere — eine
  minimale Vorfilterung wäre möglich, kein Durchbruch.
- Frequenz-Features (spec_flat, spec_corr, share_std, mirror_sum) verbessern
  das Modell NICHT (AUC 0.600 mit vs 0.632 ohne). Die Spektral-Komplementarität
  ist eine Eigenschaft ALLER niedrig-Q-Zustände, kein Diskriminator.
- Schlussfolgerung: Die Löse-Outcomes sind von der Trajektorien-Zufälligkeit
  (Becken-Erreichbarkeit) dominiert, nicht von messbaren Zustands-Features.
  ML-Potenzial liegt im Erlernen der LÖSUNGS-Verteilung selbst (PZinn-Stil),
  nicht in der Run-Vorhersage.

## Status

- [x] Q1 Escape-Mechanik + Flip-Muster (n=52, 500 Seeds)
- [x] Q2 Initial-Q-Vorhersage (n=52, 500 Seeds)
- [x] Q3 Trajektorie-Stats (n=52 + kbit-Benchmarks)
- [x] Q4 Lösungsstruktur (n=44/48/50/52)
- [x] Q6/ML-Bewertung (Datenlage)
- [x] Q1e Feintrajektorie der Sieger-Quenches (Tabu-Walk bricht Q≤8→0 durch)
- [x] Escape-Design-Vergleich A/B/C/E: top-5-Selektion nach Q_grid verliert
      alle Lösungen (6/500 → 0/500, 0/6, 0/6)
- [x] Spiegelpunkt-Lemma (Q=1→0-Charakterisierung, `mirror_point_lemma.md`)
- [x] Variante G (adaptiver Tabu bei Q≤1): 4/12 vs Baseline 7/12 — schlechter;
      Q=1-Durchbruch skaliert nicht mit Tabu-Schritten
- [x] n=44-Datensammlung: 379 Runs, 231 gelöst (61%), Escape+Spektral+Mirror-
      Features (`collect_n44.jsonl`, `collect_n44_spec.jsonl`)
- [x] ML-Pilot n=44: AUC ~0.6 (schwach); Frequenz-Features ohne Mehrwert
- [x] **Allgemeiner Satz d = -u** (algebraisch geprueft, `mirror_point_lemma.md`):
      Ein Single-Flip löst Q'=0 ⟺ d = -u. Konsequenz: Exception-Menge
      supp(u) statt Einzel-Lag; Korrektur des σ-Vorzeichens; P = 2^{-(m+1)}.
- [x] **Pre-Flip-Instrumentierung** (`preflip_solver.py`, `collect_preflip.py`,
      `preflip_n44.jsonl`): Bei 92+ echten Lösungen (n=44) wird der Zustand
      UNMITTELBAR vor dem lösenden Tabu-Flip geloggt (u_pre, d_solve, Q_pre,
      supp). **d = -u in 100% der Lösungen**; u_pre immer ternär;
      Q_pre ∈ {2,...,11} mit Modus 6 (NIEMALS Q_pre=1): Die finalen
      Übergänge sind Multi-Exception-Flips Q=4..9→0, kein Q=1-Spiegelpunkt.
      100% der lösenden Flips kommen aus dem Tabu-Walk (0 Greedy, 0 Kick),
      Walk-Eintritts-Q 2..13, der Walk erreicht C meist ÜBER seinem
      Q-Minimum (35/39: Aufwärts-Exkursion nötig).
- [x] **Cancellation Manifold C = {x : ∃(s,c), d_{s,c} = -u}**
      (`q10_preflip_analysis.py`, `q10_results.json`): dist0 = dist1 = 0 für
      ALLE 350 Escape-Zustände (n=44) und 435 (n=52) — C ist statisch
      unsichtbar, mindestens 2 Flips entfernt. min_1flip_q korreliert nicht
      mit Erfolg (corr +0.067): lokale C-Nähe des Escape-Zustands sagt
      nichts voraus (konsistent mit AUC ~0.6).
- [ ] Q5 PZinn-Transformer-Umsetzung (empfohlen nach Datensammlung:
      124+ Lösungen bei n=44 in solutions.json vorhanden)
- [x] **C-Defizit-Trajektorien** (`q11_c_deficit.py`, `q11_results.json`):
      Der Tabu-Walk nähert C NICHT graduell an (Schritt-Defizit-Korrelation
      -0.05, Defizit 1 Schritt vor dem Lösungs-Flip im Mittel 6.15);
      lösende und nicht-lösende Walks sind bis auf den C-Treffer
      ununterscheidbar (min-Defizit 3.79 vs 3.69). C-Dichte ~8e-5 pro
      Walk-Schritt => ~1.6% pro 200-Schritte-Walk => erklärt die ~60%-
      Solve-Rate. C wird getroffen, nicht konstruiert.
- [x] **Konstruktions-Versuch** (`q12_steepest_descent.py`,
      `q12_results.json`): Steilster Abstieg (global best-flip) und
      Kick-Descent lösen 0/350 Escape-Zustände (Boden Q~3). Kein lokales
      Regelwerk erreicht C; der Tabu-Walk bleibt das notwendige Instrument.
- [ ] Offene Frage: Warum kommt Q_pre=3 in 136 Lösungen nie vor
      (Q_pre in {2,4..11}, Modus 6)? Strukturelles Verbot oder
      extreme Seltenheit kombinierbarer 3-Exception-Muster.
- [x] **d²-Schale und C-Hazard** (`q13_d2_stats.py`, `q14_c_hazard.py`,
      `q15_targeted_escape.py`, `d2_shell_hazard.md`): ||d||² ~ Bin(m, 1/2)
      ist zustandsunabhängig (n=44: 10.5, n=50: 12.0); die Lösungsschale
      liegt im Niedrig-Q-Schwanz (97.8% der Treffer bei Q_pre 4..9).
      Per-Schritt-Hazard h(q) skaliert mit 2^(-q) (Modus bei Q 4..6,
      ~3e-2), Positions-Hazard steigt bis 7.2e-2 am Walk-Ende.
      Rekordtief-Zustände (Q=1) sind C-arm (0 Treffer in 920 Re-Seed-Walks,
      p<1.6e-3 bei erwarteten 15); Konstruktion C = 1-Flip-Umgebung von
      Lösungen numerisch verifiziert (38/38). Solver-Tabu-Walk ist
      DETERMINISTISCH (tabu_noise=0).
- [x] **n=50-Kollektion** (`collect_preflip_n.py`, `preflip_n50.jsonl`):
      100 Runs, Solve-Rate 4% (vs 60% n=44), Q_pre {6:2, 7:1, 8:1};
      1.25 Mio. Walks, per-Walk-Trefferrate 3.2e-6 vs 6.3e-5 (n=44),
      Faktor ~20 (> 2^(-3) = 8x Vorhersage).
- [x] **Varianten H/I** (`variant_h.py`): Re-Seeding (4 Rausch-Walks/
      Stuck-Punkt, ungezielt: 11 vs 9 p~0.69; gezielt an Entry-Q 3..12:
      10 vs 9 bei n=44, 1 vs 3 bei n=50) — kein zuverlässiger Effekt;
      4x Budget (base400) 9/20 = 9/20, 18/20 gleiche Seeds — bei
      tabu_noise=0 sind Extra-Walks deterministische Replays desselben
      Attraktors. Fazit: Walk-Politik-Hebel ausgemessen, die Solve-Rate
      wird von der C-Dichte der Zustandsverteilung bestimmt.
- [x] **QWindow-Solver** (`qwindow_solver.py`): Eigener Solver, der die
      Suche im C-reichen Q 4..9-Fenster hält (Descent-Stopp bei Q<=9,
      gedeckelter Rausch-Walk, kleine Kicks). n=44: 16/20 = 80% vs.
      Baseline 9/20 = 45% (McNemar p=0.065, gleiche Zeit, 170x mehr
      Walks/Run); n=50: 0/75 vs. 4/75 — die n=50-Baseline ist bereits
      schalendicht und die Schale ~20x C-dünner (Dichte-Wand).
- [x] **n-Sweep (QWindow, 30..52)**: Schwierigkeit streng monoton in
      m = (n-1)/2: n=30/40: 100% (0.0s/0.9s), n=42: 92%, n=44: 68%,
      n=46: 12%, n=48/50/52: 0%. Kein X2-Muster — n=52 ist die
      schwerste. Die 2^(-m)-Skalierung wird qualitativ bestätigt;
      für n>=48 fällt die Rate UNTER die Vorhersage (steilere Wand).
- [x] **Enrichment-Profil** (Teil 7 in `d2_shell_hazard.md`):
      Treffer(q) = Schale(q) x Walk-Zeit(q) x Anreicherung(q). Die
      Walk-Zustände sind bei Q=4 17x C-reicher als zufällig, bei Q=5
      5x, bei Q=8-9 C-ärmer. Die ideale Such-Schale ist Q 4..6
      (maximales Enrichment), nicht die Schalen-Masse bei Q 9-11.
- [ ] Hinweis: `archive_q1_q9/` enthält die abgeschlossenen q1-q9-Skripte
      und Zwischendaten (Ergebnisse im README dokumentiert); die aktuelle
      Linie (q10+) importiert sie nicht.
