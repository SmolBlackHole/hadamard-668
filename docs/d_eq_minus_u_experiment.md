# d = -u: Exact-Cancellation-Experiment (laufendes Labor-Journal)

Stand: 2026-08-08. Alle Experimente in `experiments/learning/`, Solver-Snapshot
`dcfacc5` unverändert.

## Das zentrale Theorem (algebraisch geprueft)

Ein Single-Flip löst (Q'=0) **genau dann**, wenn sein Delta-Vektor das
Residuum exakt kancelt:

```text
Q' = ||u + d||^2  =  0   <=>   d = -u
```

Da `d_t in {-1, 0, +1}`, folgt: Der Zustand VOR dem lösenden Flip ist nur dann
direkt lösbar, wenn alle Residuen ternär sind, `u_t in {-1, 0, 1}`. Damit ist
`Q_pre = |supp(u_pre)|` (Q = Anzahl der Exceptions).

Komponentenweise (Deltaformel `d[s,c](t) = -(1/2) x_c (x_{c+t} + x_{c-t})`):

- `u_t = 0`  => `x_{c-t} = -x_{c+t}` (Antisymmetrie, "Spiegelpunkt")
- `u_t = +1` => `x_{c-t} = x_{c+t} = +x_c` (positive Exception)
- `u_t = -1` => `x_{c-t} = x_{c+t} = -x_c` (negative Exception)

Die Exception-Menge ist `supp(u)`, NICHT nur ein einzelner Lag. Das Q=1-Lemma
(Spiegelpunkt mit Exception k) ist der Ein-Exception-Sonderfall. **Korrektur**
gegenüber `mirror_point_lemma.md`: bei `u = -e_k` gilt `x_{c-k} = x_{c+k} = -x_c`
(Vorzeichenfehler im alten Lemma; Seltenheit `2^{-(m+1)}`, nicht
`2^{-(n+2)/2}` — Faktor 2 daneben).

## Zentrale Zielmenge: Cancellation Manifold

```text
C = { x : existiert (s,c), d_{s,c}(x) = -u(x) }
```

Der Solver löst also, wenn der Tabu-Walk transiente Zustände in C erreicht:

```text
beliebiger Zustand -> Tabu veraendert das Move-Woerterbuch D -> x in C -> ein Flip -> Q=0
```

## Experiment 1: Pre-Flip-Instrumentierung (`preflip_solver.py`, `collect_preflip.py`)

**Was:** Kopie des dcfacc5-Pipelines mit erweitertem Numba-Tabu-Kernel, der im
Moment des lösenden Flips (q+delta_q == 0) den Zustand UNMITTELBAR davor
snapshotted: `u_pre`, `d_solve`, `Q_pre`, `supp(u_pre)`, Flip-Position (s,c),
Walk-Eintritts-Q, Flip-Nummer im Walk, Q-Trajektorie des Walks. Zusätzlich
unabhängige Rekonstruktion: `d` aus der Closed-Form auf dem rekonstruierten
Pre-Zustand, Vergleich `d == -u`.  Greedy-Descent und Random-Kick werden
ebenfalls gefangen (Pfad "greedy" bzw. "kick4").

**Daten:** `preflip_n44.jsonl` — ein JSON pro Run (n=44, 100k steps),
~8-12 s/Run, Solve-Rate ~60%. Batch 1: Seeds 6000-6084 (90 Runs, 53 gelöst).
Batch 2: Seeds 6120-6185 mit Q-Trajektorie. Batch 3: Seeds 6200-6274 mit
C-Defizit-Trajektorien (deficit_hist pro Walk-Schritt).

**Validierung des Loggings:** Der Kernel-Snapshot (pre_seq) wird mit der
Rekonstruktion "gelöster Zustand, Flip zurückdrehen" verglichen
(`tabu_pre_seq_match`) — 100% Übereinstimmung.

### Endergebnisse (228 Runs, 136 Lösungen, n=44)

1. **d = -u gilt in 136/136 Lösungen** (`d_equals_minus_u_all: true`). Die
   Mirror-Exception-Struktur gilt in 136/136 (`mirror_exception_ok_all: true`).
   `max|u_pre| = 1` in allen Fällen (ternär). Kein Logging-Artefakt.
2. **Alle 136 über den Tabu-Walk** (0 via Greedy, 0 via Kick). Der lösende
   Zustand wird nie von Greedy gesehen — Tabu erreicht C von innen.
3. **Q_pre-Verteilung: {2:1, 4:6, 5:16, 6:43, 7:32, 8:27, 9:9, 10:1, 11:1}**
   — Modus 6, NIEMALS Q_pre=3, nur 1x Q_pre=2. Die letzten lösenden
   Übergänge sind Multi-Exception-Flips Q=4..9 -> 0 (Ausnahmen 2, 10, 11).
   Der klassische "Q=1-Spiegelpunkt" tritt in keiner dieser 136 Lösungen als
   letzter Schritt auf. Median Q_pre = 6.
4. **Walk-Eintritts-Q: 2..13.** Der Tabu-Walk startet teils bei Q=13
   (nicht-ternäres u!) und findet C bei Q_pre 2-11. Flip-Nummer im Walk:
   mean 102, max 199 von 200 — der Walk braucht praktisch das ganze Budget.
5. **Pre-Q in 80/86 (93%) ÜBER dem Walk-Minimum:** Der Walk macht
   Aufwärts-Exkursionen in Q, um C zu treffen. Trajektorien-Tails
   oszillieren bei Q 6-9 direkt vor dem Lösungs-Flip. Der Walk sinkt NICHT
   zu Q=1, bevor er löst — er trifft C bei höherem Q. Starke Unterstützung
   für die C-Charakterisierung: Nicht "je niedriger Q desto besser", sondern
   "C-Zustand mit exaktem Gegen-Residuum".
6. **Exception-Lags:** Histogramm über die 21 Lags fast uniform (Mittel ~27,
   Range 22-39 pro Lag über 613 Exceptions in 92 Lösungen; 287 positive vs
   326 negative Exceptions — beide Sigma-Typen, die sigma=-1-Korrektur ist
   empirisch relevant). Kein bevorzugter Lag, keine Positionierung um c.
   Flip-Zentren und Sequenzen uniform verteilt.
7. **Exact-Cancellation ist der letzte Schritt JEDER Lösung** — aber nie
   bei Q_pre in {1, 3}: Q_pre=1-Zustände sind Sackgassen (kein
   Mirror-Motiv), Q_pre=3 kommt strukturell offenbar kaum vor (3
   Exceptions + Rest-Antisymmetrie ist extrem selten kombinierbar).

## Experiment 3: C-Defizit-Trajektorien (`q11_c_deficit.py`, Batch 3)

**Was:** Jeder Tabu-Walk zeichnet pro Schritt das C-Defizit auf: das minimale
Q, das EIN Flip vom aktuellen Zustand erreicht (0 = Zustand in C). Frage:
Nähert sich der Walk C graduell an (steuerbar) oder trifft er C zufällig?

### Ergebnisse (47 lösende Walks, 2953 nicht-lösende Walks)

1. **Kein Steuersignal:** Schritt-Defizit-Korrelation -0.046; das Defizit
   1 Schritt vor dem Lösungs-Flip ist mean 6.15 (der C-Treffer ist ein
   EINZELNER Sprung von Defizit ~6 auf 0, kein graduelles Absenken).
   KEIN lösender Walk erreicht je Defizit 1 vor dem Lösen; das min Defizit
   vor dem Lösen ist mean 3.79 (>= 2 immer).
2. **Lösende vs nicht-lösende Walks ununterscheidbar:** min Defizit 3.79 vs
   3.69; finale Defizite 0 vs 7.0 (trivial). Die einzige Unterscheidung ist
   der C-Treffer selbst.
3. **C-Dichte:** 47 C-Schritte in 595.587 Walk-Schritten insgesamt ~= 8e-5
   pro Schritt. Pro Walk (200 Schritte) ~1.6% Treffer-Wahrscheinlichkeit;
   ~40 Walks pro Run -> ~50-60% Solve-Rate pro Run (passt zur Messung).
   Nicht-lösende Walks haben 0/590.600 C-Schritte (per Score-Argument
   tautologisch: ein C-Zustand wird vom Walk immer sofort gelöst, Score 0
   ist global minimal).
4. **Fazit Konstruktion:** Der Tabu-Walk ist ein Zufalls-Sampler über
   Zustände; C ist eine seltene Teilmenge (~1e-4); es gibt KEINE
   beobachtbare monotone Annäherung an C. Ein Greedy auf dem Defizit
   (steilster Abstieg) löst 0/350 Escape-Zustände (q12), auch mit Kicks
   (Boden bei Q~3).

## Experiment 4: Konstruktions-Versuch (`q12_steepest_descent.py`)

**Was:** Kann C konstruktiv erreicht werden? Getestet auf allen 350
Escape-Zuständen (n=44): (a) steilster Abstieg (immer der Single-Flip, der
Q global minimiert), (b) steilster Abstieg + 4-Flip-Random-Kick bei
lokalen Minima (2000 Schritte).

### Ergebnisse

1. **0/350 Lösungen in beiden Varianten.** Min Q: mean 3.0, median 3
   (Histogramm Q=1:2, Q=2:81, Q=3:184, Q=4:83) — identisch mit und ohne
   Kicks. Steilster Abstieg landet im selben Q~3-Floor wie der
   first-improving-Greedy des Solvers.
2. **Fazit:** Kein lokales Regelwerk (best-flip, first-improving, mit/ohne
   Kicks) erreicht C. Der Tabu-Walk ist dafür das notwendige
   Zufalls-Instrument — C wird getroffen, nicht konstruiert.

## Experiment 2: Distanz zur Cancellation Manifold (`q10_preflip_analysis.py`)

**Was:** Für alle gesammelten Escape-Zustände (n=44: `collect_n44_spec.jsonl`,
350 Stück; n=52: `all_seeds_escape.json`, 435 Stück) exakt berechnet:
- `dist0` = Anzahl Flips mit d == -u (Zustand in C),
- `dist1` = existiert EIN Flip zu einem Zustand in C (über die Deltamatrix
  aller 4n Nachbarn, vektorisiert),
- `min_1flip_q` = min Q nach einem Flip (= ||d+u||^2-Minimum, "Defekt").

Korrelation mit Solve-Erfolg (n=44, 228 gelöst / 122 ungelöst).

### Ergebnisse

1. **dist0 = 0 für alle 350 (n=44) und alle 435 (n=52) Escape-Zustände.**
   Kein Escape-Zustand ist in C. Das "Motiv ist transient" — jetzt exakt.
2. **dist1 = 0 für alle.** Auch EIN Flip von einem Escape-Zustand erreicht C
   nie. C ist von Escape-Zuständen mindestens 2 Flips entfernt.
3. **min_1flip_q: gelöst 5.75 vs ungelöst 5.62 (n=44)** — praktisch identisch;
   Korrelation corr = +0.067 (kein Signal). Die lokale C-Nähe des
   Escape-Zustands sagt NICHTS über den Lösungs-Erfolg voraus.
   (n=52: min_1flip_q mean 7.6, median 8 — gleichförmig.)
4. Konsistent mit Q6/Q8 (AUC ~0.6): Die Outcome-Zufälligkeit wird durch die
   Walk-Trajektorie bestimmt (erreicht der Walk C?), nicht durch lokale
   Merkmale des Escape-Zustands.

## Interpretation (final)

- Die alte Formulierung "Tabu-Walk 8->0" (README Q1e) war der Spezialfall der
  neuen Wahrheit: Der Tabu-Walk trifft einen Zustand in C mit Q_pre in 2..11
  (Modus 6), dessen u ternär ist und dessen Delta-Zeile das Residuum exakt
  darstellt.
- "Der letzte Schritt jeder Lösung ist per Lemma ein Spiegelpunkt-Flip" gilt
  nur für Q_pre=1 (nie beobachtet). Allgemein ist der letzte Schritt ein
  Multi-Exception-C-Flip (d=-u) mit |supp(u)| = 2..11, fast immer 4..9.
- **Der C-Treffer ist ein Zufalls-Ereignis, kein Steuerziel:** C-Dichte
  ~8e-5 pro Walk-Schritt, kein monotones Defizit-Signal, lösende und
  nicht-lösende Walks sind bis auf den Treffer identisch. Das erklärt
  quantitativ die 60%-Solve-Rate: ~1.6% Treffer-Wahrscheinlichkeit pro
  200-Schritte-Walk, ~40 Walks pro Run.
- Die mathematische Struktur von C (Mirror-Exception-Set) ist exakt
  charakterisiert und empirisch zu 136/136 validiert — die Nutzung als
  Konstruktions-Regel ist damit NICHT möglich (Q12), aber die Struktur
  definiert, wonach jeder zukünftige Such-Algorithmus suchen muss.

## Nächste Experimente (offen)

1. **Beziehung C -> Tight-Frame**: Der lösende Flip kancelt U(zeta_k) in
   allen Bins; bei Q_pre = k ist |U| konstant k auf supp. Verbindung zur
   Spektral-Komplementarität (Q7-Befund) ausarbeiten.
2. **C-Dichte analytisch**: Warum genau ist P(Spiegelpunkt mit Exception-
   Menge S) ~ 2^{-|S|}-skaliert? Gibt es eine Formel fuer die beobachtete
   Dichte 8e-5? (n=44, Q~6-Zustände in Tabu-Nähe.)
3. **Q_pre=3-Phänomen**: Warum kommt Q_pre=3 nie vor? (2,4-11 beobachtet,
   3 fehlt.) Vermutung: |S|=3 erzwingt u + d = 0 mit 3 Exceptions und
   komplette Antisymmetrie ausserhalb — kombinatorisch deutlich seltener
   als gerade Anzahlen? Oder strukturelles Verbot?

## Offene Dateien

- `preflip_solver.py` — instrumentierter Solver (Kernel mit Pre-Flip-Snapshot,
  Q-Trajektorie und C-Defizit pro Schritt)
- `collect_preflip.py` — Sammlung (preflip_n44.jsonl: 228 Runs, 136 Lösungen)
- `q10_preflip_analysis.py` — Theorem-Validierung + C-Distanz (q10_results.json)
- `q11_c_deficit.py` — Defizit-Trajektorien (q11_results.json)
- `q12_steepest_descent.py` — Konstruktions-Versuch (q12_results.json)
- `d_eq_minus_u_experiment.md` — dieses Journal
- `mirror_point_lemma.md` — aktualisiert um den allgemeinen Satz d = -u
