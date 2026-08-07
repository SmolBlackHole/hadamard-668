# Recherche- und Ablationsnotizen

Stand: 2026-08-07, Commit `96b4fa3`

## Erledigt

- [x] **NAF-Tracker** — `u = r/4`, `int8` Delta-Cache, 8× kompakter als vorher
- [x] **Batch-Flip** — `flip_batch()` Matmul statt 64 einzelner Dot-Produkte
- [x] **Vorkomputierte Update-Geometrie** — accept() ist reine Indexed-Addition
- [x] **norm2-Cache** — `_single_delta_q` kein `count_nonzero` mehr
- [x] **Ablationstest** — 7 Configs, n=32, 30 Seeds
  - Triples: 1000-Seed-Test zeigt p=0.855 (nicht signifikant) → Code entfernt
  - Restarts: mit korrigiertem Tracker kein Beleg für Nutzen → Code entfernt
  - rescue_mode: kein Effekt → Code entfernt
  - Z-Test + Wilson-CI in `src/benchmark_stats.py`
- [x] **Budget-Scaling** — n=32 erreicht 99% bei 1.6M Steps (reines Compute-Limit)
  - n=32: 5%→99% über 50k→1.6M
  - n=34: 2%→78%
  - n=36: 2%→46% (flacht ab)
  - n=38: 0%→14% (Suchdynamik-Limit)
- [x] **Wilson-CI** in Sweep-Output
- [x] **Phase-Timing** — single_time, rescue_time, kick_time, rebuild_time
- [x] **Evals-Counter** — single_evals, rescue_evals, kick_evals pro Run
- [x] **SolverConfig** mit Feature-Flags
- [x] **Benchmark** — 100 Seeds × 16 n in 36s Wall-Clock

## Offene Hypothesen

### 1. Tabu-Walk (nächster Ablationstest)

Bei lokalen Minimum (Plateau): bestes erlaubtes Single-Bit flippen (auch wenn
schlechter), kürzlich verwendete Bits für Tenure ~n/4 tabu setzen, 50-200
Schritte laufen. Stärkster Einzelkandidat gegen Plateaus.

Referenz: `pzinn/hadamard` `improve.py`

Test: n=36,38 mit Tabu statt Kick. Parameter: Tenure = n/4, 100 Steps.

### 2. K-Tuning

Aktuell `K = sqrt(B)*3`. Testen: 3√B, 4√B, 5√B bei n=32,36.

### 3. Adaptives Kick

1→2→4 Bits pro Sequenz statt immer 1. Erst nach Tabu-Walk testen.

### 4. Gray-Code Subset-Rescue

Statt nur Pairs: top-k Bits, alle 2^k-1 Teilmengen testen. k=8→255,
k=9→511 Kombinationen. Erst naiv, dann Gray-Code mit inkrementellem Update.

### 5. Cyclotomischer Startfilter — verworfen

Bei n=36 brachte der Filter in 1000 Seeds keinen signifikanten Gewinn:
Default 63/1000, Cyclo 69/1000 (`p=0.589`), Tabu 623/1000 und Cyclo+Tabu
645/1000 (`p=0.307` gegen Tabu). Die Implementierung wurde wieder entfernt.

### 6. Spektralreparatur

Negazyklische FFT-basierte Ganzfolgenreparatur. Eine Folge aus Spektren der
anderen drei rekonstruieren. Größter strategischer Umbau, zurückgestellt.

## Nicht übertragbar

- Feste Segment-Summen (zirkulant) — nicht negazyklisch
- Gewöhnliche DFT-Frequenzen — wir brauchen Nullstellen von z^n+1

## Quellen

- P. Zinn et al., *Generating Hadamard matrices with transformers*, arXiv:2604.11101
- K. Đoković und I. Kotsireas, *Negaperiodic Golay pairs and Hadamard matrices*, arXiv:1508.00640
