# Benchmark

Stand: 2026-08-09

**Solver:** Singles + Tabu-200 + Kick, `tenure=5`, `decay=0.7`, `noise=0.0`

**Tracker:** reduziertes NAF-Residual, `int8`-Delta-Cache, Batch-Early-Exit

**Tabu:** Numba-Kernel

## Referenz-Sweep

100 Seeds, 200.000 Steps:

| n | Ordnung | Gelöst | 95-%-Intervall | Zeit/Run |
| ---: | ---: | ---: | :---: | ---: |
| 24 | 96 | 100/100 | 0,96–1,00 | 71 ms |
| 28 | 112 | 100/100 | 0,96–1,00 | 9 ms |
| 32 | 128 | 100/100 | 0,96–1,00 | 47 ms |
| 36 | 144 | 91/100 | 0,84–0,95 | 266 ms |
| 38 | 152 | 58/100 | 0,48–0,67 | 486 ms |
| 40 | 160 | 17/100 | 0,11–0,26 | 667 ms |
| 42 | 168 | 6/100 | 0,03–0,13 | 703 ms |
| 44 | 176 | 2/100 | 0,01–0,07 | 710 ms |

## Ungerade Größen

100 Seeds, 200.000 Steps:

| n | Ordnung | Gelöst | Zeit/Run |
| ---: | ---: | ---: | ---: |
| 35 | 140 | 92/100 | 327 ms |
| 37 | 148 | 46/100 | 562 ms |
| 39 | 156 | 28/100 | 602 ms |
| 41 | 164 | 4/100 | 688 ms |
| 43 | 172 | 3/100 | 696 ms |

## Große gerade und ungerade Größen

50 Seeds, 6,4 Millionen Steps, vier Worker:

| n | Gelöst | Bestes E | Bestes Q | Zeit/Run |
| ---: | ---: | ---: | ---: | ---: |
| 51 | 0/50 | 3264 | 1 | 8,4 s |
| 52 | 0/50 | 3328 | 1 | 11,6 s |
| 53 | 0/50 | 3392 | 1 | 7,9 s |
| 54 | 0/50 | 3456 | 1 | 12,1 s |
| 55 | 0/50 | 7040 | 2 | 7,9 s |
| 56 | 0/50 | 3584 | 1 | 14,0 s |
| 57 | 0/50 | 7296 | 2 | 9,9 s |
| 58 | 0/50 | 7424 | 2 | 11,9 s |

Ungerade Größen waren in diesem Sweep ungefähr ein Drittel schneller. Der
wahrscheinliche technische Anteil ist, dass sie keinen identisch toten
Mittelpunkt-Lag mitschleppen. Ein Suchdurchbruch war damit nicht verbunden:
Beide Paritäten endeten an derselben `Q=1/2`-Barriere. Dass `n=56` bis `Q=1`
kam, während `n=55` nur `Q=2` erreichte, zeigt außerdem, dass die beobachtete
Wand nicht strikt mit `n` wächst.

## Budget-Skalierung

100 Seeds:

| n | 20k | 50k | 100k | 200k |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 16 % | 49 % | 68 % | **91 %** |
| 38 | 6 % | 22 % | 36 % | **58 %** |
| 40 | 2 % | 4 % | 8 % | **17 %** |
| 42 | 0 % | 0 % | 3 % | **6 %** |
| 44 | 0 % | 0 % | 1 % | **2 %** |

Der Solver hat mit 12 Millionen Steps auch Lösungen bei `n=52` gefunden. Die
kleine Zahl gefundener Lösungen reicht dort noch nicht für eine stabile
Solve-Rate-Schätzung.

## Historische Pair-Entscheidung

Die gepaarte 1000-Seed-Ablation ergab ohne Pair-Rescue:

- `n=36`: 938 statt 905 Lösungen, `p=0,010161`;
- `n=38`: 619 statt 532 Lösungen, `p=0,000101`;
- `n=40`: 254 statt 221 Lösungen, `p=0,093042`.

Pair aus war außerdem schneller. Der Pfad wurde deshalb vollständig entfernt.
Die Einordnung steht in [`SEARCH_FINDINGS.md`](SEARCH_FINDINGS.md).

## Interpretationsregel

Für neue Suchphasen zählen gepaarte Solve-Rate und CPU-Zeit pro Lösung. Eine
niedrigere mittlere Energie oder mehr akzeptierte Verbesserungen genügen nicht:
Online-Gray senkte `Q` häufig und reduzierte dennoch die Solve-Rate.
