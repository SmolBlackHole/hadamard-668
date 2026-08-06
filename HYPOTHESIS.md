# GS4-Energie via negaperiodischer Autokorrelation

Stand: 2026-08-07, Commit `96b4fa3`

## Was der Solver löst

Gesucht sind vier Folgen `a,b,c,d ∈ {±1}^n`, deren negazyklische Matrizen
in der Goethals-Seidel-Konstruktion eine Hadamard-Matrix der Ordnung 4n bilden.
Das ist äquivalent zu einem **negaperiodischen komplementären Sequenzsatz**:

```
NAF_a(t) + NAF_b(t) + NAF_c(t) + NAF_d(t) = 0,   t = 1,…,n-1
```

## Zielfunktion und Normierung

Die Repository-Metrik zählt ungeordnete Off-Diagonalpaare:

```
E_repo = ½‖HHᵀ - 4nI‖_F² = 2n Σ r_t² = 4n Σ_{t=1}^{⌊n/2⌋} r_t²
```

Alle `r_t` sind durch 4 teilbar. Der Tracker speichert:

```
u_t = r_t / 4,   Q = Σ u_t²,   E_repo = 64n·Q
```

Das halbiert die unabhängige Lag-Dimension und erlaubt einen `int8`-Delta-Cache.

## Single-Flip

Sei `d` das reduzierte Delta eines Flips:

```
Q' = Q + 2uᵀd + ‖d‖²
```

`‖d‖²` ist im Cache als `norm2` gespeichert — kein `count_nonzero` mehr im
Hot-Path.

## Batch-Flip

Für einen Block von k Flips (k ≤ 64 im Solver):

```
ΔQ_block = 2D_block·u + norm2_block
```

Ein Matmul ersetzt k einzelne Dot-Produkte. Der Solver bricht beim ersten
Treffer ab (Early-Exit), daher zahlen Batches vor allem bei erfolglosen Scans.

## Mehrfach-Flips (Rescue)

Für eine Menge von Flips genügt die Summe ihrer Single-Deltas plus für jedes
Paar derselben Folge genau eine Korrektur. Triple-Terme existieren nicht.
Der Abstand `⌊n/2⌋` bei ungeradem n ist kein Sonderfall.

## Inkrementelles Delta-Update

Nach einem akzeptierten Flip: nur O(n) statt O(n²). Vorkomputierte Geometrie
(lag, sign, betroffene Cache-Zeilen) wird beim `build()` einmal berechnet und
bei `accept()` als reine Indexed-Addition angewandt.

## Spektrale Formulierung

Für `ζ_k = exp(iπ(2k+1)/n)` (Nullstellen von z^n+1):

```
Σ |X(ζ_k)|² = 4n   für alle k
```

```
E_repo = 2 Σ p_k²,   p_k = Σ|X(ζ_k)|² - 4n
```

Die Spektraldarstellung reduziert die Freiheitsgrade nicht, ist aber die Basis
für spätere Spektralreparatur (ganze Folge aus Spektren der anderen drei
rekonstruieren).

## Performance (Consumer-Hardware, 12 Cores)

| Operation | n=32 | n=167 |
| --- | --- | --- |
| Build + Scan | 0.3ms | 1.2ms |
| 200k Steps Run | 313ms | 554ms |
| 100 Seed Sweep | ~3s wall | ~5s wall |

Der NAF-Tracker ist **17× schneller** als der initiale AutocorrTracker und
**~5000× schneller** als der ursprüngliche GramTracker.

## Einordnung

- Kombinatorische Konstruktion von Hadamard-Matrizen
- Signalverarbeitung/Radar: flache Leistung auf negaperiodischen Frequenzen
- Quartische {±1}-Optimierung / 4-Spin-Ising-Modell
- Lokale Suche auf diskretem Vektor-Balancing-Problem
