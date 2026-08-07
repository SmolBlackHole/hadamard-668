# Technisches Modell des GS4-Solvers

Stand: 2026-08-07

Diese Datei beschreibt ausschließlich bestätigte Mathematik und den aktuellen
Produktionspfad.

## Ziel

Gesucht sind vier Folgen `a,b,c,d in {+1,-1}^n`, deren negazyklische Blöcke in
der Goethals-Seidel-Konstruktion eine Hadamard-Matrix der Ordnung `4n` bilden.
Äquivalent gilt für alle nichttrivialen Lags:

```text
NAF_a(t)+NAF_b(t)+NAF_c(t)+NAF_d(t)=0.
```

## Reduziertes Residuum

Der kombinierte Residualvektor `r` besitzt nur

```text
m_eff = floor((n-1)/2)
```

unabhängige Koordinaten. Beim geraden `n` ist der Mittelpunkt-Lag `n/2`
identisch null. Weil alle Residuen durch vier teilbar sind, speichert der
Tracker

```text
u = r/4,
Q = ||u||_2^2,
E_repo = 64 n Q.
```

Damit sucht der Solver exakt nach `u=0`, ohne die volle Gram-Matrix zu bauen.

## Single-Flip-Scoring

Für das gecachte Residualdelta `d` eines Flips gilt:

```text
Q' = Q + 2 u^T d + ||d||_2^2.
```

Der Tracker speichert `d` als `int8` und seine Norm separat. Bis zu 64
Single-Scores werden mit einem Matrix-Vektor-Produkt ausgewertet. Der Solver
übernimmt den ersten gefundenen verbesserten Flip und behält dadurch den
Early-Exit-Vorteil.

Nach einem Accept werden nur die betroffenen Delta-Einträge mit
vorkomputierten Spalten-, Lag- und Vorzeichenindizes aktualisiert.

## Exakte Multi-Flips

`Tracker.combo_energy()` wertet eine beliebige Flipmenge exakt aus. NAF ist
quadratisch: Die Summe der Single-Deltas plus eine Korrektur für jedes
Same-Sequence-Paar ist vollständig. Es existieren keine zusätzlichen Triple-
oder Vierer-Energieterme.

Der allgemeine Prüfer bleibt für mathematische Experimente erhalten, ist aber
keine Phase des Produktionssolvers.

## Suchpfad

```text
greedy Singles -> Tabu-Walk -> zufälliger Vier-Bit-Kick
```

1. **Singles:** Batch-Scan mit Early Exit; jede unmittelbare Verbesserung wird
   akzeptiert.
2. **Tabu:** Bis zu 200 zustandsabhängige Single-Schritte dürfen bergauf gehen.
   Der Numba-Kernel merkt sich den besten besuchten exakten Trackerzustand.
   Übernommen wird er nur, wenn er besser als der Walk-Start ist.
3. **Kick:** Findet Tabu keine Verbesserung, wird je Folge ein zufälliges Bit
   geflippt. Dieser schlechtere Zustand wird bewusst übernommen; anschließend
   beginnt der greedy Single-Abstieg erneut.

Pair-Rescue und Trace-Erfassung gehören nicht mehr zum Solver.

## Zentrale Grenze

`Q` ist ein exaktes Fehlermaß, aber kein beobachtetes Maß für die spätere
Erreichbarkeit der Lösung. Das zustandsabhängige Flip-Wörterbuch ändert sich
nach jedem Move. Deshalb kann eine unmittelbare Verbesserung ein schlechteres
Einzugsgebiet öffnen und eine kontrollierte Verschlechterung langfristig
besser sein.

Die Tight-Frame-Sicht und die Kantenkoordinaten stehen in
[`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md).
