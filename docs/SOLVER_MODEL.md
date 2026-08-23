# Technisches Modell des GS4-Solvers

Stand: 2026-08-09

Diese Datei beschreibt ausschließlich bestätigte Mathematik und den aktuellen
Produktionspfad. Die Beziehungen zu den alternativen Koordinaten stehen in
[`MATH_MAP.md`](MATH_MAP.md).

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

## Heuristischer Suchpfad

```text
greedy Singles -> Q-Window -> Tabu-Walk
               -> einmaliger Target-Escape mit Quench oder zufälliger Zwei-Bit-Kick
```

1. **Singles:** Batch-Scan mit Early Exit; jede unmittelbare Verbesserung wird
   akzeptiert.
2. **Tabu:** Bis zu 400 zustandsabhängige Single-Schritte dürfen bergauf gehen.
   Der Numba-Kernel merkt sich den besten besuchten exakten Trackerzustand.
   Übernommen wird er nur, wenn er besser als der Walk-Start ist.
3. **Target-Escape:** Wird das bisher niedrigste Q wiederholt erreicht, werden
   einmalig bis zu 625 strukturierte Vier-Sequenz-Kandidaten erzeugt. Jeder
   Kandidat startet einen eigenen ILS-Quench. Die Kandidaten werden über alle
   aktuell verletzten und lokal reparierbaren Residual-Lags verteilt.
4. **Random Kick:** Andernfalls werden Bits in zwei zufällig ausgewählten
   Folgen geflippt. Dieser schlechtere Zustand wird bewusst übernommen;
   anschließend beginnt der greedy Single-Abstieg erneut.

Pair-Rescue und Trace-Erfassung gehören nicht mehr zum Solver.

## Experimenteller Symbolsolver

Die CLI-Strategie `gs4-symbol` ist bewusst vom Produktionssolver getrennt. Sie
verwendet pro Spalte genau zehn Moves in einer festen gemeinsamen Reihenfolge:

```text
4n Single-Flips       = Wechsel zwischen den beiden H4-Basen
6n Spaltenpaare       = Bewegung innerhalb der aktuellen Basis
```

In jedem Schritt werden alle 10n exakten End-Q-Werte vektorisiert berechnet.
Der beste nicht-tabu Move wird auch dann ausgeführt, wenn er Q unverändert lässt
oder erhöht. Ein globales Best wird separat gespeichert und am Ende exakt in
den Tracker zurückgebaut. Es gibt keine Kicks, Q-Window-Logik, Targeted Escapes
oder geschachtelten Quenches.

Die Variante `gs4-symbol-channel` verfolgt zusätzlich für alle drei Paarungen
der vier Folgen die beiden Rohresiduen `w` und `v`. Sie rangiert einen Move nach

```text
score = Q' / (1 + max_pairing(||w'||² + ||v'||²) / 16).
```

Ein exakter Lösungsmove behält damit Score null. Die Normierung bevorzugt bei
vergleichbarem Q Zustände, deren zwei Kanäle bereits stark gegeneinander
arbeiten, ohne `w=0` und damit die Paley-/Golay-Unterfamilie zu erzwingen. Der
Kanalcache wird mit denselben Singleton-Deltas wie der Tracker aktualisiert.
Die Variante ist experimentell und nicht der Default.

## Deterministischer Konstruktionspfad

`paley-ng` ist vom heuristischen Solver getrennt. Wenn `p=2n-1` prim ist,
erzeugt die zweite Paley/Ito-Reihe direkt ein negaperiodisches Golay-Paar und
damit über den vorhandenen GS4-Builder eine exakte Hadamard-Matrix. Seed,
Schrittbudget, Tabu und Kicks werden dabei nicht verwendet.

Für `n=52` ist `p=103` prim. Die Matrix der Ordnung 208 wird daher ohne Suche
konstruiert. Die komplexe Halb-Längen-Faltung steht in
[`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md), direkte
Konstruktionen in [`CONSTRUCTION_SPACE.md`](CONSTRUCTION_SPACE.md).

## Zentrale Grenze

`Q` ist ein exaktes Fehlermaß, aber kein beobachtetes Maß für die spätere
Erreichbarkeit der Lösung. Das zustandsabhängige Flip-Wörterbuch ändert sich
nach jedem Move. Deshalb kann eine unmittelbare Verbesserung ein schlechteres
Einzugsgebiet öffnen und eine kontrollierte Verschlechterung langfristig
besser sein.

Die Tight-Frame-Sicht und die Kantenkoordinaten stehen in
[`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md).
Reproduzierte Suchresultate stehen in [`SEARCH_FINDINGS.md`](SEARCH_FINDINGS.md).
