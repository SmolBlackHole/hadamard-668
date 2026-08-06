# Recherche- und Ablationsnotizen

## Belastbare Ausgangslage

Die Suche arbeitet im Raum von vier negaperiodischen \(\pm1\)-Folgen, nicht
direkt im Raum beliebiger Hadamard-Matrizen. Die exakte Zielfunktion und ihre
Reduktion sind in [HYPOTHESIS.md](HYPOTHESIS.md) dokumentiert.

Die NAF-Tracker-Energie entspricht der echten Gram-Energie von `best_seq`.
Sie zählt ungeordnete Off-Diagonalpaare; `solver_e` und `metrics.energy` müssen
deshalb dieselbe Normierung verwenden.

## Korrigierte Implementierungsbefunde

- Ein Single-Flip wirkt auf zwei Produktterme je Lag; sein Gesamtdelta liegt in
  \(\{-4,0,4\}\), nicht in einem einzelnen Produktterm.
- Der unabhängige Residualraum hat nur \(\lfloor n/2\rfloor\) Koordinaten.
  `r_{n-t}=-r_t`; bei geradem \(n\) ist der mittlere Lag null.
- Für Mehrfachflips sind Single-Deltas plus Paar-Korrekturen innerhalb derselben
  Folge vollständig. Triple- und größere Moves benötigen keinen Neuaufbau der
  NAF.
- Ein Paar mit Abstand `n // 2` darf nur bei geradem `n` wegfallen. Die frühere
  Bedingung übersprang bei ungeradem `n` fälschlich einen echten Lag.
- Rescue-Bewertungen waren bisher nicht Teil des nominellen Schrittbudgets.
  Ablationen müssen daher Laufzeit und getrennte Evaluationszähler berichten;
  ein reines `steps`-Vergleichsbudget wäre irreführend.

## Laufende Ablationen

Gray-Code-Subset-Rescue bleibt bewusst **ausgeschaltet**, bis die bestehenden
Ablationsläufe abgeschlossen und vergleichbar dokumentiert sind. Dasselbe gilt
für neue Startbedingungen und Spektral-Moves: Sie ändern die Suchdynamik und
gehören in eigene Experimente.

Die vorhandenen Konfigurationen isolieren Singles, Pairs, Triples, Kick und
Restart. Triple-Moves werden nur ausgeführt, wenn `SolverConfig.triples=True`;
damit bedeutet die Ablationsflagge tatsächlich das, was sie behauptet.

## Externe Vergleichsideen

`pzinn/hadamard` und das zugehörige Paper zu *Generating Hadamard matrices with
transformers* verwenden einen zirkulanten Raum und eine andere Scorefunktion
(spektraler Log-Det-artiger Score). Die Suchideen sind interessant, aber die
Angabe `n=50` dort bezeichnet nicht ohne Weiteres dieselbe Blocklänge wie hier.
Vergleiche müssen immer Matrixordnung und Folgenlänge getrennt nennen.

Übertragbar als spätere, getrennt zu testende Hypothesen:

1. Tabu-Walk statt oder vor zufälligem Kick.
2. Top-k-Subset-Rescue; zunächst naiv und nur nach belegtem Nutzen Gray-Code.
3. Negaperiodische Phasen-Reparatur auf den Wurzeln von \(z^n+1\), nicht auf
   den gewöhnlichen DFT-Frequenzen.

Nicht direkt übertragbar ist eine feste gewöhnliche Segment- oder Nullmodus-
Nebenbedingung aus zirkulanten Konstruktionen. Für ungerade \(n\) liegt
allerdings \(z=-1\) auf dem negaperiodischen Spektrum und liefert eine eigene
notwendige Alternating-Sum-Bedingung; das ist eine Hypothese für einen separaten
Initialisierungsversuch, keine aktuelle Solver-Restriktion.

## Nächste Experimente nach dem Baseline-Freeze

1. Mit den neuen Zählern die gleichen Konfigurationen erneut messen.
2. `kick` mit und ohne aktivierte Triples isolieren.
3. Erst danach einen Tabu-Walk als einzelne neue Variable testen.
4. Spectral Repair erst evaluieren, wenn lokale Moves nachweislich stagnieren.

## Quellen

- P. Zinn et al., *Generating Hadamard matrices with transformers*,
  arXiv:2604.11101.
- K. Đoković und I. Kotsireas, *Negaperiodic Golay pairs and Hadamard
  matrices*, arXiv:1508.00640.
