# Audit: Basins, Passhöhen und Rückkehrwahrscheinlichkeiten

## Kurzfassung

Für den GS4-Suchraum müssen drei verschiedene Objekte sauber getrennt werden:

1. Die **intrinsische Energielandschaft** besteht aus Zuständen, zulässigen Moves
   und der exakten Energie `Q`.
2. Ein **Basin eines Quench-Verfahrens** entsteht erst durch eine festgelegte
   deterministische Abstiegsregel `G`, einschließlich Tie-Break und
   Scan-Reihenfolge.
3. Eine **Rückkehr- oder Übergangswahrscheinlichkeit** entsteht erst durch eine
   zusätzliche stochastische Exkursionsregel `K`, beispielsweise den aktuellen
   bandbegrenzten Shell-Walk mit Tabu.

Die minimale Passhöhe ist eine Eigenschaft der Landschaft. Basin-Hashes,
Rückkehrraten und Committor-Werte sind dagegen solver- beziehungsweise
dynamikabhängig.

## 1. Der diskrete GS4-Landschaftsgraph

Für festes `n` sei

\[
\Omega_n=\{-1,+1\}^{4\times n}
\]

der Zustandsraum der vier Folgen. Eine festgelegte Move-Menge `M` definiert
einen ungerichteten Graphen

\[
\mathcal G_M=(\Omega_n,E_M),
\]

wobei zwei Zustände benachbart sind, wenn ein Move aus `M` sie ineinander
überführt. Für die bisherigen Basin-Experimente ist `M` die Menge aller
`4n` Single-Flips. Sobald Symbol-, Pair- oder Intervallmoves hinzukommen, ist es
eine andere Landschaft.

Die Energie

\[
Q(x)=\lVert u(x)\rVert_2^2
\]

ist für einen Zustand eindeutig und vom Solver unabhängig. Ein Zustand `m` ist
ein Single-Flip-Lokalminimum, wenn

\[
Q(m)\le Q(y)\qquad\text{für alle }y\sim m.
\]

Falls neutrale Moves zugelassen werden, sollten zusammenhängende Plateaus oder
Sink-Komponenten statt einzelner Minima betrachtet werden. Andernfalls kann
eine willkürliche Tie-Break-Regel ein Plateau künstlich in mehrere „Basins“
zerteilen.

## 2. Solverabhängige Basins durch einen deterministischen Quench

Sei `G` eine vollständig festgelegte, terminierende und deterministische
Abstiegsregel. `G(x)` bezeichnet das terminale Minimum, das aus `x` erreicht
wird. Das zu `m` gehörende Basin ist dann

\[
B_G(m)=\{x\in\Omega_n:G(x)=m\}.
\]

Wenn `G` für jeden Zustand eindeutig definiert ist, partitionieren diese Mengen
den Zustandsraum.

Im aktuellen `experiments/learning/basin_probe.py` ist `G` konkret:

- alle Single-Flips auswerten,
- den Flip mit kleinstem Folge-`Q` wählen,
- bei Gleichstand den ersten flachen Array-Index wählen,
- nur strikt verbessernde Moves akzeptieren.

Das ist ein wohldefinierter **Best-Improvement-Single-Quench**. Sein Basin ist
aber nicht „das Basin des GS4-Zustands“. Eine andere Indexreihenfolge, ein
First-Improvement-Scan, neutrale Moves, `geo_weight`, Symbolmoves oder
Pair-Moves können `G(x)` ändern.

Der Production-Solver besitzt zudem Tabu-Gedächtnis, Tenure/Decay und optional
Zufall. Dieser Prozess ist auf dem Zustand `x` allein nicht deterministisch und
nicht einmal Markov. Dafür gibt es zwei saubere Möglichkeiten:

1. Basins ausschließlich für einen separaten deterministischen Quench `G0`
   definieren.
2. Den Zustand um Tabu-Tabelle, Iterationsphase und gegebenenfalls RNG-Zustand
   erweitern. Dann lebt das Basin nicht mehr nur im GS4-Bitraum.

Zwei scanunabhängigere, aber gröbere Begriffe sind möglich:

- **Schwaches Basin:** Alle Zustände, von denen irgendein erlaubter
  absteigender Pfad nach `m` führt. Schwache Basins können überlappen.
- **Starkes Basin:** Alle Zustände, deren jeder erlaubte absteigende Pfad in
  `m` endet. Dieses Basin kann wesentlich kleiner sein.

Auch diese Begriffe hängen von der Move-Menge und davon ab, ob nur strikt
fallende oder auch neutrale Kanten erlaubt sind. Sie hängen jedoch nicht von
einer konkreten Scan-Reihenfolge ab.

## 3. Bandbegrenzte Uphill-Exkursion und Rückkehr

Eine Exkursionspolicy definiert einen Übergangskern

\[
K_{H,L,T}(x,y)
=\Pr(\text{Exkursion von }x\text{ endet in }y).
\]

Dabei stehen beispielsweise:

- `H` beziehungsweise `[Q_low,Q_high]` für das erlaubte Energieband,
- `L` für die maximale Zahl akzeptierter Moves,
- `T` für Auswahlregel, Reverse-Verbot und Tabu-Policy.

Für ein Quench-Basin `B_G(m)` ist die Rückkehrwahrscheinlichkeit nach einer
Exkursion aus `m`

\[
R_{G,K}(m)
=\sum_y K(m,y)\,\mathbf 1\{G(y)=m\}.
\]

Allgemeiner ist die Übergangswahrscheinlichkeit zwischen zwei Quench-Minima

\[
P_{G,K}(m,m')
=\sum_y K(m,y)\,\mathbf 1\{G(y)=m'\}.
\]

Die Lösungswahrscheinlichkeit der Exkursion mit anschließendem Quench ist

\[
P_{\mathrm{solve}}(m)
=\sum_y K(m,y)\,\mathbf 1\{Q(G(y))=0\}.
\]

Diese Größen sind nützlich, aber nicht intrinsisch. Sie ändern sich mit
Energieband, Exkursionslänge, Move-Auswahl, Tabu-Tenure, Rücksprungverbot,
Tie-Break und Quench-Regel. Die aktuelle Hash-Gleichheit in `basin_probe.py`
misst daher korrekt die Rückkehr in dasselbe **`G`-Basin**, nicht in ein
solverunabhängiges Basin.

Empirisch wird `R` durch den Anteil zurückkehrender unabhängiger Replikate
geschätzt. Für Vergleiche sollten dieselben Startzustände, gepaarte Seeds und
Binomial-Konfidenzintervalle verwendet werden.

## 4. Intrinsische Passhöhe

Die minimale Passhöhe zwischen zwei Zuständen `a` und `b` ist

\[
\Phi(a,b)
=\min_{\gamma:a\leadsto b}\;\max_{x\in\gamma} Q(x),
\]

wobei über alle Pfade im festgelegten Move-Graphen minimiert wird. Die
gerichtete Barriere aus `a` ist

\[
\Delta(a\to b)=\Phi(a,b)-Q(a).
\]

Für eine Lösungsmenge `S_0={x:Q(x)=0}` gilt

\[
\Phi(a,S_0)=\min_{s\in S_0}\Phi(a,s).
\]

Äquivalent ist `Phi(a,b)` das kleinste `H`, für das `a` und `b` in dem durch

\[
\Omega_{\le H}=\{x:Q(x)\le H\}
\]

induzierten Subgraphen verbunden sind. Diese Definition hängt nur von `Q` und
dem Move-Graphen ab, nicht von Scan-Reihenfolge oder Tabu.

`experiments/learning/barrier_graph.py` setzt genau diese Sicht als BFS in einem
Sublevel-Graphen um. Ohne Node-/Tiefenabbruch kann ein erfolgloser vollständiger
Scan bis `H` beweisen, dass die Passhöhe größer als `H` ist. Ein wegen Limits
abgebrochener Scan liefert dagegen keinen solchen Beweis.

Ein Disconnectivity Graph fasst die Merge-Höhen der Sublevel-Komponenten über
steigendem `H` zusammen. Er wäre für kleine `n` eine solverunabhängige Karte der
Kraterstruktur. Für große `n` kann er nur ausschnittsweise angenähert werden.

## 5. Committor und Transition Paths

Ein Committor ist keine Eigenschaft von `Q` allein. Für einen festgelegten
Markov-Kern `P` und zwei Zielmengen `A,B` ist

\[
q_B(x)=\Pr_x(\tau_B<\tau_A).
\]

Außerhalb von `A\cup B` erfüllt er die diskrete harmonische Gleichung

\[
q_B(x)=\sum_yP(x,y)q_B(y),
\]

mit Randwerten `0` auf `A` und `1` auf `B`. Er misst daher die
dynamikabhängige Wahrscheinlichkeit, `B` vor `A` zu erreichen.

Für den aktuellen Tabu-Walk muss entweder der Zustand um das Tabu-Gedächtnis
erweitert oder eine bewusst memorylose Ersatzdynamik definiert werden. Die oben
definierte endliche Exkursion plus Quench ist mit einem Committor verwandt, aber
nicht identisch mit ihm.

## 6. Was ist intrinsisch und was solverabhängig?

| Größe | Status bei festem `M` und `Q` |
| --- | --- |
| Energie `Q`, Adjazenz, lokale Minima/Plateaus | intrinsisch |
| Sublevel-Komponenten, Passhöhe `Phi` | intrinsisch |
| Disconnectivity Graph | intrinsisch |
| Basin `B_G`, Basin-Hash | abhängig von `G` |
| Rückkehr- und Basinwechselrate | abhängig von `G` und `K` |
| Committor, reaktive Pfade, mittlere Trefferzeit | abhängig vom Übergangskern |
| exakt gezählter lokaler Branching-Grad unter einem Cap | intrinsisch lokal |
| gesampelter „Lidar“-Score/prospective degree | abhängig vom Sampler |
| Hamming-Distanz | koordinatenabhängig, aber nicht scanabhängig |

Damit ist die beobachtete `Q=1`-Wand zweigeteilt:

- Aussagen über fehlende Verbindungen in vollständig untersuchten
  `Q<=H`-Sublevel-Graphen sind echte Landschaftsaussagen für die betreffende
  Move-Menge.
- Aussagen wie „91 Prozent der Lösungen entstehen während einer
  Uphill-Exkursion“ beschreiben die konkrete Solver-Policy und dürfen nicht als
  allgemeines Gesetz der GS4-Landschaft gelesen werden.

## 7. Empfohlene kanonische Messdefinitionen

Für weitere Experimente sollten Namen und Parameter explizit eingefroren
werden:

1. `G_best_single`: strikter Best-Improvement-Single-Quench mit festem
   Index-Tie-Break.
2. `K_shell(H,L,T)`: uniform oder gewichtet gewählte bandbegrenzte Exkursion mit
   vollständig dokumentierter Tabu-Regel.
3. Getrennt berichten:
   - Rückkehr `R(G,K)`,
   - Wechsel in ein anderes `G`-Basin,
   - Lösung nach Quench,
   - intrinsische Passhöhen-Untergrenze aus vollständiger Sublevel-BFS.
4. Robustheit gegen Solverartefakte prüfen, indem mehrere feste Tie-Break- oder
   Scan-Reihenfolgen verwendet und die Übereinstimmung der Basin-Zuordnung
   gemessen wird.

So bleibt die „Kraterkarte“ mathematisch interpretierbar: `Phi` kartiert die
Topographie, `G` legt fest, wie bergab gelaufen wird, und `K` beschreibt, wie
der Solver zwischen Kratern springt.

## Primärliteratur und Terminologie

- Stillinger und Weber führten die Abbildung von Konfigurationen auf lokale
  Minima als *inherent structures* ein:
  [Phys. Rev. A 25, 978 (1982)](https://doi.org/10.1103/PhysRevA.25.978).
- Becker und Karplus formulieren Attraction Basins und Disconnectivity Graphs
  für multidimensionale Energielandschaften:
  [J. Chem. Phys. 106, 1495 (1997)](https://doi.org/10.1063/1.473299).
- Transition Path Theory und Committor-Funktionen werden als Eigenschaften
  einer festgelegten stochastischen Dynamik behandelt:
  [E und Vanden-Eijnden, J. Stat. Phys. (2006)](https://doi.org/10.1007/s10955-005-9003-9).
- Die übliche Communication-Height-Definition für diskrete metastabile
  Landschaften findet sich beispielsweise in
  [Cirillo und Nardi, J. Stat. Phys. (2015)](https://doi.org/10.1007/s10955-015-1334-6).
