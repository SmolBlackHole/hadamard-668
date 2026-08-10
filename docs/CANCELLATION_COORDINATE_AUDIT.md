# Audit: Cancellation-Koordinate und Zwei-Schritt-Shell

## Ergebnis in einem Satz

Das Cancellation-Barrieren-Lemma ist exakt. Die vier Einzel-NAF-Kanäle
bestimmen `Gamma` jedoch **nicht allgemein**: Ihre vollständige Konstanz bei
`n=5` und `n=6` ist ein Kleinordnungs-/Symmetrieeffekt, und bei `n=12` gibt es
ein explizites Gegenbeispiel mit gleichen vier `rho_s` und gleichem `u(Fx)`,
aber unterschiedlicher exakter `Gamma`-Länge.

Die neue gerichtete Shell ist ein korrekt berechneter stochastischer
**Two-ply receding-horizon lookahead**. Sie ist weder Pair-Rescue noch das
klassische `2-opt`: Sie bewertet exakte Zwei-Flip-Endpunkte, führt aber nur den
ersten Flip aus und plant am nächsten Zustand neu.

## 1. Exaktes Cancellation-Lemma

Fixiere einen Move-Graphen, seine Energie `Q`, die Lösungsmenge

\[
S_0=\{x:Q(x)=0\}
\]

und deren Nichtlösungs-Vorgänger

\[
C_1=\{x\notin S_0:\exists y\in S_0\text{ mit }x\sim y\}.
\]

Für eine Menge `Y` sei

\[
\Phi(x,Y)=\min_{\gamma:x\leadsto Y}\max_{z\in\gamma}Q(z).
\]

Dann gilt für jede Nichtlösung `x`, mit `+infinity` für unerreichbare Mengen,

\[
\boxed{\Phi(x,S_0)=\Phi(x,C_1)}.
\]

**Beweis.** Jeder Pfad von `x` zu seiner ersten Lösung enthält unmittelbar
davor einen Zustand aus `C1`; daher kann die minimale Passhöhe zu `C1` nicht
größer sein. Umgekehrt lässt sich jeder Pfad nach `C1` durch dessen lösenden
Move ergänzen. Dieser Endpunkt hat Energie null, also steigt die maximale
Pfadenergie nicht. Beide Ungleichungen ergeben Gleichheit.

Für die lexikographische Verfeinerung

\[
\Gamma(x)=(H_C(x),L_C(x))
\]

ist `L_C` die kürzeste Weglänge bis `C1` bei optimaler Höhe. Die kürzeste
entsprechende Strecke bis zur Lösung hat deshalb Länge `L_C+1`. Das vorhandene
Enumerationsskript speichert `solution_l`; für Nichtlösungen ist dies exakt um
eins größer als `L_C`. Das Lemma ist korrekt, lediglich die Benennung muss bei
der Interpretation beachtet werden.

## 2. Reproduktion von n=5 und vollständige Enumeration n=6

`navigation_synthesis.py` wurde unverändert erneut ausgeführt.

| Größe | n=5 | n=6 |
| --- | ---: | ---: |
| normalisierte Zustände | 65.536 | 1.048.576 |
| Lösungen | 7.500 | 105.408 |
| Zustände mit notwendigem Uphill | 0 | 0 |
| Fasern der vier `rho_s` mit variierendem `(H,L)` | 0 / 256 | 0 / 625 |

Bei `n=6` bestimmt sogar `u` allein die enumerierte Reachability-Koordinate;
bei `n=5` war das noch in vier `u`-Fasern falsch. Das wechselnde Verhalten ist
bereits eine Warnung gegen eine allgemeine Formel aus diesen Kleinordnungen.

## 3. Warum der Einzel-NAF-Lead klein-n exakt aussieht

Für eine einzelne Folge wurden alle global-vorzeichen-normalisierten Folgen
enumeriert und nach ihrem vollständigen NAF-Vektor gruppiert. Anschließend
wurde geprüft, wie viele unabhängige Orbits unter Negashift, Reversal und
globalem Vorzeichen in jeder NAF-Faser liegen.

| n | Folgen | NAF-Fasern | Fasern mit mehreren Symmetrieorbits |
| ---: | ---: | ---: | ---: |
| 5 | 16 | 4 | 0 |
| 6 | 32 | 5 | 0 |
| 7 | 64 | 9 | 0 |
| 8 | 128 | 12 | 0 |
| 9 | 256 | 23 | 0 |
| 10 | 512 | 34 | 0 |
| 11 | 1.024 | 63 | 0 |
| 12 | 2.048 | 96 | 6 |

Die verwendeten Transformationen wurden separat gegen den vollständigen
NAF-Vektor geprüft. Sie erhalten außerdem Single-Flip-Adjazenz, `Q`, `S0` und
`C1`. Wenn jede Einzelkanal-Faser nur ein solcher Orbit ist, sind zwei
Vierfolgenzustände mit denselben vier `rho_s` durch vier unabhängige
Graphautomorphismen verbunden. Dann muss `Gamma` identisch sein.

Damit erklärt Symmetrie die Beobachtung bis `n=11`; es ist dort noch keine
zusätzliche Reachability-Formel sichtbar. Bei `n=12` treten erstmals
nichtäquivalente homometrische Folgen auf.

## 4. Exaktes n=12-Gegenbeispiel

Das reproduzierbare Beispiel steht in
`experiments/learning/cancellation_coordinate_audit.py`. Zwei Zustände
unterscheiden sich nur in ihrer ersten Folge. Diese beiden Folgen besitzen
denselben NAF-Vektor, liegen aber in verschiedenen Negadihedral-Orbits.

Für die beiden vollständigen Vierfolgenzustände gilt exakt:

```text
vier rho_s gleich:       ja
u gleich:                (2, 2, 0, -2, 1)
Q gleich:                13
u(Fx) gleich:            (-2, -2, 0, -2, 3)
bestes Nachbar-Q:         5 / 5
lösende Singles:          0 / 0
nichtschlechtere Singles: 25 / 22
```

Eine vollständige BFS bis zur bereits durch den Start erzwungenen Höhe `13`
findet kürzeste Lösungspfade der Längen `3` und `4`. Da jeder Pfad den
Startzustand mit `Q=13` enthält und beide Pfade innerhalb `Q<=13` bleiben, ist
die minimale Passhöhe in beiden Fällen exakt `13`. Nach Abzug des letzten
Cancellation-Moves folgt

\[
\boxed{\Gamma(x_1)=(13,2),\qquad\Gamma(x_2)=(13,3).}
\]

Das falsifiziert gleichzeitig folgende allgemeinen Behauptungen:

- die vier `rho_s` bestimmen `Gamma`;
- die vollständigen invertierbaren H4-/Pair-Kanäle bestimmen `Gamma`;
- `(rho_1,rho_2,rho_3,rho_4,u(Fx))` bestimmt `Gamma`;
- ein bestes Nachbar-`Q` beziehungsweise ein skalares Cancellation-Deficit
  bestimmt `Gamma`.

Es widerlegt nicht, dass diese Größen statistisch nützliche Features sind.

## 5. Was andere vorgeschlagene Darstellungen leisten

### Pair- und H4-Kanäle

Die vollständigen vier H4-Kanäle sind eine invertierbare lineare
Transformation der vier `rho_s`. Sie enthalten exakt dieselbe Information und
erben daher das n=12-Gegenbeispiel. Skalare Kanalenergien enthalten noch
weniger.

### Dualkoordinate `u(Fx)`

Sie enthält zusätzliche, zustandsabhängige Information, rettet die
Eindeutigkeit aber nicht: Das Gegenbeispiel hält auch `u(Fx)` fest.

### Kantenwort

Das vollständige Kantenwort integriert sich bis auf das globale Vorzeichen
zur ursprünglichen Folge. Vier vollständige Kantenwörter bestimmen damit den
GS4-Zustand im verwendeten Gauge und folglich auch `Gamma`. Das ist korrekt,
aber keine Reduktion und noch keine einfache Formel: Die globale
Reachability-Suche ist lediglich anders kodiert.

### Flip-Wörterbuch `D`

Aus `u` und allen aktuellen Zeilen `d_i` folgt exakt der Ein-Schritt-Wert

\[
\min_i\lVert u+d_i\rVert^2.
\]

Damit sind direkte Cancellation und der lokale Deficit bekannt. `Gamma`
benötigt jedoch die Wörterbücher der Folgezustände `D(x')`; ein statisches
`D^T D` hilft nicht, weil es bereits durch `u` festgelegt ist. Eine einfache
Formel für das globale Minimax-/Längenproblem wurde nicht gefunden, und das
n=12-Beispiel zeigt ausdrücklich, dass Ein-Schritt-Aggregate nicht genügen.

## 6. Audit der gerichteten Zwei-Schritt-Shell

Für jeden aktuell erlaubten ersten Flip `i` berechnet die Policy

\[
V_2(i)=\min_{j\ne i}Q(x\mathbin{\mathrm{flip}}i
                         \mathbin{\mathrm{flip}}j)
\]

und führt nur das `i` mit kleinstem `V2` aus. Danach wird am neuen Zustand neu
gerechnet. Bei `sample_size=8` wird dies nur für acht zufällig ausgewählte
zulässige erste Moves getan.

Die richtige Einordnung ist:

- **nicht Pair-Rescue:** Der zweite Flip wird nicht ausgeführt;
- **nicht klassisches 2-opt:** Es wird kein permanenter Zweier-Austausch aus
  einem festen kombinatorischen Objekt übernommen;
- **Two-ply lookahead / rollout / receding horizon:** Der beste erreichbare
  Zwei-Schritt-Endwert bewertet die erste Aktion;
- wegen der quartischen GS4-Energie auch als exakte Tiefe-zwei-Auswertung der
  quartischen Zielfunktion beschreibbar, aber ohne neue Energieformel.

Operativ ist die Kombination neu und sinnvoll: Der vorhandene exakte
Pair-Scorer wird nicht als Rescue, sondern als Richtungsgeber innerhalb einer
Uphill-Shell verwendet.

### Korrektheit von `pair_qs_after_first`

60.800 Zwei-Flip-Endpunkte wurden bei `n=5,6,7,12,13,52` gegen einen frischen
Tracker nach tatsächlich ausgeführtem ersten Flip verglichen. Alle Werte waren
exakt gleich. Das deckt insbesondere gerade/ungerade Längen,
Same-Sequence-Korrekturen und Mittelpunktfälle ab.

Der Backflip wird korrekt behandelt:

- `pair_qs[first]` ist exakt das ursprüngliche `Q`, weil zweimal derselbe Flip
  den Zustand wiederherstellt;
- der Lookahead setzt diesen Eintrag auf unendlich;
- der Degree-Scorer subtrahiert genau diesen immer vorhandenen Rückweg, solange
  der aktuelle Zustand innerhalb des Caps liegt.

Eine wichtige semantische Einschränkung bleibt: Nur der **erste** Move wird
gegen Cap und aktuelle Tabu-Tabelle geprüft. Der hypothetische zweite Move
wird außer dem Backflip weder gegen die zukünftige Tabu-Tabelle noch gegen das
Cap gefiltert. `V2` ist daher ein optimistischer Lookahead-Wert, keine exakte
Simulation der nächsten Policyentscheidung.

### Unabhängige Reproduktion

Mit neuem Seed, je 128 unabhängigen Roots und unveränderten Parametern:

| n | Random tiefere Basins | Degree | Lookahead `V2` |
| ---: | ---: | ---: | ---: |
| 40 | 60 / 128 | 71 / 128 | 104 / 128 |
| 44 | 56 / 128 | 65 / 128 | 94 / 128 |
| 48 | 60 / 128 | 92 / 128 | 99 / 128 |
| 52 | 56 / 128 | 92 / 128 | 107 / 128 |

Erneut entstanden keine Lösungen. Der Lookahead benötigte im Mittel ungefähr
81.000 bis 89.000 gezählte Bewertungen, Random ungefähr 9.000 bis 10.000. Das
Signal für tiefere Zielbasins ist robust, aber noch kein kostenfairer
Solvergewinn und kein Cancellation-Hazard.

## 7. Konsequenz

Die vier Einzel-NAF-Kanäle bleiben sinnvolle Features, sind aber kein GPS.
Der bislang stärkste operative Sensor ist lokales, zustandsabhängiges
Zwei-Schritt-Lookahead. Er beantwortet jedoch nur

> „Welcher erste Shell-Move besitzt einen günstigen denkbaren zweiten Move?“

und nicht

> „Welcher Move senkt die exakte Cancellation-Barriere `Gamma`?“

Der nächste saubere Vergleich muss deshalb bei gleicher Kandidatenarbeit
prüfen, ob ein eingeschränktes `V2`, eine echte ausgeführte Pair-Policy oder
ein direkt auf `C1` gerichteter Zwei-Schritt-Deficit den größten
Downstream-Solve-Hazard erzeugt.

## Reproduktion

```text
python -m experiments.learning.cancellation_coordinate_audit
python -m experiments.learning.cancellation_coordinate_audit --full-n6
python -m experiments.learning.topology_guided_shell --ns 40 44 48 52 \
  --roots 128 --replicas 1 --sample-size 8 --workers 12 \
  --seed 20260814 --output data/directed-probe-shell-audit.json
```
