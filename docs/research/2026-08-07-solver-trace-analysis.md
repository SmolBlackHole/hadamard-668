# Empirische Forensik des GS4-Solvers

Stand: 2026-08-07

> Archiviertes Untersuchungsprotokoll. Aktuelle Hypothesen und Aufgaben stehen
> in [`../HYPOTHESES.md`](../HYPOTHESES.md) und [`../TODO.md`](../TODO.md).

## Fragestellung

Der Solver findet inzwischen GS4-Hadamard-Matrizen deutlich oberhalb der
früheren praktischen Grenze. Untersucht wurde deshalb, ob Initialisierung,
NAF-Energie, Singles, Pair-Rescue, Tabu und Kicks gemeinsam eine zusätzliche,
bisher nicht explizite Struktur optimieren.

Der wichtigste Befund ist kein unbekanntes zweites Skalarziel. Der Solver ist
eine adaptive diskrete Least-Squares-Suche, deren Move-System beim Annähern an
eine Lösung selbst zu einem Tight Frame wird. Tabu nutzt diese zunehmend gut
konditionierte Geometrie, während Pair-Rescue in der aktuellen Reihenfolge
wahrscheinlich sogar schadet.

## Messaufbau

- Vollständig gelesen: `builder.py`, `generator.py`, `solver.py`, `tracker.py`,
  `HYPOTHESIS.md`, `RESEARCH.md` und alle Tests.
- Endlösungen: committed Snapshot `HEAD:data/solutions.json`, da die aktuelle
  Datei während des laufenden Sweeps bewusst neu aufgebaut wird. Enthalten
  waren n=24 bis n=40 mit 16 bis 217 Einträgen je n.
- Zufallskontrollen: unabhängige gleichverteilte ±1-Folgen gleicher Größe.
- Pfadtraces: temporäre, rein instrumentierende Wrapper um `_update_best`,
  `_rescue`, `_tabu_walk` und `Tracker.accept`; keine Produktionslogik wurde
  verändert.
- Trace-Stichproben:
  - n=30: 40 Seeds, 100k Steps.
  - n=35: 30 Seeds, 200k Steps.
  - n=38: zuerst 40, danach 60 Seeds, jeweils 200k Steps.
- Ablation Pair-Rescue: dieselben Seeds mit `pairs=True/False` bei n=38 und
  n=40, sonst identische Defaults.
- Kleine exakte Enumeration: alle Einzelfolgen und alle geordneten Vierer für
  die resultierenden NAF-Signaturen bis n=14.
- Alle Wegwerfskripte wurden nach den Messungen entfernt.

## Exakt validierte mathematische Struktur

### 1. Nur ungefähr n/2 Bedingungen

Für vier Folgen der Länge n gilt

```text
r_t = NAF_a(t) + NAF_b(t) + NAF_c(t) + NAF_d(t)
r_{n-t} = -r_t
```

Bei geradem n ist zusätzlich `r_(n/2)=0` identisch. Die Zahl echter
unabhängiger Residuen ist daher

```text
c = floor((n - 1) / 2).
```

Für n=52 sind das 25 ganzzahlige Bedingungen bei 208 Bits, nicht Tausende
unabhängiger Bedingungen der fertigen 208×208-Matrix. Der GS4-Builder macht
die übrige Orthogonalität automatisch.

Für jede einzelne Folge wurde außerdem geprüft:

```text
NAF_x(t) = n - 2t  (mod 4).
```

Deshalb ist die Summe der vier NAF-Werte durch vier teilbar und der Tracker
arbeitet exakt mit `u_t=r_t/4`.

### 2. Exakter 2n-Lift und relative Difference Family

Für `y=(a,-a)` gilt exakt

```text
PAC_y(t) = 2 NAF_a(t),                  0 < t < n
PAC_y(n) = -2n.
```

Für vier gelöste Folgen ist die summierte zyklische Autokorrelation der
gelifteten Folgen deshalb nicht die eines gewöhnlichen flachen zyklischen
Komplementärsatzes, sondern

```text
C(0)=8n,  C(n)=-8n,  C(t)=0 sonst.
```

Entsprechend ist das Summenspektrum geformt:

```text
sum_s |FFT(a_s,-a_s)[k]|² = 0      für gerade k
                           16n     für ungerade k.
```

Die freie zyklische Relaxation muss daher dieses Zweiniveauspektrum beachten;
ein gewöhnliches Flat-Spectrum-Ziel ist das falsche Problem.

Setzt man `D_s={j : y_s[j]=-1}` in `Z_(2n)`, enthält jedes `D_s` genau ein
Element aus jedem antipodalen Paar `{j,j+n}`. Für die geordneten
Differenzmultiplizitäten

```text
N_s(t) = |D_s ∩ (D_s - t)|
```

ist die Hadamard-Bedingung äquivalent zu

```text
sum_s N_s(t) = 2n     für t nicht in {0,n}.
```

Der Solver sucht damit auch nach vier antipodalen Transversalen einer
zyklischen relativen Difference Family. Ein Bitflip tauscht genau die beiden
Vertreter eines antipodalen Paars.

### 3. Der Delta-Cache ist ein adaptives Tight Frame

Sei `D` die Matrix der reduzierten Single-Flip-Deltas. Sie hat `4n` Zeilen
(Moves) und `c` Spalten (echte Residuen). Mit der geraden, antiperiodisch
fortgesetzten Residualfolge und `r_0=4n` gilt für Lags `t,v` exakt

```text
(DᵀD)_(t,v) = 1/2 · (r_(v-t) + r_(v+t)).
```

Damit folgt für jede Lösung:

```text
DᵀD = 2n I_c.
```

Das wurde für alle geprüften Lösungen n=24…40 und zusätzlich für eine frisch
erzeugte ungerade n=35-Lösung bitgenau bestätigt. Für gerade n wurde an
Zufallszuständen n=6…14 zusätzlich die Identität

```text
||DᵀD - 2nI||_F² = 4(n-4) Q
```

bestätigt. Diese proportionale Frobenius-Identität gilt in dieser einfachen
Form nicht für die verwendeten reduzierten Koordinaten bei ungeradem n.

Interpretation: Je kleiner Q wird, desto näher kommen die möglichen
Single-Moves einem isotropen Tight Frame im Residualraum. Die Größen `D u`,
die der Solver ohnehin in den Single-Scores berechnet, sind dessen
Frame-Koeffizienten. Greedy Singles verhalten sich daher wie Matching Pursuit
in einem sich selbst konditionierenden, aber nach jedem Flip verändernden
Dictionary. Tabu erlaubt nichtmonotone Änderungen dieses Dictionarys.

### 4. Spektrale Schranke jeder einzelnen Folge

An jeder Nullstelle `z^n=-1` gilt in einer Lösung

```text
sum_s |A_s(z)|² = 4n.
```

Wegen der Nichtnegativität folgt für jede Komponente notwendig

```text
|A_s(z)|² <= 4n.
```

Das erklärt einen Teil der beobachteten Glättung der Einzelfolgen, ohne eine
zusätzliche Solverenergie annehmen zu müssen.

## Empirische Endzustände

Die Summe der quadrierten individuellen NAFs war in Lösungen deutlich kleiner
als in gleich großen Zufallskontrollen:

| n | Lösungen | Lösung / Zufall |
| ---: | ---: | ---: |
| 24 | 210 | 0.696 |
| 26 | 217 | 0.664 |
| 28 | 216 | 0.664 |
| 30 | 204 | 0.673 |
| 32 | 143 | 0.653 |
| 34 | 117 | 0.655 |
| 36 | 93 | 0.620 |
| 38 | 51 | 0.626 |
| 40 | 16 | 0.670 |

Das ist real, aber nicht automatisch ein verstecktes Nebenziel: Schon die
notwendige individuelle PSD-Schranke schließt spektral spitze Folgen aus. Bei
n=38 waren nur 27.5% der zufälligen Vierer so beschaffen, dass alle vier
Einzelfolgen die Schranke an allen negazyklischen Frequenzen erfüllten.

Bei n=38 hatten gelöste Zustände und erfolglose beste Q=1-Zustände praktisch
dieselbe individuelle NAF-Rauheit (1767 gegen 1750 im Mittel). Diese Größe
trennt also gerade den entscheidenden letzten Erfolg nicht.

Die vier individuellen NAF-Vektoren einer Lösung summieren sich zu null. Ihre
mittlere paarweise Kosinusähnlichkeit lag bei n=30 und n=38 bei ungefähr
-0.32, nahe -1/3. Das sieht wie ein fast reguläres Simplex aus, folgt aber
weitgehend schon aus `sum_s v_s=0` und ähnlich großen Normen; kein zusätzlicher
Invarianzbefund.

Für n≡2 mod 4 wurde der kleine Charakter `z=i` geprüft. Die sortierten vier
Ganzzahlnormen `|A_s(i)|²` bilden zwar immer eine Partition von `4n`, aber es
gab keine einzelne bevorzugte Lösungsklasse:

- n=30: 28 verschiedene Partitionen unter 204 Lösungen.
- n=38: 26 verschiedene Partitionen unter 51 Lösungen; häufigste nur 6-mal.

Eine starre Startbedingung aus genau einer solchen Partition ist daher nicht
plausibel. Die kleinen Charaktere bleiben als grober hierarchischer Filter
interessant.

## Pfaddynamik

### Aktueller n=52-Trace: schneller Abstieg, danach eine echte Mehrbit-Barriere

Der inzwischen verfügbare produktive Trace
`data/traces/gs4-n0052-seed010000.npz` enthält 380 Snapshots eines Runs
mit 12 Mio. Budget-Schritten und 35.4 Mio. tatsächlichen Bewertungen. Der Run
endete ungelöst bei Best-Q=2.

Der globale Bestwert entwickelte sich so:

```text
Initial Q=128
Singles: Q=8 bei Eval 620
Tabu:    Q=6 bei Eval 1,931
         Q=5 bei Eval 13,335
         Q=4 bei Eval 14,646
         Q=3 bei Eval 264,600
         Q=2 bei Eval 5,376,385
danach kein Q=1/0 bis Eval 35.4 Mio.
```

Im ersten Q=2-Bestzustand waren nur

```text
u_18=+1, u_23=-1
```

nicht null. Eine unabhängige exakte Auswertung dieses Snapshots ergab:

- bester Single-Nachbar: Q=7;
- kein neutraler oder verbessernder Single;
- exhaustive Auswertung aller `C(208,2)=21,528` Bitpaare: ebenfalls
  bestes Q=7, kein verbesserndes Paar.

Dieser Zustand ist damit nicht nur für die Top-K-Pair-Heuristik schwierig. Er
ist eine echte Hamming-Radius-2-Barriere: Jeder Ein- oder Zwei-Bit-Move wird
deutlich schlechter. Das erklärt, warum weitere Millionen Bewertungen ohne
neue Move-Geometrie wirkungslos bleiben können, und stärkt den Fall für einen
gezielten Q<=2-Mehrbit-/Beam-Move. Es ist bislang ein einzelner n=52-Trace,
also noch keine Aussage über alle n=52-Fehlläufe.

### Replikation mit 150 vollständigen n=52-Traces

Ein anschließender Sweep mit 150 Seeds, 12 Millionen Budgetschritten und 12
Workern löste 2/150 Runs. Alle 150 Runs erreichten `Q<=2`; ihre globalen
Endbestwerte waren:

```text
Q=0:   2 Runs
Q=1:  62 Runs
Q=2:  86 Runs
```

Die beiden Lösungen, Seeds 93 und 140, sprangen innerhalb eines Tabu-Walks
direkt von `Q=2` auf `Q=0`. Keine der beiden Trajektorien besaß zuvor einen
global besten `Q=1`-Zustand:

```text
Seed 93:  Q=2 bei Eval 3.655.826  -> Q=0 bei Eval 16.694.750
Seed 140: Q=2 bei Eval   735.270  -> Q=0 bei Eval 18.433.473
```

Das erstmalige Erreichen von `Q<=2` trennte Erfolg und Misserfolg nicht. Die
Medianwerte lagen bei etwa 2,20 Millionen echten Bewertungen für gelöste Runs,
2,22 Millionen für spätere `Q=1`-Fehlläufe und 2,74 Millionen für spätere
`Q=2`-Fehlläufe; die Verteilungen überlappten breit.

Damit ist der frühe Abstieg in den niedrigen Residualraum bei n=52 praktisch
gelöst. Die verbleibende Seltenheit liegt in der diskreten Realisierbarkeit
eines Escape-Pfads aus einem konkreten `Q=1/2`-Wörterbuch, nicht im Erreichen
dieser Energieniveaus. Das stützt gezielte Low-Q-Lookaheads stärker als weitere
globale Startfilter oder eine bloße Erhöhung des Greedy-Budgets.

### Startbedingungen

In 40 n=38-Runs unterschieden sich erfolgreiche und erfolglose Starts kaum:

| Startmetrik | später gelöst | später ungelöst |
| --- | ---: | ---: |
| Q | 164.5 | 181.4 |
| Residual-Support | 15.8 | 16.3 |
| individuelle NAF-Quadratsumme | 2638 | 2628 |
| bester Single-Delta-Q | -43.4 | -46.8 |

In der unabhängigen 60-Seed-Stichprobe lag Start-Q bei später gelösten Runs
bei 160.5 und bei ungelösten bei 196.0. Niedrigeres Start-Q kann ein schwacher
Prädiktor sein; die übrigen getesteten Startgrößen lieferten kein Signal. Das
ist noch kein Startfilter: Selektion nach Q wurde bereits indirekt durch frühere
Filterexperimente als wenig wirksam eingeschätzt und muss gepaart getestet
werden.

### Q=1 ist kein Plateau, sondern eine strikte Falle

Im ersten n=38-Test endeten alle 18 erfolglosen Runs mit exakt

```text
Q=1, support(u)=1, |u_t|=1.
```

Im detaillierten Trace wurden die lokalen Q=1-Zustände untersucht:

| n | Q=1-Zustände | neutraler Single | direkter Single nach Q=0 |
| ---: | ---: | ---: | ---: |
| 35 | 474 | 2 Zustände mit je einem | 0 |
| 38 | 697 | 0 | 0 |

Insbesondere bei n=38 sind die beobachteten Q=1-Zustände strikte
Single-Minima. Tabu muss zuerst bergauf gehen und das Delta-Frame verändern,
bevor das letzte Residuum überhaupt löschbar wird.

Die Escape-Raten in den detaillierten Traces waren:

| n | Phase | Q=1 | Q=2 |
| ---: | --- | ---: | ---: |
| 35 | Pair-Rescue | 0/474 | 5/1953 = 0.26% |
| 35 | Tabu | 3/474 = 0.63% | 194/1948 = 9.96% |
| 38 | Pair-Rescue | 0/697 | 9/4934 = 0.18% |
| 38 | Tabu | 0/697 | 177/4925 = 3.59% |

Bei n=38 entstanden alle 34 Lösungen der 60 Runs direkt innerhalb eines
Tabu-Walks; Pair-Rescue schloss keine einzige Lösung ab. Bei n=35 kamen 24 von
28 Lösungen direkt aus Tabu und vier aus Pair-Rescue.

Tabu ist bei n=38 also nicht deshalb erfolgreich, weil es zuverlässig aus Q=1
herausläuft. Es verhindert oder überspringt die Falle, indem es aus Q>=2/3
einen Pfad bis Q=0 findet. Scheitert dies, landet die äußere Suche häufig in
Q=1, kickt wieder heraus und probiert eine neue niedrigenergetische Region.

### Keine bevorzugte Lag-Reparaturfolge gefunden

Bei gelösten n=30- und n=38-Runs hatte der vorletzte globale Bestzustand fast
immer Support eins. Der letzte noch nicht verschwindende Lag war aber breit
über alle unabhängigen Lags verteilt. Auch die Q=2-Tabu-Erfolgsrate zeigte in
den Stichproben keine monotone oder robuste Abhängigkeit vom Abstand der zwei
nichtnulligen Lags.

Die Hypothese „dieselben Frequenzen/Lags werden immer zuerst oder zuletzt
repariert“ wird durch diese Daten nicht gestützt. Ein n=52-Trace kann dennoch
größenabhängige Effekte zeigen.

## Überraschender Pair-Rescue-Befund

Pair-Rescue wurde auf denselben Seeds deaktiviert, alle anderen Defaults
blieben gleich:

| n | Konfiguration | gelöst | Zeit/Run |
| ---: | --- | ---: | ---: |
| 38 | Pairs an | 34/60 | 244 ms |
| 38 | Pairs aus | 49/60 | 230 ms |
| 40 | Pairs an | 6/40 | 546 ms |
| 40 | Pairs aus | 9/40 | 391 ms |

Das ist ein Explorationsbefund, noch kein finaler Defaultwechsel. Für den
gepaarten McNemar-Test wurden die einzelnen Discordants in diesem Wegwerflauf
nicht persistiert. Das Signal ist dennoch stark genug für eine große saubere
Ablation.

Plausibler Mechanismus: Die aktuelle Reihenfolge ist

```text
Single -> Pair -> Tabu -> Kick.
```

Pairs liefern zwar monotone Verbesserungen, können den Zustand aber gerade in
die Q=1-Falle drücken, in der Tabu bei n=38 empirisch 0/697 Erfolg hatte. Ohne
Pairs bekommt Tabu den noch reicheren Q>=2/3-Zustand und kann die Lösung direkt
innerhalb des Walks treffen. „Verbessert Q sofort“ und „verbessert die spätere
Solve-Wahrscheinlichkeit“ sind hier nicht dasselbe.

## Kleine exakte Suchraumzählung

Bis n=14 wurden alle `2^n` Einzelfolgen nach ihrer NAF-Signatur gruppiert und
die Anzahl geordneter komplementärer Vierer exakt gezählt:

| n | unabhängige Residuen | NAF-Signaturen | mittlere Faser | Zufalls-P(Lösung) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 3 | 12 / 256 | 21.3 | 2^-5.71 |
| 10 | 4 | 34 / 1024 | 30.1 | 2^-8.17 |
| 12 | 5 | 96 / 4096 | 42.7 | 2^-10.70 |
| 14 | 6 | 317 / 16384 | 51.7 | 2^-13.67 |

Die große Faser kommt wesentlich aus Negashift, Umkehrung und Vorzeichen;
zusätzlich wirken Blockpermutationen. Ein grobes unabhängiges Gaußmodell gibt
für n=52 eine rohe Trefferwahrscheinlichkeit um `2^-79`, gleichzeitig aber
etwa `2^129` rohe Lösungen im gesamten `2^208`-Raum. Die Extrapolation ist
keine Zählformel, zeigt aber die richtige Größenordnung: Lösungen können sehr
zahlreich und für zufälliges Sampling trotzdem praktisch unsichtbar sein.

Die beobachteten Millionen Bewertungen sind daher keine Enumeration. Der
Solver gewinnt viele Größenordnungen, indem er im nur 25-dimensionalen
Residualraum navigiert.

## Widerlegte oder derzeit nicht gestützte Hypothesen

1. **Q=1 ist ein neutrales Plateau:** falsch für die beobachteten n=38-Minima;
   es gab keinen neutralen Single-Move.
2. **Pair-Rescue ist der typische letzte Retter:** falsch in der n=38-Stichprobe;
   alle Lösungen wurden innerhalb von Tabu abgeschlossen.
3. **Ein bestimmter Lag wird systematisch zuletzt repariert:** kein Signal.
4. **Eine einfache Startmetrik trennt Erfolg und Misserfolg:** außer einem
   schwachen Start-Q-Signal kein Befund.
5. **Eine einzige kleine spektrale Partition charakterisiert Lösungen:**
   widerlegt durch viele `z=i`-Partitionen.
6. **Gewöhnliche zyklische Flat-Spectrum-Optimierung ist nach dem Lift exakt:**
   falsch; das korrekte Zielspektrum ist 0 auf geraden und 16n auf ungeraden
   Bins.
7. **Niedrige individuelle NAF-Rauheit erklärt den letzten Erfolg:** nein;
   Lösungen und gescheiterte Q=1-Bestzustände waren darin gleich.

## Offene Hypothesen und konstruktive Moves

### A. Tabu vor Pair oder ganz ohne Pair

Der stärkste unmittelbare Kandidat ist nicht eine neue Formel, sondern die
Phasenreihenfolge:

```text
Single -> Tabu -> optional Pair -> Kick
```

oder zunächst schlicht `pairs=False`. Das ist kleiner als eine neue Rescue-
Implementierung und wird direkt vom Wegwerfvergleich gestützt.

### B. Q=1 braucht einen Barrier-Move, keinen besseren Single

Für `u=±e_t` existierte bei n=38 kein neutraler oder lösender Single. Ein
sinnvoller Spezialmove muss deshalb bewusst kurzfristig Q erhöhen und die
Ableitungsgeometrie verändern. Kandidaten:

- begrenzte 3- bis 8-Bit-Subset-Suche nur bei Q=1;
- kleiner Beam über den ersten bergauf führenden Flip und anschließende exakte
  Single-/Pair-Auswertung;
- Gray-Code über die nach `D u` ausgewählten Frame-Richtungen;
- Auswahl des ersten Flips danach, wie viele direkte Löschrichtungen im
  aktualisierten Delta-Frame entstehen, nicht nur nach dem unmittelbar neuen Q.

Eine globale Triple-Suche wurde bereits verworfen. Die neue Hypothese ist
enger: Mehrbit-Lookahead ausschließlich im nachgewiesenen Q=1-Flaschenhals.

### C. Tight-Frame-Rekonstruktion als Proposal Generator

Nahe der Lösung gilt näherungsweise

```text
u ≈ (1/(2n)) Dᵀ(Du).
```

`D u` ist bereits vorhanden. Seine größten Koeffizienten könnten statt nur
für unabhängige Singles als koordinierter Subset-Vorschlag dienen. Das ist die
diskrete Entsprechung eines Frame-/Matching-Pursuit-Schritts; akzeptiert würde
weiterhin ausschließlich mit dem exakten Tracker-Q.

### D. Hierarchische 2n-Charaktere

Die Lift-Difference-Family kann zuerst auf Quotienten von `Z_(2n)` betrachtet
werden. Der kleinste Charakter wird durch die Zweierpotenz in n bestimmt:

- n ungerade: `z=-1`, vier ganzzahlige Quadrate summieren sich zu `4n`;
- n≡2 mod 4: `z=i`, vier Gaußnormen summieren sich zu `4n`;
- höhere Zweierpotenzen: entsprechend kleine 8., 16., … Einheitswurzeln.

Diese Bedingungen sind keine eindeutigen Lösungssignaturen, könnten aber eine
mehrstufige Initialisierung oder einen Proposal-Generator bilden: zuerst den
kleinen Quotienten reparieren, dann die verbleibenden Frequenzen. Ein bloßer
Startfilter ist wegen des negativen Cyclo-Experiments weniger attraktiv als
ein konstruktiver Repair-Move.

## Priorisierte nächste Tests

1. **Pairs sauber ablatieren:** mindestens 1000 identische Seeds bei n=38 und
   n=40, zusätzlich n=52 soweit bezahlbar; `pairs=True/False`, McNemar,
   Zeit/Lösung und vollständige Phase-Counter. Danach optional Reihenfolge
   `Single -> Tabu -> Pair` testen.
2. **Q-Übergänge dauerhaft loggen:** Phase, Q vorher/nachher, nichtnullige
   Lags, Tabu-Minimum/-Maximum, letzte erfolgreiche Phase. Nicht jeden Step
   speichern; Ereignisse und ausgewählte Q-Snapshots genügen.
3. **n=52-Endlösungen und Pfade auswerten:** dieselben PSD-, Tight-Frame-,
   Q=1/2- und Charaktermetriken gegen n=38/40 vergleichen. Zum Zeitpunkt
   dieser Analyse lag noch keine neue n=52-Lösung in der laufenden Datei vor.
4. **Gezielter Q=1-Lookahead:** nur auf gespeicherten Q=1-Snapshots 3–8 Bit,
   Beam/Gray und Frame-Koeffizienten vergleichen; Metrik ist Escape nach Q=0
   pro CPU-Sekunde.
5. **Quotienten-Repair als Wegwerfprototyp:** kleine Charakterverletzung vor
   und nach aktuellen Moves messen. Nur weiterverfolgen, wenn sie den späteren
   Erfolg besser vorhersagt als Q allein.

## Arbeitsmodell

Der derzeit beste gemeinsame Nenner ist:

```text
vier Binärfolgen
    -> vier realisierbare NAF-/Difference-Vektoren
    -> deren Summe u soll null werden
    -> 4n Bitflips bilden ein adaptives Move-Frame D
    -> mit Q -> 0 wird D automatisch fast isotrop
    -> Singles betreiben Matching Pursuit
    -> Tabu verändert das Frame nichtmonoton und überspringt Q=1-Fallen
    -> Kick liefert einen neuen Anlauf, wenn der niedrige Frame-Zustand tot ist.
```

Das erklärt sowohl die hohe Leistung als auch die Budgetskalierung, ohne eine
geheime zweite Zielfunktion vorauszusetzen. Die aussichtsreichste unmittelbare
Änderung ist, Tabu nicht mehr durch monotone Pairs in schlechte Q=1-Basins
vorstrukturieren zu lassen. Die aussichtsreichste mathematische Erweiterung
ist ein koordinierter Mehrbit-Proposal aus der Tight-Frame- oder
2n-Difference-Family-Sicht.
