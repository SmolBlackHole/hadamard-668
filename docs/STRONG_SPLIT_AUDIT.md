# Unabhängiger Audit: starke H4-Zerlegung `A=C=0`

Stand: 2026-08-09

## Ergebnis in einem Satz

Die starke Zerlegung ist algebraisch korrekt und eine echte GS4-Unterfamilie,
aber kein neues unabhängiges Energieobjekt: Sie fordert exakt, dass sowohl ein
GS4-Tupel `X=(a,b,c,d)` als auch sein spaltenweises Tripleprodukt-Dual

```text
F(X) = (bcd, acd, abd, abc)
```

GS4-Lösungen sind. Diese Dualbeschreibung ist der klarste neue Projektbefund;
eine wissenschaftliche Neuheit oder eine Identität mit einer bekannten
Konstruktionsfamilie ist damit nicht belegt.

## 1. Exakte Herleitung

Setze an jeder Spalte

```text
x_j    = (a_j,b_j,c_j,d_j)^T,
beta_j = a_j b_j c_j d_j,
v_j    = H4 x_j / 4.
```

`beta_j` unterscheidet die beiden H4-Basen. Trenne die negaperiodische
Skalarprodukt-Autokorrelation von `v` danach, ob zwei Spalten dieselbe oder
verschiedene Basen verwenden:

```text
u_t = A_t + C_t.
```

Das beweist unmittelbar die Suffizienz

```text
A=0 und C=0  =>  u=0  =>  GS4.
```

Für gerades `n` fehlt dabei keine Bedingung. Der Lag `n/2` ist für jede
negaperiodische Autokorrelation identisch null; der reduzierte Bereich
`1..floor((n-1)/2)` ist vollständig.

### Die wichtigere Dualform

Definiere die spaltenweise Involution

```text
F(x_j) = beta_j x_j
       = (b_j c_j d_j, a_j c_j d_j, a_j b_j d_j, a_j b_j c_j)^T.
```

Es gilt `F(F(X))=X`, und `F` erhält das Basislabel `beta`. Für zwei Spalten
ändert sich ihr Skalarprodukt um den Faktor `beta_j beta_k`: Same-Basis-Terme
bleiben gleich, Cross-Basis-Terme wechseln das Vorzeichen. Deshalb gilt exakt

```text
u(X)  = A + C,
u(FX) = A - C.
```

Folglich

```text
A = (u(X)+u(FX))/2,
C = (u(X)-u(FX))/2,

A=C=0  <=>  u(X)=0 und u(FX)=0.
```

Die starke Familie ist also

```text
GS4 ∩ F(GS4).
```

Auch die vorgeschlagene Split-Energie vereinfacht sich vollständig. Mit
`A2=2A` und `C2=2C` gilt per Parallelogrammidentität

```text
||A2||² + ||C2||²
  = ||u(X)+u(FX)||² + ||u(X)-u(FX)||²
  = 2 (||u(X)||² + ||u(FX)||²).
```

`Q_split` ist damit nicht ein dritter versteckter Indikator neben `Q`, sondern
die Summe zweier gewöhnlicher GS4-Ziele auf durch `F` gekoppelten Zuständen.
Das kann algorithmisch trotzdem nützlich sein, ist aber eine wichtige
Begrenzung der Interpretation.

## 2. Unabhängige Enumeration `n=2..6`

Der Prüfer
[`strong_split_audit.py`](strong_split_audit.py) fixiert die erste Spalte auf
`(+,+,+,+)` und entfernt damit exakt die vier globalen Folgenvorzeichen. Er
rechnet ausschließlich ganzzahlig mit `2A` und `2C`.

| `n` | normierte Zustände | GS4 | `A=C=0` | davon basisgemischt |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 16 | 16 | 8 |
| 3 | 256 | 108 | 96 | 72 |
| 4 | 4.096 | 1.536 | 1.248 | 1.008 |
| 5 | 65.536 | 7.500 | 3.360 | 2.880 |
| 6 | 1.048.576 | 105.408 | 49.920 | 45.696 |

Diese Zahlen reproduzieren den gemeldeten Befund exakt. Für jeden gefundenen
gemischten Witness wurden frisch geprüft:

- `2u(X)=2A+2C`;
- `2u(FX)=2A-2C`;
- Trackerenergie von `X` und `F(X)` ist null;
- Buildermatrix von `X` und `F(X)` ist vollständig orthogonal.

Damit ist ab `n=3` sowohl die echte Einschränkung als auch die Existenz vieler
gemischter Lösungen belegt. Die Enumeration beweist keine Fortsetzung für alle
Längen.

## 3. Frischer Audit archivierter Solverlösungen

Alle gespeicherten Lösungen wurden aus `seqs_b64` neu dekodiert und nicht über
ihre gespeicherte Energie beurteilt. Für jede wurden `A`, `C`, `u(X)` und
`u(FX)` neu berechnet und gegen einen frischen `Tracker` geprüft.

- Paley `n=52,54,64`: jeweils stark (`A=C=0`).
- Freie GS4-Lösungen: `0/233` bei `n=44`, `0/13` bei `n=46`, `0/62` bei
  `n=48`, `0/24` bei `n=50` und `0/19` bei `n=52` stark.
- Auch keine der archivierten Symbolsolver-Lösungen bei `n=28,30,32` war
  stark.

Das bestätigt: Die starke Familie enthält die direkte Paley-Unterfamilie,
erklärt aber nicht die bisher beobachteten freien großen Lösungen.

## 4. Einordnung gegenüber bekannten Räumen

### Nega-Williamson / negakomplementäre Vierer

Bright, Kotsireas und Ganesh nennen ein binäres Vierertupel mit verschwindender
summierter negaperiodischer Autokorrelation eine *nega Williamson sequence*;
Symmetrie wird dafür ausdrücklich nicht verlangt. In dieser bereits bekannten
Sprache lautet der neue Befund schlicht:

> `X` und sein Tripleprodukt-Dual `F(X)` sind beide nega-Williamson-Tupel.

Quelle: [Bright, Kotsireas, Ganesh, IEEE TIT 2020](https://doi.org/10.1109/TIT.2020.3016510),
insbesondere die Definitionen der periodischen und odd-periodic correlation
sowie der nega-Williamson-Sequenzen.

### Base-/T-Sequences und Paley/Golay

Bei der vorhandenen Base-/T-Einbettung liegt jede H4-Spalte in derselben Basis.
Auch Tupel `(a,b,a,b)` haben `beta_j=+1` an jeder Spalte. Dann ist `F(X)=X`
und die zweite Bedingung redundant. Deshalb liegen diese konkreten klassischen
Einbettungen innerhalb der starken Familie.

Die gemischten starken Witnesses zeigen umgekehrt, dass `A=C=0` nicht mit
dieser achsenreinen Repo-Parametrisierung identisch ist. Eine Äquivalenz unter
größeren Hadamard- oder Sequenzsymmetrien ist dadurch nicht ausgeschlossen.

### Quaternionen, Williamson-type und relative difference sets

Die 16 H4-Symbole sind, bis auf eine feste Koordinatenkonvention, dasselbe
Alphabet

```text
Q+ = Q8 ∪ q Q8,  q=(1+i+j+k)/2,
```

das in der Quaternionenliteratur verwendet wird. Diese Übereinstimmung der
Koordinate ist also bekannt. Perfekte Quaternionfolgen verlangen jedoch außer
der skalaren Autokorrelationsbedingung noch drei Kreuzkorrelationsbedingungen.
Genau diese zusätzlichen Bedingungen führen zu QT-/Williamson-type-Sequenzen
und zu den bekannten relative-difference-set-Korrespondenzen.

Die starke Zerlegung erzwingt diese drei imaginären Bedingungen nicht. Schon
die frischen gemischten Witnesses für `n=2..6` haben nichtverschwindende
imaginäre Quaternion-Autokorrelation. Daher ist die starke Familie im
Allgemeinen weder QT/Williamson-type noch die bekannte relative-difference-set-
Familie.

Primärquellen:

- [Barrera Acevedo und Dietrich 2018](https://doi.org/10.1007/s12095-017-0224-y)
  für `Q+` und die relative-difference-set-Korrespondenz;
- [Quaternionic Perfect Sequences and Hadamard Matrices](https://cs.curtisbright.com/reports/perf-quat.pdf)
  für die explizite Trennung von Autokorrelations- und drei
  Kreuzkorrelationsbedingungen sowie QT/Williamson-type.

### Supplementary difference families

Jedes GS4-Tupel besitzt bereits die übliche Übersetzung in eine
negazyklische supplementary-difference-family-Sprache. Die starke Bedingung
liefert entsprechend zwei durch `F` gekoppelte solche Objekte. Im geprüften
Material fand sich kein Primärquellenbeleg dafür, dass genau diese
Tripleprodukt-Involution als eigene Standardfamilie benannt ist. Das ist kein
Neuheitsbeweis, sondern nur der aktuelle negative Literaturbefund.

## 5. Endurteil

Belastbar ist:

1. `A=C=0` ist exakt hinreichend und strikt stärker als GS4.
2. Die Familie ist klein-enumerativ groß und enthält gemischte Basiswörter.
3. Sie ist exakt die doppelte GS4-Bedingung für `X` und `F(X)`.
4. Sie enthält die konkreten Base/T- und `(a,b,a,b)`-Unterfamilien, aber nicht
   die archivierten freien großen Lösungen.
5. Sie ist nicht automatisch eine perfekte Quaternion-, Williamson-type- oder
   relative-difference-set-Konstruktion.

Noch nicht belastbar ist:

- wissenschaftliche Neuheit der Tripleprodukt-Dualbedingung;
- eine allgemeine Existenz- oder Produktkonstruktion;
- ein Vorteil gegenüber der freien GS4-Suche bei gleichem Candidate-Budget;
- eine neue Hadamard-Äquivalenzklasse.

Die sinnvollste Bezeichnung im Projekt ist daher vorerst:

> **dual geschlossene nega-Williamson-/GS4-Unterfamilie unter der
> Tripleprodukt-Involution `F`.**
