# Strong split `A=C=0`

Stand: 2026-08-09

## Kurzfassung

Die exakte H4-Zweibasenkoordinate zerlegt das bekannte GS4-Residuum in einen
Within-Basis-Anteil `A` und einen Cross-Basis-Anteil `C`:

```text
u = A + C.
```

Die stärkere Bedingung

```text
A = 0  und  C = 0
```

ist daher hinreichend für GS4. Vollständige Enumeration bis `n=6` zeigt, dass
dies eine echte, große Unterfamilie und nicht nur eine Umbenennung der gesamten
GS4-Bedingung ist. Sie enthält außerdem viele Lösungen mit gemischter
Basiswahl. Ein einfacher experimenteller Best-Move-Tabu-Walk findet solche
Lösungen zuverlässig bis `n=16`, stößt aber ab `n=20..24` erneut an eine kleine
diskrete Restwand. Keine der 19 archivierten freien `n=52`-Lösungen erfüllt die
starke Zerlegung; die direkte Paley-Lösung erfüllt sie.

Der Fund ist damit ein konkreter, experimentell erzeugbarer Zwischenraum
zwischen bekannten stark strukturierten Konstruktionen und freiem GS4. Ob er
unter einer anderen Sprache bereits als Base-/T-Sequence-,
relative-difference-family- oder complementary-sequence-Konstruktion bekannt
ist, ist noch nicht unabhängig geklärt.

## 1. Exakte Definition

Schreibe die vier binären Folgen spaltenweise als

```text
x_j = (a_j, b_j, c_j, d_j)^T in {+-1}^4
```

und setze

```text
v_j    = H4 x_j / 4,
beta_j = a_j b_j c_j d_j.
```

Wie in `docs/CONSTRUCTION_SPACE.md` beschrieben, zeigt `beta_j`, welche der
beiden orthonormalen H4-Basen die Spalte `v_j` verwendet. Für einen reduzierten
Lag `t` wird die negazyklisch vorzeichenbehaftete Summe der Skalarprodukte nach
gleicher und verschiedener Basiswahl getrennt:

```text
A_t = sum_j sign(j,t) <v_j, v_(j+t)> * [beta_j =  beta_(j+t)]
C_t = sum_j sign(j,t) <v_j, v_(j+t)> * [beta_j != beta_(j+t)].
```

Dabei werden Indizes modulo `n` gelesen und `sign(j,t)` ist das übliche
negazyklische Wrap-around-Vorzeichen. Jeder Summand gehört exakt zu einer der
beiden Klassen. Deshalb gilt koordinatenweise

```text
u_t = A_t + C_t.
```

Für eine ganzzahlige Implementierung kann ohne Brüche mit

```text
A2_t = 2 A_t,
C2_t = 2 C_t
```

gerechnet werden. Aus den Rohspalten folgt dann direkt

```text
A2_t = sum_same  sign(j,t) * <x_j,x_(j+t)> / 2,
C2_t = sum_cross sign(j,t) * <x_j,x_(j+t)> / 2,
A2_t + C2_t = 2 u_t.
```

## 2. Exakte Suffizienz und echte Einschränkung

Aus `u=A+C` folgt sofort

```text
A=0 und C=0  =>  u=0  =>  GS4  =>  H(4n).
```

Die Umkehrung gilt nicht: Eine allgemeine GS4-Lösung darf `A=-C!=0` besitzen.
Damit ist die starke Zerlegung keine bloße Koordinatentransformation des
gesamten Lösungsraums, sondern eine echte hinreichende Zusatzbedingung.

Eine passende experimentelle Energie ist beispielsweise

```text
Q_split = ||A2||^2 + ||C2||^2.
```

Ihre Nullstellen sind exakt die stark zerlegten Lösungen. Außerhalb der
Nullstellen enthält sie bewusst mehr Information als `Q=||u||^2`, weil sie
die gegenseitige Aufhebung `A=-C` nicht als bereits gelöst behandelt.

## 3. Vollständige Enumeration `n=2..6`

Enumeriert wurden alle Zustände nach Entfernung der vier globalen
Folgenvorzeichen, also mit fest normierter erster Spalte. `mixed` bedeutet,
dass das Basiswort `beta` beide Vorzeichen enthält.

| `n` | alle GS4 | `A=C=0` | davon mixed | starke Basiswörter | alle GS4-Basiswörter |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 16 | 8 | 2 | 2 |
| 3 | 108 | 96 | 72 | 4 | 4 |
| 4 | 1.536 | 1.248 | 1.008 | 8 | 8 |
| 5 | 7.500 | 3.360 | 2.880 | 16 | 16 |
| 6 | 105.408 | 49.920 | 45.696 | 32 | 32 |

Zwei Aussagen sind dadurch exakt für diese Größen belegt:

1. Ab `n=3` ist `A=C=0` strikt stärker als GS4.
2. Die starke Familie ist nicht auf ein konstantes oder achsenreines
   Basiswort beschränkt. Bis `n=6` tritt jedes der `2^(n-1)` normalisierten
   Basiswörter sogar in mindestens einer starken Lösung auf.

Diese Enumeration beweist keine Fortsetzung für beliebiges `n`.

## 4. Wegwerf-Prototyp: Best-Move-Tabu

Der Prototyp verwendete keine Produktionslogik. Pro Schritt wurden alle `4n`
Single-Flips als NumPy-Batch erzeugt, `Q_split` für alle Kandidaten exakt
ausgerechnet und der beste nicht-tabu Move akzeptiert. Die Tabu-Tenure war 7;
es gab weder Kicks noch Pair-Rescue, Targeted Escape oder spezielle Seeds.

Ergebnis des ersten Piloten:

| `n` | Seeds | Budget pro Seed | gelöst | mixed unter den Lösungen |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 10 | 3.000 | 10 | 9 |
| 10 | 10 | 3.000 | 10 | 9 |
| 12 | 10 | 3.000 | 10 | 10 |
| 14 | 10 | 3.000 | 10 | 9 |
| 16 | 10 | 3.000 | 10 | 10 |
| 20 | 10 | 20.000 | 5 | 5 |
| 22 | 5 | 20.000 | 0 | 0 |
| 24 | 5 | 20.000 | 0 | 0 |

Die ungelösten Läufe endeten typischerweise bei `Q_split=4` oder `8`. Der
stärkere Raum beseitigt die bekannte diskrete Schlusswand daher nicht
automatisch; er verschiebt nur das Suchproblem in eine andere, enger
definierte Landschaft.

Eine reproduzierte gemischte `n=20`-Lösung hatte zunächst zehn Spalten in der
zweiten Basis. Im vollständig geprüften Orbit aus unabhängigen Negashifts und
Reversals der vier Folgen blieb mindestens eine Minderheitsbasis-Spalte übrig.
Sie ist damit innerhalb dieses konkreten Projektorbits nicht achsenrein. Das
ist weder ein Beweis einer neuen Hadamard-Äquivalenzklasse noch ein Ausschluss
umfangreicherer bekannter Äquivalenzen.

## 5. Verhältnis zu Paley und den freien Archivlösungen

- Die archivierte direkte Paley-Lösung bei `n=52` erfüllt `A=C=0`. Bei ihrer
  konstanten Basiswahl ist `C=0` automatisch und `A=u=0`.
- Keine der 19 archivierten freien `n=52`-GS4-Repräsentationen erfüllt die
  starke Bedingung. Für die ganzzahlige Größe `||A2||^2` lag der Bereich bei
  `216..440`; die Anzahl nichtverschwindender Koordinaten bei `16..23`.
- Auch unter den freien Archiven bei `n=44` und `n=50` wurde keine starke
  Lösung beobachtet. Die Bereiche für `||A2||^2` waren `72..588` und
  `180..520`.

Damit erklärt `A=C=0` nicht die bisher gefundenen freien großen Lösungen. Es
ist vielmehr eine zusätzliche konstruktive Unterfamilie, die Paley enthält
und bei kleinen `n` viele weitere, auch basisgemischte Repräsentationen
besitzt.

## 6. Was noch nicht behauptet werden darf

- Keine wissenschaftliche Neuheit ist belegt.
- `mixed` und der geprüfte Negashift-/Reverse-Orbit beweisen keine neue
  Hadamardklasse.
- Die kleinen Enumerationen beweisen weder Existenz noch einen einfachen
  Konstruktor für große `n`.
- Der naive Tabu-Pilot ist kein fairer Performancevergleich mit dem
  Produktionssolver; er demonstriert nur, dass der Raum algorithmisch
  zugänglich ist.
- Das Ausbleiben starker freier `n=52`-Archivlösungen kann sowohl echte
  Seltenheit als auch eine Folge der bisherigen Suchdynamik sein.
- `Q_split` kann Navigation verschlechtern, weil es zulässige Aufhebung
  zwischen `A` und `C` absichtlich bestraft.

## 7. Erforderlicher unabhängiger Audit

Vor jeder Produktionsintegration sind mindestens diese Punkte unabhängig zu
prüfen:

1. Herleitung von `u=A+C` einschließlich Skalierung, reduziertem Lagbereich
   und geradem Mittelpunkt-Lag.
2. Frischer Tracker- und Builder-Audit zufälliger sowie enumerierter
   `Q_split=0`-Zustände.
3. Vergleich mit Base Sequences, T-Sequences, TT, negazyklischen
   supplementary difference families und bekannten Zwei-Kanal-Zerlegungen.
4. Prüfung, ob die gemischten kleinen Lösungen unter einer größeren bekannten
   Äquivalenzgruppe doch achsenrein oder Paley-/Golay-artig werden.
5. Erst danach ein kandidatenbudgetgleicher Vergleich eines inkrementellen
   `Q_split`-Trackers gegen freien GS4 und strukturierte Starts.

Bis dieser Audit abgeschlossen ist, lautet der vorsichtige Status:

> Exakt hinreichende, empirisch große und gemischt realisierbare
> GS4-Unterfamilie; bekannte Einordnung und konstruktiver Vorteil offen.
