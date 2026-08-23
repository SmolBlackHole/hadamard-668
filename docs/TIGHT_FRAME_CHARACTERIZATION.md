# Tight-Frame-Charakterisierung des GS4-Solvers

Stand: 2026-08-09

Dieses Dokument ist die kanonische Darstellung des mathematischen
Hauptbefunds. Die Gesamtlandkarte steht in [`MATH_MAP.md`](MATH_MAP.md),
Suchfolgen und negative Ablationen in
[`SEARCH_FINDINGS.md`](SEARCH_FINDINGS.md).

## 1. Ausgangsproblem

Seien `x^(1),...,x^(4) in {+1,-1}^n`. Ihre antiperiodische Fortsetzung erfüllt

```text
x_bar[c+n] = -x_bar[c].
```

Definiere die negaperiodische Autokorrelation

```text
NAF_x(t) = sum_{c=0}^{n-1} x_bar[c] x_bar[c+t]
```

und das kombinierte Residuum

```text
r_t = sum_{s=1}^4 NAF_{x^(s)}(t).
```

Die vier Folgen erzeugen genau dann eine GS4-Hadamard-Matrix der Ordnung `4n`,
wenn alle nichttrivialen Residuen verschwinden.

Wegen

```text
r_{-t}=r_t,
r_{t+n}=-r_t
```

genügen

```text
m_eff = floor((n-1)/2)
```

unabhängige Lags. Der Mittelpunkt-Lag `n/2` ist bei geradem `n` identisch
null. Alle `r_t` sind durch vier teilbar. Setze

```text
u_t = r_t/4,
Q = ||u||_2^2.
```

Die im Repository verwendete volle Matrixenergie ist `E_repo=64nQ`.

## 2. Single-Flip-Operator

Für jeden der `4n` möglichen Bitflips sei `d_i` die Änderung des reduzierten
Residuals `u`. Stapelt man die Deltas als Zeilen, erhält man

```text
D in {-1,0,1}^{4n x m_eff}.
```

`D` ist zustandsabhängig: Nach jedem Flip ändern sich Folge, Residuum und
zukünftige Flip-Richtungen.

Für einen Single-Flip gilt exakt

```text
Q' = Q + 2 <d_i,u> + ||d_i||_2^2.
```

Für eine Folge `x` lautet der entsprechende Block:

```text
B(x)[c,t]
  = -(1/2) x_bar[c] (x_bar[c+t]+x_bar[c-t]).
```

Zusätzlich gilt die Balanceidentität

```text
sum_i d_i = -4u.
```

## 3. Grammatrixidentität

Erweitere `r` mit `r_0=4n`, `r_{-k}=r_k` und `r_{k+n}=-r_k`. Für zwei
unabhängige Lags `t,l` gilt:

```text
boxed:
(D^T D)_{t,l} = (r_{l-t}+r_{l+t})/2.
```

### Beweis

Aus der Deltaformel folgt

```text
(D^T D)_{t,l}
  = (1/4) sum_{s,c}
      (x[c+t]+x[c-t])
      (x[c+l]+x[c-l]).
```

Die vier ausmultiplizierten Summen werden durch Indexverschiebung in der
antiperiodischen Fortsetzung zweimal zu `r_{l-t}` und zweimal zu `r_{l+t}`.
Der Faktor `1/4` liefert die behauptete Formel.

Damit ist die komplette Spalten-Grammatrix des lokalen Move-Wörterbuchs bereits
durch das aktuelle Residuum bestimmt. Rang, Konditionierung und
Spaltenkohärenz liefern daher keine unabhängige lokale Information neben `u`.

Diese Aussage ist zugleich eine harte Grenze der Frame-Sicht. Bereits bei
`n=5` existieren zwei vollständig enumerierte Zustände mit identischem
`u=(0,1)` und identischer Matrix

```text
D^T D = [[12,-2],
         [-2,10]].
```

Der erste besitzt keinen unmittelbar lösenden Single-Flip, der zweite sechs.
Die lokale Erreichbarkeit steckt daher in der konkreten Zeilenanordnung und
Faktorisierung von `D`, nicht in `D^T D` oder einer daraus abgeleiteten
quadratischen Kennzahl. Das Gegenbeispiel ist als Regressionstest im Tracker
festgehalten.

## 4. Tight-Frame-Äquivalenz

Für `n>=5` gilt mit dem reduzierten Wörterbuch:

```text
boxed:
GS4-Hadamard
<=> u=0
<=> Q=0
<=> D^T D = 2n I_{m_eff}.
```

Ist `u=0`, verschwinden alle nichttrivialen Residuen. In der
Grammatrixidentität bleiben auf der Diagonale `r_0/2=2n` und außerhalb der
Diagonale null.

Die Rückrichtung folgt aus den Normidentitäten im nächsten Abschnitt: Ein
verschwindender Frame-Defekt erzwingt `Q=0`.

Am Ziel bilden die `4n` Zeilen von `D` somit ein balanciertes ternäres Tight
Frame des `m_eff`-dimensionalen Residualraums mit Framekonstante `2n`.

## 5. Exakte Normidentitäten

Definiere

```text
F = D^T D-2nI.
```

Für gerades `n>=6` gilt:

```text
boxed:
||F||_F^2 = 4(n-4)Q.
```

Für ungerades `n>=5` setze

```text
a = sum_{t=1}^{(n-1)/2} (-1)^t u_t.
```

Dann gilt:

```text
boxed:
||F||_F^2 = 4(n-4)Q + 8a^2.
```

Der zusätzliche nichtnegative Term ist der Defekt an der negazyklischen
Frequenz `z=-1`. Auch bei ungeradem `n` bleibt die Lösungsäquivalenz exakt.

Folgerung: Das explizite Framepotential ist keine neue effizientere
Zielfunktion. Für gerades `n` ist es nur eine skalierte Darstellung von `Q`;
für ungerades `n` kommt der bekannte Spektralterm hinzu.

## 6. Exakte Integrabilität

Ein beliebiges ternäres Tight Frame muss nicht aus vier binären Folgen stammen.
Für einen Folgenblock lässt sich diese Erzeugbarkeit jedoch exakt beschreiben.

Definiere das periodische Kantenwort

```text
p_c = x_bar[c] x_bar[c+1].
```

Dann gilt

```text
p_c in {+1,-1},
product_c p_c = -1.
```

Für das zyklische Intervallprodukt

```text
P_t(c)=product_{j=0}^{t-1}p_{c+j}
```

gilt

```text
boxed:
B(x)[c,t] = -(P_t(c)+P_t(c-t))/2.
```

Umgekehrt erzeugt jedes solche Kantenwort rekursiv eine antiperiodische Folge,
eindeutig bis auf ihr globales Vorzeichen.

Für die erste Spalte folgt

```text
B[c,1]=-(p_c+p_{c-1})/2.
```

Aus einer ternären ersten Spalte lassen sich höchstens zwei Kantenkandidaten
propagieren. Nach zyklischem Abschluss und Produktprüfung bestimmt jede gültige
erste Spalte den vollständigen Block eindeutig. Nur bei `n = 2 mod 4` fallen
die zwei alternierenden Kantenwörter auf denselben Block.

Ein markierter `n x m_eff`-Block ist daher in `O(n*m_eff)` exakt auf
Integrabilität prüfbar.

## 7. Was die Kantenkoordinate leistet

Ein Flip von `x_c` negiert genau die beiden benachbarten Kanten `p_{c-1}` und
`p_c`. Ein zusammenhängender Folgenintervallflip verändert nur seine beiden
Randkanten. Außerdem gilt

```text
r_t = sum_{s,c} P_t^(s)(c).
```

Eine Lösung balanciert auf jeder relevanten Fensterlänge exakt `2n` positive
und `2n` negative Intervallprodukte.

Die Kantenparametrisierung entfernt jedoch nur das globale Vorzeichen jeder
Folge:

```text
4n Folgenbits -> 4(n-1) unabhängige Kantenbits.
```

Das ist Faktor 16, keine asymptotische Suchraumreduktion. Der Wert der
Koordinate muss aus stärkeren Moves, Constraint-Propagation oder einer
konstruktiven Zerlegung entstehen.

## 8. Zyklischer Lift

Der exakte Lift

```text
x -> (x,-x)
```

überführt NAF in periodische Autokorrelation der Länge `2n`. In der DFT
verschwinden alle geraden Bins; die ungeraden Bins sind die Nullstellen von
`z^n+1`.

Dies erlaubt zyklische und spektrale Werkzeuge nur dann verlustfrei, wenn die
Antipodalbedingung erhalten bleibt. Eine freie zyklische Optimierung verlässt
im Allgemeinen den gültigen GS4-Unterraum.

## 9. Exakte komplexe Halb-Längen-Faltung für gerades n

Für `n=2h` zerlege `x=(a,b)` in zwei Hälften und setze

```text
q_j = a_j + i b_j,
omega = exp(i pi/(2h)),
y_j = omega^j q_j.
```

Ist `C_t` die über alle vier Folgen summierte periodische Autokorrelation der
`y`-Folgen, dann gilt exakt

```text
C_t = omega^(-t) (r_t + i r_{h-t}),   1 <= t < h.
```

Somit ist GS4 bei Länge `2h` äquivalent zu einer periodisch komplementären
Familie aus vier chirp-modulierten QPSK-Folgen der Länge `h`. Ferner gilt

```text
Q = (1/32) sum_{t=1}^{h-1} |C_t|².
```

Die Darstellung halbiert die Folgenlänge und öffnet exakt den zyklischen
FFT-/Filterbankraum. Sie entfernt keine binären Freiheitsgrade. Konstruktionen
stehen in [`CONSTRUCTION_SPACE.md`](CONSTRUCTION_SPACE.md), die Suchfolgen in
[`SEARCH_FINDINGS.md`](SEARCH_FINDINGS.md).

## 10. Verifikation

Geprüft wurden unter anderem:

- Grammatrix- und Balanceidentität auf zufälligen Zuständen bis `n=52`;
- gerade und ungerade Normidentität für jedes `n=5,...,53`;
- echte Lösungen und Nichtlösungen;
- Kantenrekonstruktion gegen den Tracker bis `n=53`;
- vollständige Blockenumeration bis `n=12`;
- vollständige ternäre Erste-Spalten-Prüfung bis `n=10`.

Die Identitäten sind als Regressionstests in `tests/` verankert. Historische
Wegwerfprüfer wurden nach Übernahme der Ergebnisse entfernt.

## 11. Konsequenz für den Solver

Der Solver navigiert nicht annähernd den Raum von `2^(4n)` Zuständen. Er senkt
einen ungefähr `n/2`-dimensionalen ganzzahligen Residualvektor und verändert
gleichzeitig dessen strukturiertes Move-Wörterbuch.

Mit fallendem `Q` wird das Wörterbuch automatisch nahezu tight. Der letzte
Engpass ist nicht fehlende lineare Information, sondern die diskrete
Erreichbarkeit eines Zustands, dessen zulässige Flipkombination das Residuum
annihiliert.

Die nächste relevante Frage ist deshalb nicht, wie man das gegenwärtige
Framepotential nochmals optimiert, sondern welcher kontrollierte Move ein
besser lösbares nächstes Wörterbuch erzeugt.

## 12. Präziser Status

Bewiesen und computerverifiziert ist eine exakte strukturelle Umformulierung:

> Eine GS4-Lösung ist genau ein binärer Zustand, dessen reduzierter
> Single-Flip-Delta-Operator ein balanciertes ternäres Tight Frame mit
> Framekonstante `2n` bildet; die integrierbaren Blöcke sind exakt durch
> odd-parity Kantenwörter charakterisiert.

Nicht bewiesen beziehungsweise nicht gefunden sind:

- eine neue Hadamard-Existenzklasse;
- ein polynomialer Konstruktionsalgorithmus;
- eine kleinere asymptotische Parametrisierung;
- ein Neuheitsnachweis gegenüber der gesamten Literatur;
- ein zuverlässiger Generator, der Integrabilität und Tightness automatisch
  gleichzeitig erzwingt.
