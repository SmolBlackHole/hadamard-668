# Gemeinsamer Konstruktionsraum: Paley, GS4, Turyn, Halb-Faltung und Tensor

Stand: 2026-08-09

Dieser Bericht trennt drei Kategorien:

- **Literatur:** bereits bekannte Konstruktion oder Satz;
- **exakte Projektfolgerung:** algebraisch abgeleitet und gegen `Tracker`,
  `Builder` und `independent_audit` geprüft;
- **Hypothese:** ein sinnvoller Versuch, aber noch kein Satz und kein gemessener
  Solvergewinn.

Der reproduzierbare Prüfer ist:

```powershell
python -m experiments.construction_space.verify_identities
```

## 1. Der gemeinsame Kern ist eine Polynorm

Für eine Folge `x` der Länge `n` schreibe

```text
X(z) = sum_j x_j z^j,
X*(z) = X(z^-1).
```

Vier Folgen sind genau dann eine GS4-Lösung des Projekts, wenn

```text
sum_s X_s(z) X_s*(z) = 4n  mod (z^n + 1).
```

Das ist dieselbe Aussage wie `sum_s NAF_s(t)=0`. Paley/Ito konstruiert
Nullstellen dieser Norm algebraisch; der Solver versucht dieselbe Norm durch
Bitflips konstant zu machen; die Halb-Längen-Faltung diagonalisiert dieselbe
Quotientenring-Norm in einer komplexen zyklischen Koordinate.

## 2. Ein NG-Paar liefert zwei verschiedene Hadamard-Lifts

**Literatur.** Ist `(a,b)` ein negaperiodisches Golay-Paar der Länge `n` und
`A,B` sind seine negazyklischen Matrizen, dann ist

```text
K(a,b) = [ A    B  ]
         [-B^T A^T]
```

eine 2N-Hadamardmatrix der Ordnung `2n`. Dies ist Proposition 2 bei
[Balonin und Djokovic](https://arxiv.org/abs/1508.00640).

**Exakte Projektfolgerung.** Der vorhandene Builder benutzt stattdessen
`(a,b,a,b)` und erzeugt eine GS4-Hadamardmatrix der Ordnung `4n`. Damit gibt es
aus demselben Paar zwei Lifts:

```text
(a,b) -> K(a,b)                         Ordnung 2n
(a,b) -> GS4(a,b,a,b)                  Ordnung 4n
(a,b) -> H2 tensor K(a,b)              Ordnung 4n
```

Die letzten beiden Matrizen sind im Allgemeinen **nicht äquivalent**. Exakte
Smith-Normalformen unterscheiden sich bereits für die zweiten Paley-Paare bei
`n=4` und `n=12`. Somit ist `GS4(a,b,a,b)` nicht bloß eine versteckte
Sylvester-Verdopplung von `K(a,b)`.

Alle 2N-, GS4- und Tensor-Matrizen in diesen Tests bestanden den unabhängigen
Integer-Audit. Die Smith-Normalform ist unter signierten Zeilen- und
Spaltenpermutationen invariant, daher sind die beiden Gegenbeispiele beweisend.

## 3. Die wichtige Komposition ist Turyn auf Sequenzebene

### 3.1 Bekannter Satz für Paare

**Literatur.** Für ein gewöhnliches aperiodisches Golay-Paar `(a,b)` der Länge
`g` und ein NG-Paar `(c,d)` der Länge `v` geben die Formeln

```text
e(z) = (a+b)/2 c(z^g) + (a-b)/2 d(z^-g) z^(gv-g)
f(z) = (b-a)/2 c(z^-g) z^(gv-g) + (a+b)/2 d(z^g)
```

ein NG-Paar der Länge `gv`. Dies sind die Gleichungen T1/T2 in Balonin und
Djokovic; sie erweitern Turyns Golay-Multiplikation.

Für das Golay-Paar `a=(+,-)`, `b=(+,+)` wird daraus besonders einfach:

```text
e[2j]   =  c[j]       e[2j+1] = -d[v-1-j]
f[2j]   =  d[j]       f[2j+1] =  c[v-1-j].
```

### 3.2 Exakte Erweiterung auf jedes GS4-Tupel

**Exakte Projektfolgerung.** Wende dieselbe Turyn-Abbildung getrennt auf
`(x1,x2)` und `(x3,x4)` an. Aus jedem binären GS4-Tupel der Länge `n` entsteht
ein binäres GS4-Tupel der Länge `gn`.

Es ist **nicht** nötig, dass `(x1,x2)` oder `(x3,x4)` für sich komplementär
sind. Die im Originalbeweis verwendete Identität gilt vor der
Komplementaritätsannahme:

```text
N(T_g(c,d)) = g (N(c)+N(d)) evaluated at z^g.
```

Nach Addition beider Paare folgt unmittelbar

```text
sum N(output_s)(z) = g sum N(input_s)(z^g)
                    = 4gn mod (z^(gn)+1).
```

Dies wurde auch mit allgemeinen, nicht verdoppelten Solverlösungen geprüft;
deren einzelne Paare hatten nichtverschwindende Residuen, das Ausgangs-4-Tupel
und der Lift waren dennoch exakt.

Für einen nicht exakten Ausgangszustand gilt zusätzlich:

```text
r'_k = 0             falls g nicht k teilt,
r'_(g t) = g r_t,
Q' = g^2 Q.
```

Der Lift macht das Residuum also nicht nur berechenbar, sondern spektral dünn.

### 3.3 Konkrete neue Abdeckung im Projekt

Alle folgenden Matrizen bestanden `Tracker`, `Builder` und
`independent_audit`:

| Ziel-n | Basis | Weg | Matrix |
| ---: | ---: | --- | ---: |
| 50 | Solver `n=25`, Seed 0 | Turyn mal 2 | H200 |
| 56 | Solver `n=28`, Seed 0 | Turyn mal 2 | H224 |
| 56 | Paley-Primzahlpotenz `n=14`, `q=27` | Turyn mal 4 | H224 |
| 58 | Solver `n=29`, Seed 0 | Turyn mal 2 | H232 |
| 60 | aktuelles Paley `n=30`, `q=59` | Turyn mal 2 | H240 |
| 62 | Solver `n=31`, Seed 0 | Turyn mal 2 | H248 |

Damit ist der bisher offene Block `56,58,60,62` konstruktiv erreichbar. Bei
`58` und `62` ist die Basis eine allgemeine GS4-Solverlösung, kein NG-Paar.
Der Solver muss also nicht mehr im größeren Raum beginnen: Er löst die kleinen
Basen `28,29,31` und der Lift erledigt den Rest exakt.

### 3.4 Paley/Ito plus Golay bis n=167

Die zweite Paley-Reihe besitzt theoretisch Basen für jedes gerade `n`, für das
`q=2n-1` eine Primzahlpotenz ist. Bis `167` sind die echten, vom aktuellen
Primzahlcode noch nicht unterstützten Primzahlpotenzfälle `n=14` (`q=27`) und
`n=122` (`q=243`).

Mit binären Golay-Längen und Turyn-Abschluss erhält man bis `n=167` insgesamt:

```text
2,4,6,8,10,12,14,16,20,22,24,28,30,32,34,36,40,42,44,48,
52,54,56,60,64,66,68,70,72,76,80,82,84,88,90,96,100,104,
106,108,112,114,120,122,126,128,132,136,140,142,144,152,
154,156,160,164,166.
```

Gegenüber den Paley-Basen selbst kommen durch den Produktabschluss neu hinzu:

```text
8,20,28,32,44,48,56,60,68,72,80,88,104,108,128,140,144,
152,160,164.
```

Das ist Literaturabschluss, kein neuer Existenzsatz. Neu für das Projekt ist,
dass derselbe Mechanismus direkt als Generatorstrategie und als
kleiner-Basis-zuerst-Dispatcher verwendbar ist.

`n=167` profitiert davon nicht: Es ist prim, `2n-1=333` ist keine
Primzahlpotenz und es gibt keinen nichttrivialen Golay-Faktor des Ziel-n.

## 4. Verhältnis zur komplexen Halb-Längen-Faltung

**Exakte Projektidentität.** Für gerades `n=2h` ist die vorhandene Faltung

```text
q_j = x_j + i x_(j+h),
y_j = exp(i*pi*j/n) q_j
```

eine verlustfreie Koordinatentransformation. Sie erfüllt

```text
C_t = exp(-i*pi*t/n) (r_t + i r_(h-t)).
```

Turyn und die Halb-Faltung arbeiten daher nicht an zwei verschiedenen
Problemen. Turyn faktorisiert die Polynorm; die Halb-Faltung zeigt dieselbe
Norm als periodische QPSK-Komplementarität. Der naheliegende gemeinsame Rahmen
ist eine paraunitäre Faktorisierung über dem Quotientenring `z^n+1`.

Das ist eine mathematische Vereinheitlichung, aber noch kein neuer allgemeiner
Konstruktionssatz über die Literatur hinaus.

## 5. Was der Repo-Tensorpfad leistet und was nicht

Der aktuelle `tensor`-Pfad löst zwei unabhängige GS4-Probleme und berechnet
danach

```text
H = H1 tensor H2.
```

Er gibt keine vier Folgen der Länge `4*n1*n2` zurück und kann deshalb weder vom
Tracker weiterbearbeitet noch als GS4-Warmstart benutzt werden.

Die offensichtliche Sequenzvermutung

```text
z_s = x_s tensor y_s
```

ist falsch: Paley `n=4` mal Paley `n=6` ergibt bei Länge 24 `Q=80` statt null.
Auch das kartesische 16-Tupel ist im negaperiodischen Quotienten im Allgemeinen
nicht komplementär. Der Grund ist, dass zwei Kongruenzen modulo `z^n+1` und
`z^m+1` nicht wie eine aperiodische Golay-Identität multipliziert werden
können.

Ob **jede volle Matrix** `H_GS4(x) tensor H_GS4(y)` durch beliebige signierte
Zeilen- und Spaltenpermutationen irgendeine GS4-Normalform besitzt, ist hier
nicht bewiesen und sollte nicht behauptet werden. Bekannte Produktresultate,
etwa [Yang 1979](https://doi.org/10.1090/S0025-5718-1979-0525685-8), benötigen
zusätzliche Operandenstruktur wie Golay-Paare, Williamson-Tupel oder
Four-Symbol-Delta-Codes. Sie liefern keinen allgemeinen Satz
`GS4 mal GS4 -> GS4`.

Der Spezialfall `(a,b,a,b)` liefert sogar eine konkrete Warnung: Sein GS4-Lift
ist im Allgemeinen nicht einmal zur naheliegenden Matrixverdopplung
`H2 tensor K(a,b)` äquivalent.

## 6. Schrumpfen und Reparieren

Das Löschen eines Folgenbits wirkt in der Kantenkoordinate überraschend lokal:
Zwei benachbarte Kanten werden zu ihrem Produkt verschmolzen. In den
NAF-Residualen ist derselbe Schritt jedoch global, weil sich die Länge, alle
Fenster und der Modulus von `z^n+1` ändern.

Für gemeinsames Schrumpfen der Paley-Lösung `n=52` ergab die vollständige
Spaltensuche:

```text
52 -> 51, eine gemeinsame Spalte: bestes Q = 97
52 -> 50, zwei gemeinsame Spalten: bestes Q = 48
```

Bei `(a,b,a,b)` bleibt gemeinsames Schrumpfen in der verdoppelten Paarform. Für
Ziel `n=51` kann sie nicht exakt sein, weil binäre NG-Paare außer Länge 1 nur
gerade Länge besitzen. Ein Solver müsste zuerst diese Symmetrie verlassen.

Schrumpfen ist deshalb weiterhin ein legitimer Warmstart, aber keine
restriktionsstabile Konstruktion. Der Turyn-Weg ist für zusammengesetzte
Zielgrößen klarer: kleinere Basis lösen und exakt vergrößern, statt eine große
Lösung zu verkleinern und einen dichten Defekt zu reparieren.

## 7. Konkrete Solver-Seeds und Moves

Aus den exakten Identitäten folgen fünf umsetzbare Experimente:

1. **Konstruktionsdispatcher:** Paley/Primzahlpotenz, dann Turyn-Abschluss,
   erst danach freie Suche.
2. **Faktor-Solver:** Für Ziel `N` jeden binären Golay-Teiler `g` prüfen, GS4
   nur bei `N/g` lösen und exakt liften.
3. **Lift-and-repair:** Einen guten, noch nicht exakten Basiszustand liften.
   Das Zielresiduum liegt dann nur auf Lags, die durch `g` teilbar sind.
4. **Turyn-Makromoves:** Bei `g=2` entspricht ein Basisbitflip genau zwei
   strukturierten Zielbitflips. So kann der Solver im eingebetteten kleineren
   Raum navigieren und gelegentlich freie GS4-Moves zulassen.
5. **Familien-Basin-Test:** Paley-, Golay-Fold-, allgemeine Solver- und
   Turyn-Lift-Lösungen gleich stark beschädigen und Recovery messen.

Die Punkte 3 bis 5 sind Hypothesen über bessere Suchdynamik. Die Konstruktionen
und Residualidentitäten selbst sind exakt.

## 8. TT(56), Base Sequences und die Ordnung 668

### 8.1 Literaturstatus und korrekte Größen

Eine Turyn-Type-Sequence `TT(n)` besteht aus binären Folgen

```text
L = (n,n,n,n-1),      w = (1,1,2,2)
```

mit der exakten aperiodischen Normgleichung

```text
N(A)+N(B)+2N(C)+2N(D) = 6n-2.
```

Dies ist die Definition bei
[Best, Djokovic, Kharaghani und Ramp](https://arxiv.org/abs/1206.4107).
Dasselbe Paper nennt ausdrücklich

```text
TT(56) -> BS(111,56) -> Hadamardmatrix H668.
```

Die Zahlen im untersuchten Anhang sind korrekt:

| Raum | rohe Bits | nichttriviale Lag-Koordinaten |
| --- | ---: | ---: |
| GS4 `n=167` | 668 | 83 |
| `BS(84,83)` | 334 | 83 |
| `TT(56)` | 223 | 55 |

„Koordinaten“ bedeutet hier nicht 55 beziehungsweise 83 unabhängige lineare
Constraints; die Gleichungen sind gekoppelt und quadratisch. Vier globale
Folgenvorzeichen ändern keine Autokorrelation, sodass die jeweiligen
Vorzeichenquotienten vier binäre Freiheitsgrade weniger besitzen.

Der aktuelle Literaturstand ist: `TT(n)` ist für alle geraden `n<=44` bekannt;
die ersten Beispiele für `40,42,44` wurden 2025 von
[London und Kotsireas](https://doi.org/10.1007/s12095-025-00829-z)
konstruiert. `TT(56)` ist damit eine echte offene Existenzfrage. `TT(n)`
existiert für ungerades `n>1` nicht.

### 8.2 Exakte TT-zu-BS-zu-GS4-Einbettung

**Literatur.** Aus `TT(n)=(A;B;C;D)` entstehen Base Sequences durch

```text
E = C || D,
F = C || -D,
(E;F;A;B) in BS(2n-1,n).
```

Die Kreuzterme der beiden Konkatenationen löschen sich exakt:

```text
N(E)+N(F) = 2N(C)+2N(D).
```

Aus beliebigen BS-Parametern `(A;B;C;D)` der Längen `(m,m,n,n)` definiere die
vier disjunkten ternären T-Sequenzen der Länge `L=m+n`:

```text
T1 = ((A+B)/2) || 0_n
T2 = ((A-B)/2) || 0_n
T3 = 0_m || ((C+D)/2)
T4 = 0_m || ((C-D)/2).
```

Eine Hadamard-4-Mischung der `T_i` liefert vier binäre Folgen `X_i` der Länge
`L`. Für ihre kombinierten aperiodischen Residuen gilt

```text
R_X(t) = 2 R_BS(t),
```

und daher für das negaperiodische GS4-Residuum

```text
r_GS4(t) = 2 (R_BS(t)-R_BS(L-t)).
```

**Exakte Projektfolgerung.** Für `BS(n+1,n)` verschwindet der Spiegelterm auf
allen unabhängigen GS4-Lags. Deshalb gilt für jede Belegung, nicht nur am Ziel:

```text
u_GS4 = R_BS/2,
Q_GS4 = ||R_BS/2||^2.
```

Für die TT-Einbettung verschwinden zusätzlich alle BS-Lags ab `n`, sodass

```text
u_GS4 = (R_TT/2, 0,...,0).
```

Bei `TT(56)` sind das 55 variable Residualkoordinaten gefolgt von 28
identischen Nullen im 83-dimensionalen `n=167`-Tracker. Der 223-Bit-TT-Raum
ist somit eine exakte parametrisierte Untermenge des vorhandenen
668-Bit-GS4-Raums, mit exakt derselben Zielfunktion.

Der kleine vollständige Test

```text
TT(2) -> BS(3,2) -> GS4(5) -> H20
```

bestand Tracker, Builder und `independent_audit`. Zufällige nichtgelöste BS-
und TT-Belegungen bis `n=7` bestätigten ebenfalls koordinatenweise die
Identität `u_GS4=R/2`.

Reproduktion:

```powershell
.\.venv\Scripts\python.exe -m experiments.construction_space.weighted_aperiodic_audit
```

### 8.3 Ein gemeinsamer gewichteter AF-Tracker ist exakt möglich

Für variable Längen `L_s` und Gewichte `w_s` setze

```text
R_t = sum_s w_s sum_j x_s[j] x_s[j+t].
```

Ein Flip `(s,c)` hat exakt das Delta

```text
d_(s,c,t) = -2 w_s x_s[c]
             ( [c+t<L_s] x_s[c+t] + [c-t>=0] x_s[c-t] ).
```

Damit können BS und TT denselben Residual-, Delta-, Greedy-, Tabu- und
Combo-Kern verwenden. In beiden Familien ist `R_t` gerade, also ist `u=R/2`
eine natürliche exakte Integerkoordinate. Es gelten weiterhin

```text
Q' = Q + 2<u,d> + ||d||^2,
sum_all_flips d = -4u,
direkter Solve-Flip <=> d=-u.
```

Die Balanceidentität gilt allgemein für gewichtete Summen quadratischer
Autokorrelationsterme, weil jedes Paar von seinen zwei Endpunktflips je einmal
mit Delta `-2` getroffen wird.

Der produktive `BaseTracker` speichert derzeit bewusst das rohe `R` und damit
`Q_raw=||R||²=4 Q_GS4`. Nullstellen und Flip-Ranking bleiben gleich; historische
GS4-Q-Schwellen müssen bei einer Solverintegration jedoch um diesen Faktor
angepasst werden.

### 8.4 Wichtige Grenze: GS4-Tight-Frame überträgt sich nicht

Der generische Tracker vereinheitlicht die Suchmechanik, aber nicht die gesamte
GS4-Geometrie. Im TT-Fall enthält `D^T D` quadratische Gewichte `w_s^2`, während
das Residuum lineare Gewichte `w_s` enthält. Außerdem entstehen aperiodische
Randterme.

Ein vollständiges Gegenbeispiel in TT-Form `(3,3,3,2)` wurde gefunden:

```text
gleiches Residuum r = (6,4),
D1^T D1 = [[176,48],[48,48]],
D2^T D2 = [[176,32],[32,48]].
```

Somit bestimmt `r` das lokale Move-Gram im TT-/BS-Raum nicht. Die bisherige
GS4-Aussage „Framepotential enthält keine Zusatzinformation neben u“ darf
nicht in den neuen Solver kopiert werden. Hier kann das Delta-Wörterbuch echte
zusätzliche Basin-Information tragen.

### 8.5 Exakte TT-Nebenbedingungen

Die höchste Laggleichung liefert sofort

```text
A_0 A_(n-1) = B_0 B_(n-1) = -C_0 C_(n-1).
```

Außerdem folgen durch Einsetzen von `z=1` und `z=-1` die notwendigen
quadratischen Summenbedingungen

```text
A(±1)^2+B(±1)^2+2C(±1)^2+2D(±1)^2 = 6n-2.
```

Diese Bedingungen sind billige exakte Seedfilter und eignen sich besser zur
TT-spezifischen Constraint-Propagation als ein vollständig zufälliger Start.
Für `n=56` bleiben nach Parität und Betragsgrenzen bei `z=1` beziehungsweise
`z=-1` jeweils nur 24 geordnete absolute Summenquadrupel `(A,B,C,D)`, oder 12
bis auf Vertauschung von `A` und `B`.

### 8.6 Verbindung und No-go-Grenzen

- **Initialkonstruktor:** Bekannte TT- und BS-Lösungen lassen sich exakt in
  das jetzige GS4-Format und den vorhandenen Builder überführen.
- **Restricted solver:** Ein TT-Single-Flip wird in der eingebetteten
  GS4-Darstellung zu einem strukturierten 2-Bit-Move für `A/B` oder 4-Bit-Move
  für `C/D`. Der gewichtete Tracker ist die kompakte Auswertung dieser Moves.
- **Lift-and-repair:** Ein guter TT-Kandidat liefert bei `n=167` einen Zustand
  mit 28 bereits exakt ruhigen Residualkoordinaten. Danach kann der freie GS4-
  Solver die TT-Unterfamilie verlassen. Das ist eine plausible, noch nicht
  gemessene Hybridstrategie und könnte auch dann H668 finden, wenn `TT(56)`
  selbst nicht existiert.
- **Paley/NG ist nicht aperiodisch:** Eine Paley-Lösung erfüllt eine Kongruenz
  modulo `z^n+1`, keine TT-/BS-Polynomidentität. Sie ist daher kein direkter
  TT- oder BS-Konstruktor.
- **Golay als TT-Teilpaar ist ausgeschlossen:** Wären `A,B` in einem TT(n)
  bereits ein gewöhnliches Golay-Paar, erzwingt der Lag `n-1` den unmöglichen
  Term `2 C_0 C_(n-1)=0`. Golay kann nur über speziellere bekannte
  Multiplikationskonstruktionen beitragen, nicht durch simples Einsetzen.
- **Multiplikative Lifts helfen bei 167 nicht:** Die Sequenzlänge 167 ist
  prim. Weder der zuvor geprüfte Turyn-Golay-Lift noch ein nichttriviales
  Kroneckerprodukt zerlegt dieses Ziel. Insbesondere kann H668 kein
  nichttriviales Tensorprodukt zweier Hadamardmatrizen sein: Faktoren größer
  als 2 hätten beide durch 4 teilbare Ordnung, ihr Produkt wäre durch 16
  teilbar, während `668 mod 16 = 12` gilt.

Die TT-/BS-Route ist damit keine Variante des Paley-Produkts. Sie ist eine
additive, stark strukturierte Parametrisierung genau des gesuchten
GS4(167)-Problems. Der sauberste erste Versuch ist ein gemeinsamer gewichteter
AF-Tracker mit zwei Modi (`BS(84,83)` und `TT(56)`), aber mit
familienabhängigen Propagationsregeln und ohne die GS4-Tight-Frame-Heuristiken
blind zu übernehmen.
