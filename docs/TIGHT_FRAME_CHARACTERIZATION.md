# Tight-Frame-Charakterisierung des GS4-Residualsolvers

Stand: 2026-08-07

## Status dieser Notiz

Diese Notiz dokumentiert eine exakte alternative Charakterisierung der im
Repository gesuchten GS4-Hadamard-Lösungen. Sie trennt ausdrücklich zwischen:

1. algebraisch bewiesenen Identitäten,
2. maschinell überprüften Konsequenzen,
3. empirischen Beobachtungen des Solvers,
4. offenen konstruktiven und wissenschaftlichen Fragen.

Das Ergebnis ist stärker als eine empirische Korrelation: Innerhalb des hier
verwendeten GS4-Modells ist die verschwindende Residualenergie äquivalent zu
einer Tight-Frame-Bedingung für den vollständigen Single-Flip-Operator.

Nicht geklärt ist, ob diese Charakterisierung in der mathematischen Literatur
bereits bekannt ist. Sie wird daher als neues Resultat für dieses Projekt, nicht
als beanspruchte wissenschaftliche Erstentdeckung, formuliert.

Die beiden zugrunde liegenden Untersuchungsprotokolle sind:

- [`research/2026-08-07-algebraic-frame-analysis.md`](research/2026-08-07-algebraic-frame-analysis.md)
  für Herleitung und algebraische Gegenchecks,
- [`research/2026-08-07-solver-trace-analysis.md`](research/2026-08-07-solver-trace-analysis.md)
  für Solverpfade, Ablationen und Endzustände.

## 1. Ausgangsproblem

Gesucht sind vier binäre Folgen

```text
x^(s) in {+1,-1}^n,    s=0,1,2,3,
```

deren vier negazirkulante Blöcke in der Goethals-Seidel-Konstruktion eine
Hadamard-Matrix der Ordnung `4n` erzeugen.

Die vollständige Matrix besitzt `16n^2` Einträge. Der Solver sucht jedoch nicht
direkt in diesem Matrixraum. Durch die GS4-Blockstruktur reduziert sich die
Orthogonalitätsbedingung auf einen kurzen negaperiodischen Residualvektor.

### 1.1 Antiperiodische Fortsetzung

Jede Folge wird auf ganzzahlige Indizes antiperiodisch fortgesetzt:

```text
x[j+n]  = -x[j],
x[j+2n] =  x[j].
```

Ihre negaperiodische Autokorrelation ist

```text
A_s(t) = sum_{j=0}^{n-1} x^(s)[j] x^(s)[j+t].
```

Der kombinierte Residualvektor lautet

```text
r_t = A_0(t) + A_1(t) + A_2(t) + A_3(t).
```

Die Fortsetzung liefert die Symmetrien

```text
r_{-t} = r_t,
r_{t+n} = -r_t,
r_0 = 4n.
```

Die GS4-Hadamard-Bedingung ist exakt

```text
r_t = 0    für alle nichttrivialen Lags t.
```

### 1.2 Unabhängige Koordinaten

Wegen der Symmetrien genügen

```text
m_eff = floor((n-1)/2)
```

unabhängige Lags.

Für ungerades `n` ist dies `floor(n/2)`. Für gerades `n` ist der Mittelpunkt
`t=n/2` identisch null:

```text
r_{n/2} = 0.
```

Diese Koordinate enthält keine Information. Der aktuelle Tracker speichert sie
bei geradem `n` noch, doch sowohl das Residuum als auch sämtliche Flip-Deltas
sind dort exakt null. Für alle Theoreme dieser Notiz wird sie entfernt.

### 1.3 Ganzzahlige Reduktion und Energie

Für vier binäre Folgen ist jeder kombinierte Residualwert durch vier teilbar.
Der Tracker verwendet daher

```text
u_t = r_t / 4,
Q   = sum_{t=1}^{m_eff} u_t^2.
```

Die öffentliche Repository-Energie ist

```text
E_repo = 64 n Q.
```

Damit gilt

```text
GS4-Hadamard-Lösung  <=>  u=0  <=>  Q=0.
```

Für `n=52` hat der Solver somit 208 binäre Variablen, aber nur 25 echte
Residualkoordinaten. Diese starke strukturelle Reduktion ist ein wesentlicher
Grund dafür, dass Millionen gezielter Bewertungen gegenüber dem nominellen
Zustandsraum von `2^208` wirksam sein können.

## 2. Der Single-Flip-Operator

Betrachtet wird ein Flip an Position `c` in Folge `s`:

```text
x^(s)[c] -> -x^(s)[c].
```

Sein reduziertes Residualdelta am Lag `t` ist

```text
d_{s,c,t}
  = Delta u_t
  = -(1/2) x^(s)[c] (x^(s)[c+t] + x^(s)[c-t]).
```

Da der Klammerausdruck nur `-2`, `0` oder `2` sein kann, gilt

```text
d_{s,c,t} in {-1,0,1}.
```

Stapelt man die Deltas aller `4n` möglichen Single-Flips als Zeilen, erhält
man die zustandsabhängige Matrix

```text
D in {-1,0,1}^{4n x m_eff}.
```

Die Zeile `d_i` ist die vollständige Residualwirkung des Flips `i`. `D` ist
damit gleichzeitig:

- der exakte Single-Flip-Jacobian des diskreten Zustands,
- das Move-Wörterbuch des Solvers,
- ein strukturiertes ternäres Frame im Residualraum.

Die Wirkung eines Single-Flips auf die Energie ist

```text
Q_i' = ||u+d_i||_2^2
     = Q + 2 u^T d_i + ||d_i||_2^2.
```

Genau diesen Ausdruck berechnet der Tracker mit `D @ u` und dem gespeicherten
Zeilennormvektor `norm2`.

## 3. Erste globale Identität: Summe aller Moves

Für jede einzelne Folge und jeden Lag wird jeder Korrelationssummand beim
Summieren der Flip-Deltas über alle Positionen zweimal getroffen. Daraus folgt

```text
sum_c d_{s,c,t} = -A_s(t).
```

Über alle vier Folgen ergibt sich die exakte Vektoridentität

```text
boxed: sum_i d_i = -r = -4u.
```

Der Zielvektor `-u` liegt also immer im positiven Kegel des aktuellen
Move-Wörterbuchs und besitzt stets die dichte fraktionale Darstellung

```text
-u = (1/4) sum_i d_i.
```

Dies garantiert keinen kleinen binären Flip-Subset. Mehrere Flips derselben
Folge besitzen zusätzliche quadratische Paarwechselwirkungen. Die Identität
zeigt aber exakt, warum der Solver als diskretes Vector-Balancing-Verfahren
interpretiert werden kann.

## 4. Hauptidentität für den Frame-Operator

### Satz 1: Grammatrix des Single-Flip-Operators

Für zwei unabhängige Lags `t` und `l` gilt

```text
boxed:
(D^T D)_{t,l} = (1/2) (r_{l-t} + r_{l+t}).
```

Dabei wird `r` mit

```text
r_{-k}=r_k,
r_{k+n}=-r_k
```

auf alle ganzzahligen Indizes fortgesetzt.

### Beweis

Aus der Deltaformel folgt

```text
(D^T D)_{t,l}
  = sum_{s,c} d_{s,c,t} d_{s,c,l}
  = (1/4) sum_{s,c}
      (x[c+t] + x[c-t])
      (x[c+l] + x[c-l]),
```

weil `x[c]^2=1`.

Die Ausmultiplikation liefert vier Korrelationssummen:

```text
sum_c x[c+t] x[c+l],
sum_c x[c+t] x[c-l],
sum_c x[c-t] x[c+l],
sum_c x[c-t] x[c-l].
```

Durch Indexverschiebung in der antiperiodischen Fortsetzung werden der erste
und vierte Term zu `A_s(l-t)`, der zweite und dritte zu `A_s(l+t)`. Nach der
Summe über die vier Folgen bleibt

```text
(D^T D)_{t,l}
  = (1/4) (2r_{l-t} + 2r_{l+t})
  = (1/2) (r_{l-t} + r_{l+t}).
```

Damit ist die gesamte Spalten-Grammatrix des lokalen Move-Wörterbuchs bereits
durch den aktuellen GS4-Residualvektor bestimmt.

## 5. Tight-Frame-Charakterisierung

### Satz 2: Äquivalenz zur GS4-Bedingung

Für `n>=5` und das um den geraden Mittelpunkt-Lag reduzierte Delta-Wörterbuch
gilt

```text
boxed:
GS4-Hadamard
  <=> u=0
  <=> Q=0
  <=> D^T D = 2n I_{m_eff}.
```

### Vorwärtsrichtung

Ist `u=0`, verschwinden alle nichttrivialen Residuen. Für `t=l` gilt mit
`r_0=4n`

```text
(D^T D)_{t,t}
  = (1/2) (r_0 + r_{2t})
  = 2n.
```

Für `t!=l` sind sowohl `l-t` als auch `l+t` nichttriviale Lags, sodass

```text
(D^T D)_{t,l}=0.
```

Damit sind die Spalten von `D` orthogonal und besitzen alle die quadrierte
Norm `2n`.

### Rückrichtung

Die Rückrichtung folgt aus den exakten Normidentitäten in Abschnitt 6. Ist
`D^T D=2nI`, verschwindet deren linke Seite. Sämtliche Terme der rechten Seite
sind nichtnegativ und der Koeffizient von `Q` ist für `n>=5` positiv. Also
muss `Q=0` gelten.

### Interpretation

Eine GS4-Lösung ist genau ein binärer Zustand, dessen gesamtes lokales
Single-Flip-Move-System isotrop wird. Die `4n` Zeilen von `D` bilden dann ein
Tight Frame des `m_eff`-dimensionalen Residualraums mit Framekonstante `2n`.

Diese Aussage ist nicht nur eine Korrelation. Sie ist dieselbe Ziellösung in
einer anderen mathematischen Sprache.

## 6. Exakte Normidentitäten

Definiere den Frame-Defekt

```text
F = D^T D - 2nI.
```

### 6.1 Gerades n

Für gerades `n>=6` gilt exakt

```text
boxed:
||F||_F^2 = 4(n-4)Q.
```

Äquivalent:

```text
boxed:
Q = ||D^T D-2nI||_F^2 / (4(n-4)).
```

Die bisherige Residualenergie und der Tight-Frame-Defekt sind damit exakt
dieselbe Zielfunktion in zwei Darstellungen.

### 6.2 Ungerades n

Für ungerades `n>=5` setze

```text
a = sum_{t=1}^{(n-1)/2} (-1)^t u_t.
```

Dann gilt

```text
boxed:
||F||_F^2
  = 4(n-4)Q + 8a^2.
```

Der Zusatzterm ist der Defekt an der reellen negazyklischen Frequenz `z=-1`:

```text
sum_s |X_s(-1)|^2 - 4n = 8a.
```

Für das eigentliche Ziel `H(668)` ist `n=167` ungerade. Dieser Zusatzterm ist
daher ein zentraler Bestandteil der Charakterisierung und keine technische
Randnotiz.

Auch für ungerades `n` bleibt die Lösungsäquivalenz vollständig erhalten:

```text
Q=0  <=>  D^T D=2nI.
```

Die einfache Proportionalität zwischen Frame-Defekt und `Q` wird lediglich um
den expliziten nichtnegativen Spektralterm ergänzt.

## 7. Konsequenzen am Lösungszustand

### 7.1 Balancierte ternäre Spalten

Jede Spalte von `D` enthält `4n` Werte aus `{-1,0,1}`. Am Lösungszustand gilt:

```text
Spaltensumme = 0,
Spaltennorm^2 = 2n.
```

Folglich enthält jede Spalte exakt

```text
n Einträge +1,
n Einträge -1,
2n Einträge 0.
```

### 7.2 Orthogonalität aller Lag-Richtungen

Verschiedene Residualkoordinaten sind im vollständigen Move-System orthogonal:

```text
sum_i d_{i,t} d_{i,l} = 0    für t!=l.
```

Die Lösung balanciert damit nicht nur die Autokorrelationen. Sie balanciert
gleichzeitig die Wirkungen sämtlicher möglicher Single-Flips auf alle
unabhängigen Lags.

### 7.3 Frame-Koeffizienten

Ist ein Frame `D` tight, gilt für jeden extern gewählten Vektor `v` im
Residualraum

```text
sum_i <d_i,v>^2 = v^T D^T D v = 2n ||v||_2^2.
```

Dabei ist Präzision notwendig: Am GS4-Lösungszustand ist dessen eigener
Residualvektor `u=0`. Für den Delta-Cache eines anderen `Q=1`-Zustands ist
`D^T D` im Allgemeinen nur nahe, aber nicht gleich `2nI`. Die Tight-Frame-
Identität darf daher nicht ohne den zustandsabhängigen Korrekturterm auf ein
beliebiges aktuelles `u` übertragen werden.

## 8. Kleine-Q-Geometrie

Weil `u` ganzzahlig ist, besitzen kleine Energien eine starre Form:

```text
Q=1: u = +/-e_t.
Q=2: genau zwei Koordinaten sind +/-1.
Q=3: genau drei Koordinaten sind +/-1.
```

Für `Q<=3` gelten daher bereits

```text
||u||_1 = support(u) = Q,
||u||_infinity = 1.
```

Diese Zusatzmetriken enthalten an der terminalen Barriere keine neue
Information gegenüber `Q`.

### 8.1 Exakte Zählung am Q=1-Zustand

Sei

```text
u = sigma e_t,    sigma in {+1,-1}.
```

Mit der antiperiodisch fortgesetzten Koordinate `u_ext(2t)` gilt für die
betroffene Spalte:

```text
sum_i d_{i,t}   = -4u_t,
sum_i d_{i,t}^2 = 2n + 2u_ext(2t).
```

Daraus folgen die Anzahlen

```text
#(+1) = n + u_ext(2t) - 2u_t,
#(-1) = n + u_ext(2t) + 2u_t.
```

Es existieren also viele Moves mit korrektem Vorzeichen am fehlerhaften Lag.
Das Problem ist der von denselben Moves erzeugte Kollateralschaden in den
anderen Lag-Koordinaten.

Für ein richtig ausgerichtetes Atom mit `d_{i,t}=-sigma` gilt exakt

```text
Q_i' = ||d_i||_2^2 - 1.
```

Der erste Tabu-Schritt kann das alte Einheitsresiduum löschen und gleichzeitig
mehrere neue Einheitsresiduen erzeugen. Tabu betreibt an dieser Wand daher
Residualtransport und verändert mit jedem Flip zugleich das Wörterbuch `D`.

## 9. Interpretation des aktuellen Solvers

### 9.1 Greedy Singles

Singles wählen einen strukturierten Vektor `d_i`, der den aktuellen Fehler
`u` möglichst stark reduziert. Das ähnelt Gauss-Southwell-Coordinate-Descent
oder Matching Pursuit, allerdings mit einem diskreten und nach jedem Flip
veränderten Wörterbuch.

### 9.2 Pair-Rescue

Pairs kombinieren zwei vielversprechende Single-Richtungen und berücksichtigen
Same-Sequence-Wechselwirkungen exakt. Ein unmittelbar kleineres `Q` muss aber
nicht zu einem später besser lösbaren Wörterbuch führen.

### 9.3 Tabu

Tabu akzeptiert bewusst schlechtere Single-Schritte. Dadurch transportiert es
Residuen und ändert die Zeilenstruktur von `D`, bis eine annihilierende
Kombination erreichbar wird. Seine Wirkung ist nicht die Verbesserung von Rang
oder Konditionierung; diese sind bei kleinen `Q` bereits nahezu optimal.

### 9.4 Kick

Kicks verlassen erfolglose niedrige Basins und erzeugen ein neues lokales
Move-Wörterbuch. Sie sind weniger zielgerichtet als Tabu, bleiben aber ein
Fallback, wenn der Walk keinen besseren Zustand findet.

### 9.5 Gemeinsames Arbeitsmodell

```text
vier binäre Folgen
    -> kurzer ganzzahliger Residualvektor u
    -> 4n strukturierte Single-Deltas bilden D
    -> Singles reduzieren u lokal
    -> Pairs testen kleine exakte Kombinationen
    -> Tabu verändert D durch kontrollierten Residualtransport
    -> Kick erzeugt ein neues Basin
    -> Q=0 genau dann, wenn D ein Tight Frame bildet.
```

## 10. Der exakte zyklische 2n-Lift

Für eine Folge `x` definiere

```text
y = (x,-x) in {+1,-1}^{2n}.
```

Dann gilt

```text
PAC_y(t) = 2 NAF_x(t)    für 0<t<n,
PAC_y(n) = -2n.
```

Für vier gelöste Folgen besitzt die summierte periodische Autokorrelation der
gelifteten Folgen somit die Form

```text
C(0) = 8n,
C(n) = -8n,
C(t) = 0    für alle übrigen t.
```

Im Frequenzraum folgt das Zweiniveauspektrum

```text
sum_s |FFT(y_s)[k]|^2 = 0     für gerade k,
                         16n  für ungerade k.
```

Die geraden Bins verschwinden bereits durch die Antiperiodizität. Die ungeraden
Bins sind genau die Auswertungen an den Nullstellen von `z^n+1`.

Damit sind die NAF-, Lift-, Spektral- und Frameformulierungen verschiedene
Darstellungen derselben Autokorrelationsstruktur. Eine freie gewöhnliche
zyklische Flat-Spectrum-Optimierung wäre jedoch nicht exakt: Sie müsste sowohl
die Antipodalbedingung als auch das `0/16n`-Zielspektrum erhalten.

## 11. Difference-Family-Sicht

Sei für jede geliftete Folge

```text
Y_s = {j in Z_(2n) : y_s[j] = -1}.
```

Wegen `y[j+n]=-y[j]` enthält `Y_s` genau einen Vertreter aus jedem
antipodalen Paar `{j,j+n}`. Die Folge ist damit eine antipodale Transversale.

Definiert man die geordneten Differenzmultiplizitäten

```text
N_s(t) = |Y_s intersect (Y_s-t)|,
```

so ist die GS4-Bedingung äquivalent zu einer konstanten summierten
Differenzmultiplizität außerhalb der beiden ausgezeichneten Differenzen `0`
und `n`.

Ein ursprünglicher Bitflip tauscht genau die beiden Vertreter eines
antipodalen Paars. Der Solver kann daher auch als lokale Suche nach vier
antipodalen Transversalen einer zyklischen relativen Difference Family gelesen
werden.

## 12. Empirische Befunde

### 12.1 Vollständiger n=52-Trace-Sweep

Es wurden 150 Seeds mit

```text
n=52,
12.000.000 Budgetschritten,
12 Workern
```

vollständig aufgezeichnet. Der Sweep benötigte etwa 335 Sekunden Wall-Clock
und löste 2/150 Runs.

Die globalen Endbestwerte waren

```text
Q=0:   2 Runs,
Q=1:  62 Runs,
Q=2:  86 Runs.
```

Damit erreichten alle 150 Runs den Bereich `Q<=2`. Der Solver beherrscht den
Abstieg in den niedrigen Residualraum bei dieser Größe zuverlässig. Die
verbleibende Seltenheit ist die diskrete Erreichbarkeit von null.

Die beiden Lösungen waren Seeds 93 und 140. Beide sprangen innerhalb eines
Tabu-Walks direkt von `Q=2` auf `Q=0`:

```text
Seed 93:
Q=2 bei Evaluation 3.655.826,
Q=0 bei Evaluation 16.694.750.

Seed 140:
Q=2 bei Evaluation   735.270,
Q=0 bei Evaluation 18.433.473.
```

Keine der beiden Erfolgstrajektorien besaß zuvor einen global besten
`Q=1`-Zustand.

Das erstmalige Erreichen von `Q<=2` war nicht prädiktiv:

```text
später gelöst:     Median etwa 2,20 Mio. echte Evaluationen,
später Q=1:        Median etwa 2,22 Mio.,
später Q=2:        Median etwa 2,74 Mio.
```

Die Verteilungen überlappten breit. Erfolg und Misserfolg unterscheiden sich
nicht wesentlich darin, wie schnell sie in die niedrige Region gelangen.

### 12.2 Exakte Hamming-Radius-2-Barriere

Ein gespeicherter n=52-Bestzustand besaß

```text
Q=2,
u_18=+1,
u_23=-1.
```

Sein Frame-Defekt erfüllte exakt

```text
||D^T D-104I||_F^2
  = 4(52-4)2
  = 384.
```

Die Konditionszahl von `D` betrug nur etwa 1,074. Dennoch ergab die vollständige
Nachbarschaftsauswertung:

```text
bester Single: Q=7,
bestes aller C(208,2)=21.528 Paare: Q=7,
kein neutraler Single,
kein verbessernder Single,
kein verbesserndes Paar.
```

Die lineare Geometrie war bereits nahezu perfekt. Der Zustand war trotzdem in
Hamming-Radius zwei strikt isoliert.

### 12.3 Q=1-Barrieren bei n=38

In einem 60-Seed-Lauf endeten alle 26 Fehlläufe beim besten gespeicherten
Zustand mit `Q=1`.

Für 13 reproduzierte Q=1-Endzustände wurde geprüft:

- kein Single-Escape,
- kein exakter Pair-Escape unter allen 11.476 Paaren,
- kein Escape mit höchstens einem Flip aus jeder der vier Folgen,
- bei zwei Zuständen auch kein vollständiger Triple-Escape einschließlich
  Same-Sequence-Korrekturen.

In detaillierteren Traces war die beobachtete Tabu-Erfolgsrate aus `Q=1` bei
`n=38` null. Tabu löste erfolgreiche Runs überwiegend aus reicheren Zuständen
mit `Q>=2`, bevor sie in der Q=1-Falle endeten.

### 12.4 Pair-Rescue als mögliche Fehlsteuerung

Eine erste gepaarte Konfigurationsstichprobe ergab:

| n | Pairs aktiv | Pairs aus | Zeitwirkung |
| ---: | ---: | ---: | --- |
| 38 | 34/60 | 49/60 | ohne Pairs etwas schneller |
| 40 | 6/40 | 9/40 | ohne Pairs deutlich schneller |

Dieser Befund ist noch kein belastbarer Defaultwechsel. Er stützt aber einen
plausiblen Mechanismus: Ein Pair kann `Q` monoton verbessern und gleichzeitig
den Zustand in eine diskret schlechter lösbare Q=1-Falle bringen, bevor Tabu
aus einem reicheren Q=2/3-Zustand einen direkten Lösungsweg findet.

Notwendig ist eine größere Ablation mit identischen Seeds, McNemar-Test und
expliziter Erfassung von `Q_vor`, `Q_nach` sowie der nachfolgenden Lösungsphase.

### 12.5 Nicht gefundene einfache Signaturen

Folgende Hypothesen wurden durch die bisherigen Daten nicht gestützt:

- eine feste Reihenfolge, in der bestimmte Lags verschwinden,
- ein robuster Startfilter aus initialem `Q`, Support oder Alignment,
- eine einzelne bevorzugte kleine Spektralpartition,
- individuelle NAF-Rauheit als Erklärung des letzten Erfolgs,
- Rang oder Konditionierung von `D` als unabhängige Escape-Metrik.

Bei Zufallszuständen war `D` in allen getesteten Größen bis n=52 vollrangig und
gut konditioniert. Rang und Spalten-Grammatrix enthalten wegen Satz 1 ohnehin
keine unabhängige Information neben `u`.

## 13. Vollständige und zufällige Computerprüfungen

Die algebraischen Identitäten wurden gegen den exakten Tracker geprüft:

- `sum_i d_i=-4u` für Zufallszustände bei n=5,6,7,8,30,35,38,52,
- die elementweise Formel für `D^T D` bei denselben Größen,
- die gerade und ungerade Normidentität für jedes n von 5 bis 53 mit jeweils
  25 Zufallszuständen,
- die tote Mittelpunktkoordinate bei allen getesteten geraden n,
- echte gespeicherte Lösungen von n=24 bis n=40,
- eine frisch erzeugte ungerade n=35-Lösung.

Für kleine Größen wurden außerdem alle binären Einzelfolgen bis n=14 nach
ihrer NAF-Signatur gruppiert und geordnete komplementäre Vierer gezählt. Die
Identitäten zeigten in den durchgeführten vollständigen und zufälligen Tests
keine Abweichung.

Diese Computerprüfungen ersetzen den symbolischen Beweis nicht. Sie sind ein
starker Schutz gegen Index-, Vorzeichen-, Mittelpunkt- und Gerade/Ungerade-
Fehler.

## 14. Was die Charakterisierung nicht leistet

### 14.1 Noch kein Neuheitsnachweis

Ohne systematische Literaturrecherche wird keine wissenschaftliche Neuheit
beansprucht. Verwandte Begriffe umfassen:

- Goethals-Seidel-Differenzfamilien,
- negaperiodische komplementäre Sequenzen,
- partielle Wiegematrizen,
- ternäre Tight Frames,
- Inzidenz- und Ableitungsmatrizen,
- diskrete Jacobians von Autokorrelationsabbildungen,
- relative Difference Families.

### 14.2 Noch kein billigeres Ziel

Das explizite Framepotential ist rechnerisch teurer als

```text
Q = ||u||_2^2.
```

Es wäre daher falsch, im Solver `Q` durch `||D^T D-2nI||_F^2` zu ersetzen.
Die Frame-Sicht ist strukturell wertvoll, aber derzeit kein schnelleres
Scoringverfahren.

### 14.3 Noch keine Konstruktion

Ein beliebiges ternäres Tight Frame mit `D^T D=2nI` genügt nicht. Ein gültiges
`D` muss aus vier binären antiperiodischen Folgen integrierbar sein. Seine
Zeilen und Spalten erfüllen besondere lokale Produkt-, Vorzeichen-,
Verschiebungs- und Konsistenzbedingungen.

Die zentrale konstruktive Frage lautet daher:

```text
Welche balancierten ternären Tight Frames D entstehen tatsächlich als
Single-Flip-Delta-Operatoren von vier binären GS4-Folgen?
```

Erst eine einfachere Charakterisierung dieser Integrabilität könnte den
Suchraum fundamental verkleinern.

## 15. Sichere unmittelbare Konsequenzen

### 15.1 Toten Mittelpunkt-Lag entfernen

Bei geradem `n` kann der Tracker beweisbar

```text
m_eff = (n-1)//2
```

anstelle von `n//2` verwenden. Dies entfernt eine Nullkoordinate aus `u`, `D`
und der Tabu-Schleife.

Die Lag-Arbeit sinkt ungefähr um

```text
n=38: 1/19 = 5,26%,
n=52: 1/26 = 3,85%.
```

Die Suchdynamik und Zielfunktion ändern sich nicht.

### 15.2 Frame-Identität als Debug-Invariante

Stichprobenweise kann

```text
D^T D = (1/2)(r_{l-t}+r_{l+t})
```

als unabhängiger Konsistenztest für den Delta-Cache dienen. Im Hot-Path wäre
die explizite Grammatrix zu teuer.

### 15.3 Trace-Daten fokussieren

Rang und vollständige Grammatrix müssen nicht gespeichert werden, weil sie aus
`u` rekonstruierbar sind. Wertvoll sind stattdessen:

- konkrete Sequenzen und `u`,
- Phase und echte Evaluierungszahl,
- kleinste Single-/Pair-/Subset-Distanz zu null,
- Kollateralnorm richtig ausgerichteter Atome,
- Wörterbuchänderung während erfolgreicher Tabu-Walks.

## 16. Weiterführende Arbeit

Offene Vermutungen, darunter die parallelen Forschungsstränge Neuheitsprüfung
und Konstruktion über integrierbare Flip-Frames, stehen in
[`HYPOTHESES.md`](HYPOTHESES.md). Die daraus abgeleiteten Experimente werden
ausschließlich in [`TODO.md`](TODO.md) gepflegt.

## 17. Präzise gegenwärtige Aussage

Der aktuelle Forschungsstand lässt sich wie folgt formulieren:

> Für vier binäre Folgen der Länge `n>=5` ist die GS4-Hadamard-Bedingung
> äquivalent dazu, dass der um abhängige Lag-Koordinaten reduzierte
> Single-Flip-Delta-Operator ein balanciertes ternäres Tight Frame mit
> Framekonstante `2n` bildet. Für gerades `n` ist der quadrierte
> Frobeniusabstand des Frame-Operators von `2nI` exakt `4(n-4)Q`; für
> ungerades `n` kommt ein expliziter nichtnegativer Defekt der Frequenz `z=-1`
> hinzu.

Dies ist eine exakte strukturelle Umformulierung des Problems und ein
ernstzunehmendes mathematisches Ergebnis für das Projekt. Wissenschaftliche
Neuheit und ein daraus folgender konstruktiver Vorteil sind noch offen.

## 18. Schlussfolgerung

Der Solver ist nicht erfolgreich, weil er einen Zustandsraum von `2^{4n}`
annähernd enumeriert. Er navigiert in einem nur ungefähr `n/2`-dimensionalen
ganzzahligen Residualraum und verändert zugleich das strukturierte
Move-Wörterbuch dieses Raums.

Mit fallendem `Q` wird dieses Wörterbuch automatisch nahezu isotrop. Der letzte
Engpass ist nicht mangelnde lineare Annäherung an die Lösung, sondern die
diskrete Integrabilität und Erreichbarkeit eines annihilierenden Flip-Sets.

Die Tight-Frame-Charakterisierung erklärt daher sowohl die Stärke als auch die
Grenze des aktuellen Solvers. Sie öffnet drei neue Wege: eine systematische
Neuheitsprüfung, eine mögliche Konstruktion über integrierbare Frames und
gezielte Low-Q-Escape-Operatoren für die nachgewiesenen diskreten Barrieren.
