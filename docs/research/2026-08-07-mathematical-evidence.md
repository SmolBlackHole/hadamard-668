# Mathematische Evidenz: Residual, Flip-Frame und Integrabilität

Stand: 2026-08-07

Dieses Dokument bewahrt die Herleitung und Computerverifikation hinter der
kanonischen Darstellung in `../TIGHT_FRAME_CHARACTERIZATION.md`. Es trennt
bewiesene Projektresultate von offenen Neuheits- und Konstruktionsfragen.

## 1. Exaktes GS4-Ziel

Für vier Folgen `x^(s) in {+1,-1}^n` sei

```text
r_t = sum_s NAF_{x^(s)}(t).
```

Mit der Symmetrie der negaperiodischen Autokorrelation reichen

```text
m_eff = floor((n-1)/2)
```

unabhängige Lags. Beim geraden `n` ist der Mittelpunkt-Lag `n/2` identisch
null. Alle `r_t` sind durch vier teilbar. Der Tracker verwendet daher

```text
u_t = r_t/4,
Q = ||u||_2^2,
E_repo = 64 n Q.
```

Eine GS4-Lösung erfüllt genau `u=0`.

## 2. Single-Flip-Wörterbuch

Für jeden der `4n` möglichen Bitflips sei `d_i` die Änderung von `u`. Die
Zeilenmatrix

```text
D = [d_1; ...; d_{4n}] in {-1,0,1}^{4n x m_eff}
```

ist das zustandsabhängige Move-Wörterbuch des Solvers. Für einen Flip gilt

```text
Q' = Q + 2 <d_i,u> + ||d_i||_2^2.
```

Außerdem gilt exakt

```text
sum_i d_i = -4u.
```

Das erklärt die Vector-Balancing- beziehungsweise Matching-Pursuit-Sicht,
garantiert aber kein kleines binäres Flip-Subset.

## 3. Grammatrix und Tight Frame

Erweitere das kombinierte Residuum durch

```text
r_{-k}=r_k,
r_{k+n}=-r_k,
r_0=4n.
```

Dann gilt für unabhängige Lags `t,l`:

```text
(D^T D)_{t,l} = (r_{l-t} + r_{l+t})/2.
```

Der Beweis folgt durch Ausmultiplizieren der beiden Terme des Flip-Deltas und
Indexverschiebungen in der antiperiodischen Fortsetzung.

Für `n>=5` folgt:

```text
GS4-Lösung
<=> u=0
<=> Q=0
<=> D^T D = 2n I.
```

Am Ziel bilden die `4n` Flip-Richtungen damit ein balanciertes ternäres Tight
Frame mit Framekonstante `2n`.

Definiere `F=D^T D-2nI`. Dann gilt:

```text
gerades n>=6:   ||F||_F^2 = 4(n-4)Q,

ungerades n>=5: ||F||_F^2 = 4(n-4)Q + 8a^2,
                 a = sum_t (-1)^t u_t.
```

Der Zusatzterm für ungerades `n` ist der Defekt an der negazyklischen
Frequenz `z=-1`. Die Lösungsäquivalenz bleibt erhalten.

Wichtig für den Solver: Das Framepotential ist keine neue billigere
Zielfunktion. Für gerades `n` ist es exakt proportional zu `Q`; für ungerades
`n` enthält es nur den expliziten zusätzlichen Spektralterm.

## 4. Exakte Blockintegrabilität

Für eine Folge `x` mit antiperiodischer Fortsetzung `x_bar` definiere das
periodische Kantenwort

```text
p_c = x_bar[c] x_bar[c+1],
product_c p_c = -1.
```

Mit

```text
P_t(c) = product_{j=0}^{t-1} p_{c+j}
```

lautet der vollständige reduzierte Flip-Block:

```text
B(x)[c,t] = -(P_t(c) + P_t(c-t))/2.
```

Umgekehrt erzeugt jedes periodische Vorzeichenwort mit Produkt `-1` eine
antiperiodische Folge bis auf ein globales Vorzeichen. Für die erste Spalte
gilt

```text
B[c,1] = -(p_c+p_{c-1})/2.
```

Aus ihr lassen sich höchstens zwei Kantenkandidaten propagieren. Nach Prüfung
des zyklischen Abschlusses und `product(p)=-1` bestimmt sie den gesamten Block
eindeutig. Beim Sonderfall `n = 2 mod 4` erzeugen die zwei alternierenden
Kantenwörter denselben Block.

Damit ist ein beliebiger markierter Block in `O(n*m_eff)` auf Integrabilität
prüfbar. Die Parametrisierung reduziert den Suchraum aber nur um das globale
Folgenvorzeichen: vier Folgen sparen insgesamt vier Bits, also Faktor 16.

## 5. Kantenraum-Interpretation

Ein Flip von `x_c` negiert genau `p_{c-1}` und `p_c`. Ein zusammenhängender
Intervallflip in `x` verändert deshalb nur seine beiden Randkanten. Außerdem
gilt

```text
r_t = sum_{s,c} P_t^(s)(c).
```

Eine Lösung balanciert für jede Fensterlänge `t` die `4n` Kantenintervalle:
genau `2n` Produkte sind positiv und `2n` negativ. Diese Darstellung ist eine
exakte neue Suchkoordinate, aber noch kein Konstruktionsalgorithmus.

## 6. Computerverifikation

Die Projektidentitäten wurden wie folgt geprüft:

- `sum_i d_i=-4u` bei zufälligen Zuständen bis `n=52`;
- elementweise Grammatrixformel bei geraden und ungeraden Größen;
- beide Frobeniusnormformeln für jedes `n=5,...,53`;
- echte gespeicherte Lösungen und zufällige Nichtlösungen;
- Kantenrekonstruktion gegen den Tracker für jedes `n=3,...,53`;
- vollständige binäre Enumeration der Blockintegrabilität bis `n=12`;
- vollständige ternäre Enumeration möglicher erster Spalten bis `n=10`.

Der ausführbare Dauerprüfer ist:

```powershell
python -m experiments.frame_integrability
```

Diese Prüfungen schützen gegen Implementierungsfehler, ersetzen aber nicht den
symbolischen Beweis.

## 7. Wissenschaftliche Einordnung

Negakomplementäre beziehungsweise nega-Williamson-Folgen, geeignete
negazyklische Matrizen und die GS4-Hadamard-Konstruktion sind bekannt. Für den
zustandsabhängigen Single-Flip-Operator, seine Grammatrixformel, die
Tight-Frame-Äquivalenz und die hier verwendete Blockintegrabilitätsform wurde
in der bisherigen Suche kein direkter Vorläufer gefunden.

Der korrekte Status lautet daher:

> Ernstzunehmendes neues mathematisches Resultat für dieses Projekt;
> wissenschaftliche Neuheit und konstruktiver Vorteil sind noch nicht
> abschließend geklärt.

Es wurde weder eine neue Existenzklasse noch ein polynomialer
Konstruktionsalgorithmus gefunden.
