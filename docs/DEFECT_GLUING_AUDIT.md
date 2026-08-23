# Defect-Gluing-Audit

Stand: 2026-08-09

## Aussage und Normalisierung

Für eine binäre Folge `x` sei `rho_x(t)` ihre rohe negaperiodische
Autokorrelation und

```text
w = rho_a + rho_b,
v = rho_c + rho_d.
```

Die GS4-Bedingung ist exakt `w+v=0`. Das ist eine Meet-in-the-Middle-Zerlegung
der vorhandenen NAF-Gleichung, keine neue Charakterisierung. `w` ist dabei
nicht das durch vier reduzierte Trackerresiduum `u`; nur `w+v` ist immer durch
vier teilbar.

## Paritäts- und Minimalitätslemma

Für jeden Lag `t` gilt

```text
rho_x(t) = n - 2 N_-(x,t),
N_-(x,t) = t mod 2,
rho_x(t) = n - 2t mod 4.
```

Der zweite Schritt folgt daraus, dass das Produkt aller `n` signierten
Summanden der NAF an Lag `t` gleich `(-1)^t` ist. Daher gilt koordinatenweise

```text
w_t = 2n mod 4.
```

- Ungerades `n`: `w_t = 2 mod 4`, also `w_t != 0` und
  `||w||^2 >= 4 floor((n-1)/2)`. Gleichheit bedeutet `|w_t|=2` für jeden
  reduzierten Lag.
- Gerades `n`: `w_t = 0 mod 4`. `w=0` ist paritätsverträglich, aber seine
  Existenz ist die Existenz eines binären negaperiodischen Golay-Paars und
  folgt nicht aus der Kongruenz. Für jedes nichtnullige `w` gilt
  `||w||^2 >= 16`.

Die vollständige Enumeration bestätigt die Schranken bis `n=16`. Für gerade
`n=4,6,...,16` treten außerdem genau die minimalen Nichtnullvektoren
`+/-4 e_k` auf. Für ungerade `n` kollidieren minimale Vektoren mit ihren
Gegenvektoren, aber ab `n=11` nicht mehr jedes realisierbare minimale
Vorzeichenmuster. Das ist daher kein allgemeiner Satz über alle Muster.

## Der wichtige negative Befund

Ein Solver sollte `||w||` nicht blind minimieren. Sei `p_n(w)` die Anzahl oder
Wahrscheinlichkeit, mit der ein Folgenpaar das Residuum `w` erzeugt. Unter
allen GS4-Kollisionen ist die Verteilung des Zwischenresiduums exakt

```text
P(w | w+v=0) proportional to p_n(w) p_n(-w).
```

Das lässt sich als exakte erzeugende Funktion schreiben. Für formale
Mehrvariablen `y=(y_1,...,y_m)` sei

```text
F_n(y) = sum_x y^rho_x.
```

Dann sind die Koeffizienten von `F_n(y)^2` genau die Paarmultiplikitäten
`p_n(w)`, und die Anzahl geordneter GS4-Zustände ist der konstante
Koeffizient

```text
[y^0] F_n(y)^4 = sum_w p_n(w) p_n(-w).
```

Das Paarresiduum ist damit eine exakte Separatorvariable für eine
Meet-in-the-Middle- beziehungsweise Dynamic-Programming-Elimination der vier
Folgen. Der Zustandsraum der Separatorvariable bleibt allerdings exponentiell.

Schon in der vollständigen Enumeration wandert die dominante Kollisionsschale
von der minimalen Schale weg. Bei `n=16` enthält die Schale `||w||^2=112`
15.8 Prozent der geordneten Lösungen, während `w=0` nur 0.17 Prozent enthält.
Die relevante Größe ist die Komplement-Multiplizität von `-w`, nicht die Norm
von `w`.

Im aktuellen n=52-Archiv besitzen die 19 freien GS4-Lösungen unter allen drei
Paarungen kein `w=0`. Ihre kleinsten Paarresidualnormen liegen zwischen 640
und 1344 (Median 944). Damit liegt keine archivierte freie Lösung in der
minimalen Nichtnullschale `||w||^2=16`.

## Alternierende Verdopplung

Für gerade Länge gilt `rho_alt(x)(t)=(-1)^t rho_x(t)`. Für ungerade Länge ist
diese Identität falsch, weil die Wrap-Terme ihr Vorzeichen anders ändern.

Aus einer GS4-Lösung `(p,q,r,s)` der Länge `m` kann man dennoch exakt eine
Lösung der Länge `2m` bauen:

```text
A = interleave(p,q)
B = interleave(r,s)
C = alt(A)
D = alt(B).
```

Die geraden Lags decimieren auf das ursprüngliche GS4-Residuum, die ungeraden
Lags löschen sich durch Alternation. Für Paley-Basen `m=4,6,10` besitzt diese
Darstellung denselben Repo-Symmetriehash wie `double_gs4`, also die vorhandene
Golay-2/Turyn-Verdopplung. Sie ist eine saubere Koordinatendarstellung des
bekannten Abschlusses, kein belegter neuer Konstruktionsweg.

## Minimaler Solver-Test

Der sinnvolle Test ist nicht ein Minimalnorm-Codebook. Stattdessen:

1. Schätze für erzeugte Paarresiduen eine Komplementdichte für `-w`.
2. Wähle den ersten Paarzustand nach hoher Komplementdichte, nicht nach
   kleinstem `||w||`.
3. Löse das zweite Paar gegen das feste Ziel `-w` und übergib Near-Collisions
   anschließend an den freien GS4-Solver.
4. Vergleiche gegen zufällige erste Paare bei gleichem Kandidaten- und
   Wall-Time-Budget.

Eine exakte Codebook-Kollision ist für kleine `n` ein funktionierender
Konstruktor, bleibt aber exponentiell. Für `n=52` benötigt die Dichteidee erst
einen gemessenen Proxy; die Norm allein ist durch die Archiv- und
Kleinstbefunde widerlegt.

## Literaturabgrenzung

- Balonin und Djokovic, *Negaperiodic Golay pairs and Hadamard matrices*
  (2015), definiert den Spezialfall `w=0`, ordnet ihn als 2N-Typ ein und
  beschreibt die Turyn-Multiplikation. Die Beschränkung binärer NGPs auf
  gerade Längen ist dort bereits Teil des bekannten Rahmens:
  <https://arxiv.org/abs/1508.00640>.
- Near-normal sequences sind spezielle Base Sequences mit aperiodischer,
  längenungleicher Struktur. Sie sind nicht dasselbe Objekt wie das hier
  betrachtete gleichlange negaperiodische Paarresiduum:
  <https://arxiv.org/abs/0903.4390>.
- Klassische Supplementary Difference Sets beziehungsweise GS-Difference-
  Families beschreiben dieselbe Vierblock-Komplementarität für zirkulante
  Blöcke in einer endlichen abelschen Gruppe. Die Zerlegung in zwei
  Teilfamilien ist dort als Differenzmultimengen-Summe lesbar; unser `w|-w`
  ist die negazyklische Gruppenringversion davon, kein neues SDS-Objekt:
  <https://arxiv.org/abs/1802.00556>.
- In der Signalverarbeitung heißt NAF auch odd-periodic autocorrelation;
  optimale beziehungsweise fast-komplementäre Paare sind etablierte Themen.
  Der Paritätsunterrand ist daher nicht als wissenschaftlich neu einzustufen.

Offen und projektspezifisch nützlich ist die algorithmische Frage, ob eine
billige Komplementdichte die Erreichbarkeit eines Paarresiduums besser
vorhersagt als `Q`, `||w||` oder aggregierte Kanalenergien.

Reproduktion:

```powershell
$env:PYTHONPATH='.'
python -m experiments.construction_space.defect_gluing_audit
```
