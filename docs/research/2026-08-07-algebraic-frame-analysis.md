# Algebraische Struktur des GS4-Residualsolvers

Stand: 2026-08-07

> Archiviertes Untersuchungsprotokoll. Der aufbereitete, aktuelle Stand steht
> in [`../TIGHT_FRAME_CHARACTERIZATION.md`](../TIGHT_FRAME_CHARACTERIZATION.md).

Diese Notiz trennt exakte Identitäten, Messbefunde und offene Hypothesen. Sie
bezieht sich auf den aktuellen NAF-Tracker mit vier Folgen in `{+1,-1}^n`.

## 1. Das tatsächlich optimierte Objekt

Für jede Folge `x^(s)`, `s=0,1,2,3`, wird sie antiperiodisch fortgesetzt:

```text
x[j+n] = -x[j].
```

Damit ist ihre negaperiodische Autokorrelation

```text
A_s(t) = sum_j x^(s)[j] x^(s)[j+t],
```

wobei über `j=0,...,n-1` summiert wird. Der kombinierte Residualvektor ist

```text
r_t = sum_s A_s(t),       u_t = r_t / 4,
Q = sum_t u_t^2,          E_repo = 64 n Q.
```

Die GS4-Matrix ist genau dann Hadamard, wenn alle unabhängigen `u_t` null
sind. `Generator` erzeugt die vier Folgen, `Solver` verändert sie, `Tracker`
hält `u`, `Q` und die exakten Move-Deltas, und `Builder` setzt erst danach die
Matrix der Ordnung `4n` zusammen.

## 2. Die richtige Zahl unabhängiger Lags

Die antiperiodische Korrelation erfüllt

```text
r_{-t} = r_t,             r_{t+n} = -r_t.
```

Darum gibt es

```text
m_eff = floor((n-1)/2)
```

echte Koordinaten. Bei ungeradem `n` stimmt das mit `floor(n/2)` überein. Bei
geradem `n` ist der zusätzliche Lag `t=n/2` identisch null:

```text
u_{n/2} = 0
d_{i,n/2} = 0 für jeden Flip i.
```

Der aktuelle Tracker speichert diese Nullspalte noch. Für alle folgenden
Formeln bezeichnet `D` nur die ersten `m_eff` Spalten des Delta-Caches.

## 3. Single-Deltas als zustandsabhängiges Wörterbuch

Für den Flip an Position `c` in Folge `s` ist das reduzierte Delta am Lag `t`

```text
d_{s,c,t}
  = Delta u_t
  = -(1/2) x^(s)[c] (x^(s)[c+t] + x^(s)[c-t]).
```

Folglich gilt `d_{s,c,t} in {-1,0,1}`. Stapelt man alle `4n` Flips als Zeilen,
erhält man das zustandsabhängige Wörterbuch

```text
D in {-1,0,1}^{4n x m_eff}.
```

Die vom Tracker berechnete Single-Energie ist exakt

```text
Q_i' = Q + 2 u^T d_i + ||d_i||_2^2.
```

Das Matching-Pursuit-Bild ist daher korrekt, aber mit zwei Einschränkungen:

1. Das Wörterbuch ändert sich nach jedem akzeptierten Flip.
2. Mehrere Flips derselben Folge besitzen quadratische Paarwechselwirkungen;
   Deltas verschiedener Folgen addieren sich dagegen exakt linear.

## 4. Erste globale Identität: Summe aller Wörter

Jeder Korrelationssummand `x[j]x[j+t]` wird beim Summieren der Single-Deltas
über alle Positionen zweimal getroffen. Beide Endpunkt-Flips ändern ihn um
`-2 x[j]x[j+t]`. Daher gilt für jede einzelne Folge

```text
sum_c d_{s,c,t} = -A_s(t)
```

und über alle vier Folgen

```text
boxed: sum_i d_i = -r = -4u.
```

Der Zielvektor `-u` besitzt somit immer die dichte fraktionale Darstellung

```text
-u = (1/4) sum_i d_i.
```

Das garantiert keinen kleinen binären Flip-Subset. Es zeigt aber exakt, warum
das Problem ein diskretes Vector-Balancing-Problem ist: Der Zielvektor liegt
immer im positiven Kegel des aktuellen Wörterbuchs; gesucht wird eine dünne,
zulässige und wegen Same-Sequence-Termen nicht vollständig lineare
Darstellung.

## 5. Hauptbefund: Der Solver baut ein Tight Frame

Für zwei unabhängige Lags `t,l` ergibt sich

```text
(D^T D)_{t,l}
  = (1/4) sum_{s,c}
      (x[c+t] + x[c-t]) (x[c+l] + x[c-l])
  = (1/2) (r_{l-t} + r_{l+t}).
```

Im letzten Ausdruck wird `r` mit `r_{-k}=r_k` und `r_{k+n}=-r_k`
fortgesetzt. Insbesondere ist `r_0=4n`, also

```text
(D^T D)_{t,t} = 2n + r_{2t}/2.
```

Damit ist die komplette Spalten-Grammatrix des Move-Wörterbuchs bereits durch
den Residualvektor bestimmt. Für jede Lösung folgt

```text
boxed: D^T D = 2n I.
```

Die `m_eff` Delta-Spalten sind dann orthogonal und gleich lang. Der Solver
minimiert daher nicht nur eine Korrelation: Er orthogonalisiert implizit sein
eigenes ternäres Move-Wörterbuch zu einem Tight Frame.

Die Umkehrung gilt für `n>=5` ebenfalls. Sie folgt aus der folgenden exakten
Normidentität.

## 6. Exakte Framepotential-Identität

Sei

```text
F = D^T D - 2n I.
```

Einsetzen von `r=4u` in die vorherige Formel und Zählen der Vorkommen jeder
unabhängigen Lag-Koordinate liefert für gerades `n>=6`

```text
boxed: ||F||_F^2 = 4 (n-4) Q.
```

Für ungerades `n>=5` gibt es zusätzlich die spezielle Frequenz `z=-1`. Mit

```text
a = sum_{t=1}^{(n-1)/2} (-1)^t u_t
```

gilt

```text
boxed: ||F||_F^2 = 4 (n-4) Q + 8 a^2.
```

Der Zusatzterm ist derselbe Defekt, den man an der reellen negazyklischen
Frequenz sieht:

```text
sum_s |X_s(-1)|^2 - 4n = 8a.
```

Damit ist `D^T D = 2nI` für `n>=5` äquivalent zu `Q=0`. Das ist eine neue
exakte Charakterisierung, aber keine billigere Zielfunktion: `Q` kostet nur
`O(m_eff)`, während ein explizites `D^T D` deutlich mehr Arbeit benötigt.

### Konsequenzen für eine Lösung

Aus Spaltensumme und Spaltennorm folgen für jeden Lag genau

```text
n Einträge +1,
n Einträge -1,
2n Einträge 0.
```

Jede GS4-Lösung erzeugt somit ein balanciertes strukturiertes ternäres Tight
Frame. Das verbindet das Problem zusätzlich mit partiellen Wiegematrizen,
signierten Inzidenzstrukturen, Frame-Design und diskretem Vector Balancing.

## 7. Exakte Q=1-Geometrie

Wegen der Ganzzahligkeit von `u` bedeutet

```text
Q=1  <=>  u = sigma e_t,   sigma in {+1,-1}.
```

Eine Verbesserung ist dann nur noch `Q'=0`. Ein Single-Flip löst genau dann,
wenn

```text
d_i = -u.
```

Noch genauer: Bezeichne mit `u_ext(2t)` die antiperiodisch auf die unabhängigen
Lags zurückgeführte Koordinate. Für Spalte `t` gelten

```text
sum_i d_{i,t}   = -4u_t,
sum_i d_{i,t}^2 = 2n + 2u_ext(2t).
```

Daraus folgen die exakten Anzahlen

```text
#(+1) = n + u_ext(2t) - 2u_t,
#(-1) = n + u_ext(2t) + 2u_t.
```

Bei `Q=1` zeigen daher normalerweise genau `n+2` Atome in die richtige
Richtung am fehlerhaften Lag. Das Problem ist nicht fehlendes Alignment,
sondern Kollateralschaden in den übrigen Koordinaten.

Für ein richtig ausgerichtetes Atom mit `d_{i,t}=-sigma` gilt sogar

```text
Q_i' = ||d_i||_2^2 - 1.
```

Der erste Tabu-Schritt löscht den einzigen alten Fehler und erzeugt die
anderen Nichtnullkoordinaten dieses Atoms. Tabu betreibt an der Q=1-Wand also
eine exakte Residualverlagerung. Die Tenure verhindert, dass derselbe Flip
sofort rückgängig gemacht wird; weitere Schritte verändern zugleich das
Wörterbuch, bis eine annihilierende Darstellung erreichbar wird.

## 8. Q=2 und kleine Q

Ganzzahligkeit schränkt die Geometrie stark ein:

```text
Q=2: zwei Koordinaten sind +/-1.
Q=3: drei Koordinaten sind +/-1.
```

Deshalb sind für `Q<=3` die Zusatzmetriken

```text
||u||_1, support(u), ||u||_infinity
```

bereits durch `Q` festgelegt: `||u||_1=support(u)=Q` und
`||u||_infinity=1`. Erst ab `Q=4` entstehen verschiedene Formen, etwa vier
Einheitsfehler oder ein einzelner Fehler vom Betrag zwei. Diese Metriken sind
für frühe und mittlere Trajektorien sinnvoll, erklären aber die terminale
Q=1-Wand nicht zusätzlich.

## 9. Validierte Experimente

Alle folgenden Tests waren read-only Wegwerfexperimente gegen den exakten
Tracker. Es wurden keine Produktions- oder Datendateien verändert.

### Algebraische Identitäten

- `sum_i d_i = -4u` wurde für Zufallszustände bei
  `n=5,6,7,8,30,35,38,52` ohne Abweichung geprüft.
- Die elementweise Formel für `D^T D` wurde bei denselben Größen ohne
  Abweichung geprüft.
- Die gerade/ungerade Framepotential-Identität wurde für jedes
  `n=5,...,53` mit je 25 Zufallszuständen geprüft: keine Abweichung.
- Für gerade `n` waren die gespeicherte Mittelpunktkoordinate und ihre gesamte
  Delta-Spalte immer exakt null.

### Rang und Kondition

Bei je 50 Zufallszuständen war `D` für alle getesteten Größen vollrangig. Die
mittlere Konditionszahl lag ungefähr bei

```text
n=30: 2.51
n=35: 2.83
n=38: 2.69
n=52: 2.87
```

Das lineare Aufspannproblem ist somit nicht die Barriere. Bei Lösungen ist
die Konditionszahl wegen `D^TD=2nI` exakt eins. Rang, Singularwerte und
Spaltenkohärenz enthalten wegen der Grammatrix-Identität keine unabhängige
Information neben `u`. Interessant bleiben die Anordnung der Zeilen, dünne
Subset-Summen und höhere kombinatorische Beziehungen.

### Reale n=38-Endzustände

Ein Lauf mit 60 Seeds und 200.000 Steps ergab 34 Lösungen. Alle 26
Fehlschläge endeten beim besten gespeicherten Zustand mit `Q=1`.

Für 13 reproduzierte Q=1-Endzustände galt:

- jeweils genau 40 zielgerichtete Atome, wie für `n=38` vorhergesagt;
- minimale Norm dieser Atome zwischen 4 und 6;
- kein exakter Single-Escape;
- kein exakter Pair-Escape unter allen `152 choose 2` Paaren, nicht nur unter
  den Top-K-Kandidaten;
- minimale erreichbare Pair-Energie `Q=2,...,5`;
- kein exakter 1-, 2-, 3- oder 4-Flip-Escape mit höchstens einem Flip aus
  jeder Folge. Der 4-Flip-Fall wurde als exaktes Meet-in-the-Middle-4SUM in
  `O(n^2)` getestet und gefundene Treffer wären mit `combo_energy()` geprüft
  worden; es gab keinen Treffer.

Zusätzlich wurden für zwei dieser Zustände alle exakten Triple-Kombinationen,
einschließlich Same-Sequence-Korrekturen, geprüft. Die besten Triple-Ziele
hatten `Q=3` beziehungsweise `Q=2`, nicht null.

Das widerlegt keine größeren Escape-Sets. Es zeigt aber, dass die reale
Q=1-Wand nicht bloß durch die Top-K-Auswahl des Pair-Rescues entsteht.

### Startbedingungen bei ungeradem n

Bei `n=35`, 50.000 Steps, wurden zwei unabhängige Seed-Blöcke mit 150 und 100
Runs untersucht. Initiales `Q` und der Spezialterm `|a|` trennten erfolgreiche
von erfolglosen Runs nicht. Schwache Signale von `support(u)` und maximalem
Alignment im ersten Block replizierten im zweiten Block nicht.

Damit gibt es derzeit keinen belastbaren Startfilter aus diesen Größen. Das
passt zum bereits verworfenen cyclotomischen Startfilter.

### Sparse Score-Auswertung

Eine Numba-Wegwerfvariante wertete `D@u` nur auf `support(u)` aus. Sie war bei
`Q=1` etwa 1.25-1.5-mal schneller für den reinen Score-Kernel, verlor aber
bereits bei Support 5 den Vorteil oder wurde langsamer. Da ein Tabu-Walk den
Q=1-Zustand nach dem ersten Schritt verlässt, ist das kein überzeugender
Produktionspfad.

### Produktiver n=52-Trace

Der Trace `gs4-n0052-seed010000.npz` enthält 12 Millionen Budgetschritte und
35,4 Millionen echte Auswertungen. Der globale Bestpfad war:

```text
Q=128 -> ... -> 8     durch Singles, bis Evaluation 620
Q=6                   durch Tabu, Evaluation 1.931
Q=5                   durch Tabu, Evaluation 13.335
Q=4                   durch Tabu, Evaluation 14.646
Q=3                   durch Tabu, Evaluation 264.600
Q=2                   durch Tabu, Evaluation 5.376.385
```

Danach wurde weder `Q=1` noch `Q=0` erreicht. Der gespeicherte Q=2-Bestzustand
hat genau zwei Einheitsresiduen,

```text
u_18=+1, u_23=-1.
```

Sein Frame-Defekt erfüllt exakt

```text
||D^TD-104I||_F^2 = 384 = 4(52-4)Q,
```

und die Konditionszahl von `D` beträgt nur etwa 1,074. Trotz dieser fast
perfekten linearen Geometrie ist der Zustand vollständig single- und
pair-isoliert: Der beste Single-Flip und das beste unter allen `208 choose 2`
Paaren führen jeweils nur zu `Q=7`; es gibt keinen verbessernden Pair-Move.
Auch ein exaktes Meet-in-the-Middle-4SUM mit einem Flip je Folge fand keinen
direkten Escape.

Der erste deterministische Tabu-Schritt verschiebt die zwei Residuen in sieben
Einheitsresiduen (`Q=7`, `L1=7`, Support 7). Alle im Trace gespeicherten
aktuellen Zustände mit `Q=3,...,6` bestanden ebenfalls ausschließlich aus
Einheitsresiduen. Der große Lauf bestätigt damit das Residualtransport-Modell:
Nach der frühen Single-Deszendenz verbessert Tabu nicht durch bessere lineare
Konditionierung, sondern durch lange Folgen diskreter Umordnungen eines bereits
nahezu tight-frame-artigen Wörterbuchs.

`data/solutions.json` war während dieser Untersuchung noch leer. Gelöste
n=52-Endzustände konnten daher noch nicht auf höhere Zeilen- und
Subset-Strukturen verglichen werden.

## 10. Was sich wirklich kürzen lässt

### Sinnvoll

1. **Geraden Mittelpunkt-Lag entfernen.** Für gerades `n` kann der Tracker
   `m_eff=(n-1)//2` statt `n//2` verwenden. Das entfernt eine beweisbar tote
   `u`- und Delta-Spalte. Der Tabu-Innerloop würde bei `n=38` etwa 1/19 und bei
   `n=52` etwa 1/26 seiner Lag-Arbeit verlieren, ohne den Algorithmus zu
   ändern.
2. **Trace auf kombinatorische Daten ausrichten.** Neben `u` und `Q` sind
   minimale Kollateralnorm zielgerichteter Atome, Verteilung der Row-Normen,
   nächste Pair-/Subset-Distanz zu `-u` und die Phase des Übergangs wertvoll.
   Rang oder `D^TD` müssen nicht geloggt werden; sie lassen sich aus `u`
   rekonstruieren.
3. **Frame-Identität als unabhängiger Debug-Check.** Stichprobenweise kann sie
   Cachefehler erkennen. Im Hot-Path wäre sie zu teuer.

### Nicht sinnvoll oder nicht belegt

1. `Q` durch das Framepotential ersetzen: mathematisch äquivalent, aber
   rechnerisch teurer.
2. Rang oder Konditionszahl als Escape-Kriterium verwenden: sie sind durch
   `u` bestimmt und nahe Q=1 bereits fast optimal.
3. Den Start allein nach der ungeraden `z=-1`-Komponente filtern: kein
   replizierter Nutzen.
4. Nur auf den Support von `u` zu multiplizieren: außerhalb des ersten
   Q=1-Schritts kein stabiler Laufzeitgewinn.
5. Ein 4SUM mit einem Flip je Folge direkt integrieren: auf 13 realen
   Q=1-Plateaus kein einziger Treffer.

## 11. Offene mathematische Fragen

1. **Integrabilität des Frames:** Welche balancierten ternären Tight Frames
   `D` entstehen tatsächlich aus vier binären antiperiodischen Folgen? Eine
   umkehrbare Charakterisierung könnte das GS4-Problem in einen kleineren
   strukturierten Raum übertragen.
2. **Minimale Escape-Breite:** Wie groß ist für reale `Q=1`-Plateaus die
   kleinste Flipmenge mit exaktem Gesamtdelta `-u`, wenn Same-Sequence-Terme
   zugelassen sind? Hier passen Meet-in-the-Middle, SAT/ILP oder ein begrenzter
   Gray-Code-Ablationstest besser als weitere Single-Metriken.
3. **Wörterbuchwechsel statt Wörterbuchrang:** Welche Flipfolge verändert die
   Zeilenanordnung von `D` am schnellsten, während `Q` kontrolliert bleibt?
   Das ist wahrscheinlich die eigentliche Wirkung von Tabu.
4. **Residualtransport:** Verschwinden erfolgreiche Runs durch wiederkehrende
   Transportmuster `Q=1 -> k -> ... -> 0`? Die geplanten Trajektoriensnapshots
   können zielgerichtete Atome und erzeugten Kollateralsupport je Tabu-Schritt
   vergleichen.
5. **Cyclic Lift:** Der Lift `(x,-x)` erklärt dieselbe Struktur in den ungeraden
   `2n`-DFT-Bins, reduziert aber allein keine Freiheitsgrade. Interessant wird
   er erst, wenn ein zyklischer Algorithmus andere zulässige Frame- oder
   Subset-Moves erzeugt.

## Fazit

Der Solver optimiert keine zweite geheime lineare Größe neben `Q`.
Überraschender ist: `Q` ist bereits exakt ein Maß dafür, wie weit das
zustandsabhängige Single-Flip-Wörterbuch von einem balancierten ternären Tight
Frame entfernt ist. Die lineare Geometrie wird bei kleinen Q nahezu perfekt;
die verbleibende Schwierigkeit ist rein diskret: eine kleine zulässige
Kombination zu finden oder das Wörterbuch so zu verändern, dass eine solche
Kombination entsteht. Genau das erklärt die Wirkung von Tabu besser als das
Bild eines bloßen zufälligen Kicks.
