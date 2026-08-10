# Synthese: vom GS4-Fehler zur Navigation

Stand: 2026-08-09

## Kurzfazit

Die bisherigen Befunde passen in eine klare Hierarchie. `Q` ist die exakte
Höhe, `D` ist das aktuelle lokale Move-Wörterbuch, der prospektive Grad misst
lokales Volumen in einer begrenzten Schale, und ein deterministischer Quench
definiert operative Basins. Keine dieser Größen ist allein ein
Lösungskompass.

Eine im Projekt bisher nicht explizit formulierte exakte Zielkoordinate ist

```text
Gamma(x) = (H_C(x), L_C(x)).
```

`H_C` ist die kleinste Passhöhe bis zu einem Zustand mit einem direkt
lösenden Single-Move. `L_C` ist die kürzeste Weglänge unter dieser optimalen
Passhöhe. Dieses Paar formalisiert genau **Passhöhe plus laterale Bewegung**.
Es ist exakt lösungsgerichtet und enthält nach vollständiger Enumeration bei
`n=5` mehr Information als `Q`, `u` oder `D^T D`. Global ist es teuer; die
aktuellen Lidar-Sensoren sind als billige Näherungen an Teile dieser
Koordinate zu verstehen.

Ein zweiter verbindender Befund ist algebraisch: Das Tripleprodukt-Dual `F`
ist kein fremder Fernoperator. Es ist ein strukturierter Satz von
Hamming-4-Spaltenmoves, der in der H4-Koordinate Basis und Achse festhält und
nur die Orientierungszeichen ändert. `A` und `C` sind exakt die symmetrische
und antisymmetrische Residualkomponente unter diesem Dual.

## 1. Informationshierarchie

Für einen Zustand `x=(a,b,c,d)` seien die vier Einzelresiduen

```text
R(x) = (rho_a,rho_b,rho_c,rho_d)
```

und

```text
u = (rho_a+rho_b+rho_c+rho_d)/4,
Q = ||u||^2.
```

Die bisherigen Objekte liegen auf verschiedenen Auflösungsstufen:

```text
Kantenwörter / vollständiger Zustand x
        |
        +--> konkrete Zeilenfaktorisierung D(x)
        |       |
        |       +--> lokale und prospektive Move-Topologie
        |
        +--> vier Einzelresiduen R(x)
                |
                +--> Paarresiduen und H4-Residualkanäle
                |
                +--> Gesamtresiduum u --> Q und D^T D
```

Die Pfeile sind Informationsverlust. Daraus folgen harte Grenzen:

- `Q` ist nur eine skalare Höhe.
- `u` trägt Lag und Vorzeichen, aber nicht die konkrete Anordnung der
  Move-Zeilen.
- `D^T D` ist vollständig durch `u` bestimmt und kann keine zusätzliche
  Basin-Information liefern.
- Paarresiduen `w` und die vollen vier NAF-Kanäle verfeinern die `u`-Faser.
  Ihre skalaren Normen werfen diese Zusatzinformation wieder weg.
- Die Kantenwörter sind bis auf vier globale Vorzeichen eine vollständige
  Koordinate. Sie sind daher eine exakte Karte, aber noch keine Kompression.

Das erklärt, warum mehrere scheinbar neue Energien keinen neuen Kompass
lieferten: Framepotential und aggregiertes Spektrum sind Funktionen von `u`;
die Split-Energie ist eine gekoppelte Energie zweier GS4-Zustände; minimale
Paarresidualnorm ist nicht dasselbe wie hohe Gegenvektordichte.

## 2. Das H4-Dual ist ein Orientierungsoperator

Setze spaltenweise

```text
beta_j = a_j b_j c_j d_j
```

und

```text
F(a,b,c,d) = (bcd,acd,abd,abc).
```

Dann gilt exakt

```text
F(x)_{s,j} = beta_j x_{s,j}.
```

Folglich negiert `F` genau die vier Bits der Spalten mit `beta_j=-1`. In der
H4-Zweibasenkoordinate ist das ein strukturierter Hamming-4-Move:

- `beta` bleibt erhalten;
- die H4-Basis bleibt erhalten;
- die Achse bleibt erhalten;
- nur das Orientierungszeichen `eta` wird an den markierten Spalten negiert.

Mit dem verifizierten starken Split gilt

```text
u(x)   = A+C,
u(Fx)  = A-C.
```

Also

```text
A = (u(x)+u(Fx))/2,
C = (u(x)-u(Fx))/2.
```

`A` und `C` sind damit die geraden und ungeraden Komponenten des Residuums
unter der Involution `F`. Die starke Bedingung `A=C=0` bedeutet exakt, dass
sowohl `x` als auch sein Orientierungsdual `F(x)` GS4-Lösungen sind.

Das liefert eine nützliche zweite Karte `(u(x),u(Fx))`, aber keinen direkten
freien Solver: Große freie GS4-Lösungen können `u(x)=0` und `u(Fx)!=0`
besitzen. `Q(Fx)` klassifiziert daher einen H4-Orientierungssektor, nicht die
allgemeine Lösungsnähe.

## 3. Pair residual und H4-Kanäle

Für eine Paarung gilt

```text
w = rho_a+rho_b,
v = rho_c+rho_d,
u = (w+v)/4.
```

Die Lösung verlangt `v=-w`. Die drei Paarungen sind lineare Projektionen der
vier Einzelresiduen. Zusammen mit dem Gesamtresiduum rekonstruieren sie `R`;
sie sind deshalb zusätzliche Zustandsinformation gegenüber `u`, aber keine
zusätzliche Bedingung gegenüber allen vier Einzelkanälen.

Die exakte Gluing-Identität

```text
#GS4 = sum_w p_n(w) p_n(-w)
```

zeigt den konstruktiven Wert: relevant ist die Erzeugbarkeit des
Gegenvektors, nicht `||w||`. Für Navigation ist daher der **signierte Vektor**
oder eine Näherung seiner Komplementdichte plausibel; eine Kanalnorm allein
ist zu grob.

## 4. Exaktes Cancellation-Barrieren-Lemma

Fixiere einen Move-Graphen, hier zunächst alle `4n` Singles. Sei

```text
S0 = {x : Q(x)=0}
```

und die Cancellation-Grenze

```text
C1 = {x not in S0 : es gibt einen Single-Move i mit d_i(x)=-u(x)}.
```

`C1` besteht genau aus den Nichtlösungen, die einen Move von `S0` entfernt
sind. Für die Communication Height

```text
Phi(x,Y) = min_{path x->Y} max Q(path)
```

gilt für jede Nichtlösung exakt

```text
boxed: Phi(x,S0) = Phi(x,C1).
```

**Beweis.** Jeder Pfad von `x` nach `S0` besitzt unmittelbar vor seiner ersten
Lösung einen Zustand aus `C1`. Umgekehrt lässt sich jeder Pfad nach `C1` um
den direkt lösenden Move ergänzen, dessen Endenergie null ist. Die maximale
Energie des Pfads ändert sich dadurch nicht.

Verfeinere die Minimax-Aufgabe lexikographisch. Definiere

```text
H_C(x) = Phi(x,C1),
L_C(x) = kürzeste Pfadlänge nach C1 unter der optimalen Höhe H_C(x).
```

Dann ist

```text
Gamma(x)=(H_C(x),L_C(x))
```

eine exakte lösungsgerichtete Navigationskoordinate. Die kürzeste entsprechend
optimale Strecke zur Lösung hat Länge `L_C+1`.

Wichtig ist die Trennung:

- `Q` sagt, wie hoch der aktuelle Zustand liegt.
- `H_C-Q` sagt, wie weit die niedrigste notwendige Passhöhe über dem Zustand
  liegt.
- `L_C` sagt, wie viel laterale Bewegung auf dieser Höhe noch nötig ist.

Damit ist die beobachtete Höhen-/Längenwechselwirkung kein bloßes Bild mehr.

## 5. Vollständige Enumeration bei n=5

`experiments/learning/navigation_synthesis.py` enumeriert alle `65.536`
Zustände nach Entfernung der vier globalen Folgenvorzeichen. Die `4n`
Single-Moves werden im Quotient exakt repräsentiert. Ein Multi-Source-
Minimax-Dijkstra berechnet `Gamma` für jeden Zustand.

Reproduzierbarer Aufruf:

```text
python -m experiments.learning.navigation_synthesis --n 5
```

Die Ausgabe liegt in `data/navigation-synthesis-small-n.json`.

Exakte Ergebnisse:

- `7.500` Lösungen, passend zur unabhängigen GS4-Enumeration;
- `3.000` nichtgelöste strikte Best-Improvement-Minima, alle bei `Q=1`;
- kein Zustand benötigt bei `n=5` eine Passhöhe oberhalb seines Start-`Q`;
- diese `3.000` Q1-Minima besitzen `Gamma=(1,1)` zur Cancellation-Grenze und
  benötigen danach den lösenden Move, also zwei Moves bis `Q=0`;
- `Q`-Fasern besitzen in vier von zwölf Fällen verschiedene exakte
  `(H,L)`-Werte;
- selbst identisches `u` und damit identisches `D^T D` besitzt in vier von
  31 Fasern verschiedene `(H,L)`-Werte.

Die n=5-Wand ist somit rein lateral: Strict Greedy hält bei Q1, obwohl ein
neutraler Q1-Move und danach ein Solve existiert. Das steht im scharfen
Kontrast zur selektierten n=52-Stichprobe: Dort waren die geprüften Q1-
Zustände im Q1-Graph isoliert, und vollständig erschöpfte Komponenten bis
`Q=9` enthielten keine Lösung. Für diese Stichprobe gilt daher
`H_C>=10`.

Dieser Kontrast ist eine konkrete Skalierungsbeobachtung:

```text
n=5:  laterale Q1-Barriere, aber keine Uphill-Barriere
n=52: für die geprüften Q1-Zustände echte Uphill-Barriere
```

### Die vier Einzelresiduen sind ein Sensor, aber keine vollständige Karte

In derselben n=5-Enumeration waren innerhalb jeder Faser der **vollen vier
Einzel-NAF-Kanäle** folgende drei Aggregate konstant:

- `(H_C,L_C)`;
- `(bestes Nachbar-Q, Zahl lösender Nachbarn, Zahl nichtschlechter Nachbarn)`;
- terminales `Q` des deterministischen Best-Improvement-Quenchs.

Dieser Klein-n-Effekt gilt nicht allgemein. Ein vollständig reproduzierbares
Gegenbeispiel bei `n=12` verwendet in der ersten Folge die normalisierten
Bitindizes `10` beziehungsweise `33` und in den anderen drei Folgen die
Indizes `46,46,298`. Beide Vierertupel besitzen **dieselben vier signierten
Einzel-NAF-Kanäle** und dasselbe `u`. Trotzdem besitzt das erste zwei, das
zweite drei direkt lösende Single-Flips. Der Regressionstest steht in
`tests/test_navigation_synthesis.py`.

Damit sind die Einzelkanäle weiterhin ein plausibler billiger Sensor, aber
keine exakte Navigationskoordinate. Sie verlieren wie jede Autokorrelation
Phasen- beziehungsweise Anordnungsinformation, die in den konkreten Zeilen
von `D` erhalten bleibt.

## 6. Exakte operative Basin-Barriere

Für einen deterministischen Quench `G` und ein Minimum `m` sei

```text
B_G(m)={x:G(x)=m}.
```

Definiere

```text
H_G(m) = min_{y not in B_G(m)} Phi(m,y).
```

Dann gilt äquivalent

```text
boxed:
H_G(m) = min {H : die Sublevel-Komponente von m unter Q<=H
                  ist nicht vollständig in B_G(m) enthalten}.
```

Das ist die exakte Höhe der operativen Basin-Wand. Sie ist nicht
solverunabhängig, weil `B_G` vom Quench abhängt; die Pfadhöhe innerhalb des
festgelegten Move-Graphen ist danach aber exakt definiert.

Eine Ziel-Basin-ID kann grundsätzlich durch Labelpropagation entlang des
deterministischen Successors berechnet und gecacht werden. Ohne bereits
bekannte Labels oder einen Quench gibt es aus `Q` allein keine exakte
Vorhersage: Zustände derselben `u`-Faser können unterschiedliche lokale
Wörterbücher und unterschiedliche Greedy-Endpunkte besitzen.

## 7. Was der prospektive Grad wirklich misst

Der vorhandene Sensor

```text
b_H(i)=#{j != i : Q(x flip i flip j)<=H}
```

ist der lokale Sublevel-Grad nach einem Probe-Move. Summen und Mittelwerte
über zulässige `i` schätzen das Volumen der bis Tiefe zwei sichtbaren
Sublevel-Umgebung. Daher passt das negative Vorzeichen zur Rückkehr logisch:
mehr Verzweigung bietet der Shell-Policy mehr laterale Ausgänge.

Der unabhängige Audit bestätigt bei relativer Höhe `+2` echten inkrementellen
Vorhersagewert über `n`, Root-`Q` und Entry-Count hinaus. Bei `+4` ist der
Zusatzwert klein. Der kostenbelastete adaptive Aktor senkte zwar die
Rückkehrquote, fand aber pro Arbeit nicht häufiger tiefere Basins als die
einfache feste Policy.

Eine spätere direkte Kopplung verwendete `b_H(i)` nicht nur zur Höhenwahl,
sondern zur Auswahl jedes einzelnen `+4`-Shell-Moves. Aus acht zufällig
gesampelten zulässigen Moves wurde jeweils der mit dem höchsten prospektiven
Grad genommen. Auf je 256 unabhängigen Roots stieg die Zahl tieferer
Zielbasins gegenüber dem Zufallswalk bei `n=40,44,48,52` von
`98/101/100/100` auf `133/158/155/169`. Alle vier gepaarten Tests waren
signifikant; das mittlere End-`Q` sank ebenfalls. Der unoptimierte Sensor
kostete dabei etwa 10- bis 12-mal so viele Bewertungen und produzierte noch
keine Lösung.

Ein gerichteter Vergleich ersetzte die reine Gradmaximierung durch

```text
V_2(x,i)=min_{j != i} Q(x flip i flip j).
```

Auf je 256 unabhängigen Roots bei `n=40,44,48,52` fanden Zufall,
Gradmaximierung und `V_2` jeweils `117/113/117/119`, `144/149/162/180` und
`201/189/193/214` tiefere Zielbasins. Das mittlere End-`Q` sank ebenfalls;
Lösungen entstanden noch nicht. Der unoptimierte Blick nach vorn kostete
ungefähr neun- bis zehnmal so viele Endpunktauswertungen.

`V_2` ist keine neue Energie und nicht bloß Pair-Rescue. Es ist ein
gesampelter Zwei-Schritt-Bellman-Backup mit terminalem Cost `Q`: Der Solver
führt nur den ersten Flip aus, berechnet am neuen Zustand erneut und darf den
Plan ändern. Der Q-Cap kontrolliert die Passhöhe. Pair-Rescue bewertet dagegen
typischerweise an einem Minimum ein Paar und akzeptiert dessen Endpunkt
atomar. Die neue Policy ist daher eine kurze
**receding-horizon/rollout-Steuerung** im Shell-Graphen.

Das ist kein Widerspruch. `b_H` schätzt einen Teil von `H_G` beziehungsweise
lokale Wanddurchlässigkeit und kann dadurch einen Walk in andere, häufiger
tiefere Attraktoren lenken. Es schätzt weder `H_C,L_C` noch die ID oder
Solve-Qualität des Zielbasins.

### Grenze der Hessian-Sprache

Für Flipindikatoren `z` ist das Residuum exakt quadratisch:

```text
u(x flip z) = u + sum_i z_i d_i + sum_{i<j} z_i z_j h_ij,
```

wobei `h_ij` nur für Same-Sequence-Paare auftreten muss. Die Energie ist aber

```text
Q(x flip z)=||u(x flip z)||^2
```

und deshalb im Allgemeinen **quartisch** in `z`. Bei einem festen n=7-Zustand
liefert die dritte Möbius-Differenz für die Flips `(0,1,3)` den Wert `2`, die
vierte für `(0,1,2,3)` den Wert `-4`. Der Test ist reproduzierbar in
`tests/test_navigation_synthesis.py`.

Die paarweisen Residualkorrekturen reichen weiterhin aus, um
`combo_energy()` exakt zu berechnen: Erst wird das quadratische Residuum
zusammengesetzt, dann vollständig quadriert. Falsch wäre nur die stärkere
Interpretation, die **Energie** selbst besitze keine kubischen oder
quartischen Wechselwirkungen. Gerade die laufende Neuberechnung von `V_2`
kann solche zustandsabhängigen höheren Wechselwirkungen implizit erfassen,
während ein einmalig eingefrorener Hessian dies nicht kann.

## 8. Fehlende gerichtete Größe

Für einen echten Lösungskompass ist statt bloßer Verzweigung eine
Cancellation-Hazard nötig. Eine exakt definierte Familie ist

```text
Z_{H,k}(x)
  = Anzahl unter Q<=H in höchstens k Moves erreichbarer Zustände aus C1.
```

Alternativ genügt für einen Ja/Nein-Sensor

```text
z_{H,k}(x) = [Z_{H,k}(x)>0].
```

Eigenschaften:

- `Z` ist direkt lösungsgerichtet;
- für wachsende `H,k` nähert es die exakte Koordinate `Gamma` an;
- `k=1` ist der bekannte Test `d=-u`;
- reine prospective degree ersetzt die Zielmenge `C1` durch „irgendein
  weiterer Zustand“ und verliert dadurch die Richtung.

Vollständiges `Z` ist teuer. Praktische Näherungen wären:

1. beam-begrenzte Suche nach minimalem Cancellation-Deficit;
2. GPU-Batching der Zwei-/Drei-Schritt-Endpunkte;
3. Meet-in-the-middle auf konkreten Delta-Kombinationen;
4. Lernen einer Hazard aus signierten Einzelresiduen, `u(Fx)` und lokalen
   Wörterbuchmotiven, mit `Z` auf kleinen/exakt untersuchten Räumen als Label.

Für `k=1` gibt es eine exakte Kantenwortformel. Mit

```text
P_t(c)=product_{r=0}^{t-1} p_{c+r}
```

ist ein Flip `(s,c)` genau dann lösend, wenn für jeden relevanten Lag

```text
-(P_t^(s)(c)+P_t^(s)(c-t))/2 = -u_t.
```

Dies ist allerdings nur die bereits bekannte Zeilenprüfung `d_{s,c}=-u` in
Kantenkoordinaten und asymptotisch nicht billiger als ein Scan des gecachten
Wörterbuchs.

Für **zwei Flips aus verschiedenen Folgen** fällt die Same-Sequence-
Korrektur weg. Die exakte Cancellation-Hazard lässt sich dann per Hashing
zählen:

```text
für jedes Folgenpaar (s,r):
    hashe alle Zeilen d_(r,*)
    suche für jede Zeile d_(s,c) den Schlüssel -u-d_(s,c)
```

Das zählt alle direkt lösenden Cross-Sequence-Paare in erwartetem `O(nm)`
statt `O(n^2 m)`. Der Prototyp
`cross_sequence_cancellation_pairs()` stimmt im Regressionstest mit einem
vollständigen `combo_energy()`-Scan überein. Für Same-Sequence-Paare hängt die
sparse Korrektur von beiden Spalten ab; dort folgt keine entsprechende
einfache Formel allein aus den Einzel-NAF-Kanälen.

Auf den bekannten n=52-Q1-Zuständen ist `Z_{9,k}=0` für alle vollständig
erschöpften Komponenten, nicht nur für kleine Hamming-Radien. Ein sinnvoller
n52-Test muss daher mindestens die empirisch belegte Passhöhe erreichen.

## 9. Literatur- und Neuheitsaudit

Das Cancellation-Barrieren-Lemma ist **kein neues allgemeines
Graphentheorem**. Für jeden vertexgewichteten Graphen und jede Zielmenge `S`
gilt die Minimax-Gleichheit zwischen `S` und ihrer äußeren Nachbargrenze, weil
jeder Zielpfad diese Grenze passieren muss. `H_C` ist die bekannte
Communication Height beziehungsweise ein Minimax-/Bottleneck-Path-Wert;
`L_C` ist lediglich ein Hopcount-Tie-Break unter den höhenoptimalen Pfaden.

Die allgemeine Landschaftssprache ist ebenfalls etabliert:

- Stillinger und Weber definieren die Abbildung von Konfigurationen auf
  lokale Minima als *inherent structures*;
- Becker und Karplus sowie Flamm et al. beschreiben Basins,
  Disconnectivity-/Barrier-Trees und minimale Sattelhöhen;
- Local Optima Networks komprimieren kombinatorische Suchräume zu
  Quench-Optima und gewichteten Übergängen;
- Transition Path Theory verwendet für eine **festgelegte stochastische
  Dynamik** Committoren als zielgerichtete Trefferwahrscheinlichkeiten;
- Ein-Schritt-Lookahead mit erneuter Planung ist als Rollout beziehungsweise
  receding-horizon control bekannt;
- diskrete Ableitungen höherer Ordnung sind Standardbegriffe der
  Pseudo-Boolean-Theorie.

Formans diskrete Morse-Theorie ist hier nicht unmittelbar dieselbe Sache. Sie
verlangt eine Morse-Funktion auf Zellen eines CW-Komplexes und ein zulässiges
Gradient-Matching zwischen Zelldimensionen. Der aktuelle best-improvement
Successor ist nur eine deterministische Abbildung auf den **Ecken** des
Hypercubes und nicht automatisch ein diskretes Morse-Vektorfeld. Ohne eine
explizite Zellfortsetzung von `Q` liefert diese Sprache derzeit keinen neuen
Algorithmus.

Projektspezifisch substantiell sind somit nicht `Phi` oder `Gamma` als
allgemeine Begriffe, sondern:

1. die algebraische GS4-Zielgrenze `C1={x:exists i, d_i=-u}`;
2. ihre exakte Berechenbarkeit aus dem NAF-Wörterbuch;
3. der empirische Wechsel von lateral erreichbaren Q1-Minima bei `n=5` zu
   belegten hohen Barrieren in der selektierten n52-Stichprobe;
4. die Verbindung von Shell-Caps, prospektiver Wörterbuchtopologie und
   wiederholtem Zwei-Schritt-Lookahead;
5. die neue billige Hash-Zählung für Cross-Sequence-Cancellation-Paare.

Primärquellen:

- [Stillinger und Weber, *Hidden structure in liquids* (1982)](https://doi.org/10.1103/PhysRevA.25.978)
- [Becker und Karplus, *The topology of multidimensional potential energy surfaces* (1997)](https://doi.org/10.1063/1.473299)
- [Flamm et al., *Barrier Trees of Degenerate Landscapes* (2002)](https://doi.org/10.1524/zpch.2002.216.2.155)
- [Cirillo und Nardi, Communication Height für diskrete Metastabilität (2015)](https://doi.org/10.1007/s10955-015-1334-6)
- [Ochoa et al., *A Study of NK Landscapes' Basins and Local Optima Networks* (2008)](https://doi.org/10.1145/1389095.1389204)
- [E und Vanden-Eijnden, *Towards a Theory of Transition Paths* (2006)](https://doi.org/10.1007/s10955-005-9003-9)
- [Bertsekas, Tsitsiklis und Wu, *Rollout Algorithms for Combinatorial Optimization* (1997)](https://doi.org/10.1023/A:1009635226865)
- [Foldes und Hammer, höhere diskrete Ableitungen pseudo-boolescher Funktionen (2005)](https://doi.org/10.1287/moor.1040.0128)
- [Forman, *Morse Theory for Cell Complexes* (1998)](https://doi.org/10.1006/aima.1997.1650)

## 10. Ein möglicher GPS-Stack

Die vorhandenen Sensoren und Aktoren ergeben kein einzelnes magisches Skalar,
sondern einen hierarchischen Zustandsvektor:

```text
Höhe:          Q
Fehlerlage:    signiertes u
Kanalchart:    vier rho_s beziehungsweise drei Paarungen
Dualchart:     u(Fx), beta/axis/orientation
Lokale Karte:  konkrete D-Zeilen und b_H(i)
Basin-ID:      G(x) beziehungsweise gecachter Attraktor-Hash
Zielrichtung:  Näherung an Gamma=(H_C,L_C)
```

Damit lässt sich die Forschungsfrage präziser stellen:

> Welche billige Kombination aus Kanalchart, Dualchart und lokaler
> Wörterbuchtopologie sagt nicht nur „die Wand ist dünn“, sondern senkt die
> geschätzte Cancellation-Barriere oder verkürzt den Weg zur
> Cancellation-Grenze?

Das ist der Übergang vom Lidar zum GPS. Der Attraktor-Hash sagt, wo der Solver
war; die Transition-Tabelle sagt, wohin ein Aktor typischerweise führt; eine
Schätzung von `Gamma` sagt, ob dieses Ziel näher an einer lösenden Grenze
liegt.

## 11. Harte Negativresultate und Grenzen

Folgendes sollte nicht erneut als unabhängiger Kompass getestet werden:

- Framepotential, Gram-Eigenwerte oder aggregierte Spektralenergie: aus `u`
  rekonstruierbar;
- `Q_split` als dritte Energie: exakt gekoppelt an `Q(x)+Q(Fx)`;
- `||w||` als Gluing-Ziel: minimale Norm hatte weder maximale
  Kollisionsmasse noch erklärte sie freie n=52-Lösungen;
- reiner Escape als Erfolgskriterium: `+16` wechselte fast immer das Basin,
  ohne nennenswerte Solve-Hazard;
- Basin-Begriffe ohne festes `G`: sie sind nicht eindeutig;
- aus n=5 ableiten, dass volle Einzel-NAF-Kanäle allgemein `Gamma` oder die
  Cancellation-Hazard bestimmen: Das n12-Gegenbeispiel widerlegt bereits die
  exakte Form.
- eine per GF(2) rekonstruierte Zeile `d=-u` als selbstkonsistent lösenden
  Projektionsschritt behandeln: Nach Ersetzen der Folge und dem vorgesehenen
  Flip bleibt exakt `(rho_neu-rho_alt)/4`; tausende affine Kandidaten pro Root
  lieferten ab `n=24` keinen Vorteil gegenüber zufälligem Folgeneratz.

Die vollständige Herleitung und Ablation steht in
`experiments/learning/INTEGRABILITY_NAVIGATION_AUDIT.md`.

## 12. Nächste falsifizierbare Schritte

1. In neuen Multi-n-Root-Daten die vier signierten Einzelresiduen und
   `u(Fx)` speichern. Prüfen, ob sie Return, Ziel-`Q` oder wiederkehrende
   Zielbasins innerhalb festen `n,Q` vorhersagen.
2. Für vollständig beherrschbare kleine `n` `Gamma` exakt berechnen und
   Sensoren ausdrücklich gegen `H_C` und `L_C` statt nur gegen Return fitten.
3. Für n52-Q1-Zustände den ersten erreichbaren Cap adaptiv suchen und dort
   beam-/GPU-begrenzt `Z_{H,k}` beziehungsweise minimalen
   Cancellation-Deficit messen.
4. Basin-Transitionen als gerichteten Graphen mit Attraktor-Hash cachen. Ein
   Aktor wird dann nach Neuheit des Ziels und geschätzter `Gamma`-Verbesserung
   ausgewählt, nicht nur nach Escape.
5. Erst nach diesem Offline-Nachweis den Production-Solver ändern.
6. `10n`/`15n` nur als bedingten Makro-Aktor weiterverfolgen: Auf
   spaltengeclusterten Reverse-Defekten verkürzen sie den Abstand exakt, bei
   gleichem allgemeinen Auswertungsbudget waren sie gegen `4n` jedoch neutral.

## Status der Aussagen

**Exakt bewiesen oder vollständig enumeriert:** Informationsbeziehungen,
F-Orientierungsidentität, A/C-Zerlegung, Pair-Gluing, Tight-Frame-Grenze,
Cancellation-Barrieren-Lemma, operative Basin-Barriere, n=5-Enumeration,
n12-Gegenbeispiel für die Einzel-NAF-Kanäle, quartische Flip-Wechselwirkungen
von `Q` und Cross-Sequence-Pair-Hazard per Hashing.

**Empirisch multi-n belegt:** Prospective degree sagt Return bei `+2` voraus
und lenkt eine `+4`-Shell häufiger in tiefere Attraktoren; gerichteter
Zwei-Schritt-Lookahead verstärkt diesen Effekt; Passhöhe und laterale Länge
wirken getrennt; Escape beziehungsweise tieferes End-`Q` allein sagen noch
keine Solve-Qualität voraus; Symbol-Makromoves erzeugen zusätzliche Hazard nur
dann effizient, wenn ihre spaltenlokale Struktur zum Defekt passt.

**Spekulativ/offen:** Einzel-NAF-Kanäle als statistischer GPS-Chart,
Cancellation-Hazard als billiger Produktionssensor, Wiederverwendbarkeit
gelernter Basin-Transitionen über `n` und Symmetrien hinweg.
