# Basin-Navigation im GS4-Suchraum

Stand: 2026-08-09

Dieses Dokument definiert die Krater-/Basin-Sprache des Projekts exakt. Die
zugrunde liegenden NAF-, Frame-, H4- und Wörterbuchformeln stehen in
[`MATH_MAP.md`](MATH_MAP.md). Hier geht es ausschließlich um Navigation.

## 1. Drei getrennte Objekte

Für festes `n` sei

```text
Omega_n = {+-1}^{4 x n}.
```

Eine Move-Menge `M` macht daraus einen ungerichteten Zustandsgraphen. Singles,
`10n`-Symbole und `15n`-Symbole definieren verschiedene Graphen.

Folgende Ebenen dürfen nicht vermischt werden:

1. **Landschaft:** Zustände, Move-Graph und exakte Energie `Q`.
2. **Quench:** eine festgelegte deterministische Abstiegsabbildung `G`.
3. **Exkursion:** eine Policy `K`, die vorübergehend bergauf oder seitwärts
   laufen darf.

`Q` und der Move-Graph sind solverunabhängig. Ein Basin hängt von `G` ab; eine
Rückkehrwahrscheinlichkeit zusätzlich von `K`.

## 2. Deterministischer Quench und Basin

Der kanonische Forschungsquench `G_best_single` lautet:

```text
alle 4n Single-Nachbarn auswerten
-> Nachbar mit kleinstem Q wählen
-> bei Gleichstand kleinsten flachen Index wählen
-> nur strikt niedrigere Q akzeptieren
-> am lokalen Minimum stoppen
```

Damit ist `G(x)` eindeutig. Für ein terminales Minimum `m` ist

```text
B_G(m) = {x : G(x)=m}.
```

Die Relation

```text
x ~_G y  <=>  G(x)=G(y)
```

ist eine Äquivalenzrelation und partitioniert den Zustandsraum. Ein exakter
State-Hash identifiziert das operative Basin. Ein zusätzlicher GS4-Class-Hash
kontrolliert, ob zwei verschiedene Minima nur durch die im Projekt
kanonisierten Negashift-/Reverse-/Permutationssymmetrien getrennt sind.

Ein anderes Tie-Break, First-Improvement, neutrale Moves oder eine andere
Move-Menge erzeugen im Allgemeinen eine andere Partition. Das
Production-Tabu ist auf dem Bitzustand allein weder deterministisch noch
Markov, weil Tabu-Tabelle und RNG zur Dynamik gehören.

## 3. Intrinsische Passhöhe

Für Zustände `a,b` ist die Communication Height

```text
Phi(a,b) = min_path(a->b) max_{x on path} Q(x).
```

Die Barriere aus `a` ist

```text
Delta(a->b)=Phi(a,b)-Q(a).
```

Äquivalent ist `Phi(a,b)` die kleinste Höhe `H`, bei der `a` und `b` im
Sublevel-Graphen

```text
Omega_<=H = {x : Q(x)<=H}
```

verbunden sind. Diese Größe hängt nur von `Q` und der Move-Menge ab. Eine
vollständig ausgeschöpfte BFS unter `H` kann daher eine echte Untergrenze für
die Passhöhe beweisen. Ein Node-/Tiefen-limitierter Lauf kann das nicht.

Für die 109 bisher vollständig geprüften n=52-Q1-Zustände gilt unter Singles:

- keine Q0/Q1-Endpunkte innerhalb Hamming-Radius drei;
- keine Lösung in vollständig ausgeschöpften Komponenten bis `Q=9`;
- somit für diese Stichprobe Passhöhe mindestens `Q=10`.

Dies ist keine allgemeine n=52- oder GS4-Aussage. Die Stichprobe besteht aus
selektierten Endzuständen eines bestimmten Benchmarks.

Für die Cancellation-Grenze

```text
C1 = {x != Lösung : es existiert ein Single mit d=-u}
```

gilt exakt `Phi(x,Lösung)=Phi(x,C1)`: Jeder Lösungspfad passiert unmittelbar
vor seinem letzten Move `C1`, und jeder Pfad nach `C1` lässt sich ohne höhere
Energie um den Lösungsflip ergänzen. Daraus folgt die exakte gerichtete
Koordinate

```text
Gamma(x)=(H_C,L_C),
```

minimale Höhe bis `C1` plus kürzeste Weglänge auf dieser optimalen Höhe. Sie
formalisiert das beobachtete Zusammenspiel aus Passhöhe und lateraler Bewegung.
`Q`, `u` und `D^T D` bestimmen `Gamma` bereits bei vollständiger n=5-
Enumeration nicht.

## 4. Shell-Exkursion, Rückkehrkurve und Übergänge

Eine vollständig spezifizierte Exkursionspolicy sei

```text
K_shell(H,L,T),
```

mit maximaler Q-Höhe `H`, Länge `L` und Auswahl-/Tabu-Regel `T`.

Für ein Minimum `m` ist die Rückkehrwahrscheinlichkeit

```text
R_{G,K}(m) = P(G(x_after_excursion)=m).
```

Die empirische Escape-Wahrscheinlichkeit ist `1-R`. Eine praktische
charakteristische Höhe ist beispielsweise

```text
H50(m) = min {H : R_{G,K}(m)<=1/2}.
```

Zwischen Minima entsteht die Übergangsmatrix

```text
T_{G,K}(m,m') = P(G(x_after_excursion)=m').
```

Sie definiert einen policyabhängigen Basin-Graphen. Gateway-Basins, stark
absorbierende Basins und wiederkehrende Übergänge werden darin sichtbar.

Der bisherige selektierte n=52-Q1-Pilot ergab ungefähr:

| absolute Q-Decke | Rückkehr zum exakten Ausgangsminimum |
| ---: | ---: |
| 8 | 78 % |
| 12 | 20 % |
| 16 | unter 1 % |

Das ist ein starkes erstes Signal für einen Return Transition. Es ist wegen
der 109 ausgewählten Ausgangszustände noch keine allgemeine Skalierungsaussage.

Der anschließende Multi-n-Test verwendete 896 verschiedene Greedy-Minima bei
`n=28,32,...,52` und je acht Replikate. Bei 64 Walk-Schritten ergaben relative
Decken `Q_root+2,+4,+8,+16` bedingte Rückkehrraten von ungefähr
`8,0 %, 2,0 %, 0,03 %, 0 %`. Exakte State- und Class-Returns stimmten in allen
Fällen überein. Die Transition existiert damit nicht nur in der selektierten
n=52-Stichprobe; ihre genaue Lage hängt aber von Root-Verteilung, Quench und
Walk-Policy ab.

Höhe allein genügt nicht. Bei einem einzigen Exkursionsschritt stieg die
Rückkehr von `45 %` bei `+1` auf `74 %` bei `+4`: ein höherer Einzelschritt
rollt häufig denselben Hang zurück. Ab vier erlaubten Schritten lagen die
Rückkehrraten nur noch bei `14 %, 8 %, 3 %, 3 %`; längere Walks änderten sie
kaum. Operativer Escape benötigt daher **Passhöhe plus laterale Bewegung**.

## 5. Sensoren

| Sensor | Misst | Grenze |
| --- | --- | --- |
| `Q` | aktuelle Fehlerhöhe | keine Basin-Richtung |
| konkrete `d_i` | sofortige Move-Richtungen | nur aktuelles Wörterbuch |
| `d_i=-u` | sichtbarer Lösungsmove | extrem selten |
| `b_H(i)` | Zwei-Schritt-Verzweigung nach Move `i` | Topologie, nicht Solve-Nähe |
| vollständiger Quench `G(x)` | operative Basin-ID | relativ teuer und G-abhängig |
| Paarseparator `w` | Verteilung auf zwei Folgenpaare | ` | | w | | ` allein ungeeignet |
| `Q(F(x))` | Energie des Tripleprodukt-Duals | bevorzugt starken Unterraum |
| Basin-Class-Hash | Symmetrieklasse des Minimums | keine vollständige Hadamard-Äquivalenz |

Ein Sensor ist nur dann neu, wenn er nicht vollständig aus `u` oder `D^T D`
rekonstruierbar ist. Framepotential, aggregiertes Spektrum und mittlere
Delta-Norm sind daher keine unabhängigen Lidar-Signale.

Der prospektive Zwei-Schritt-Grad wurde auf dem Multi-n-Datensatz unabhängig
auditiert. Bei `+2` verbesserte er gegenüber `n`, Root-`Q` und Entry-Count den
Leave-one-n-out-Log-Loss von `0,2078` auf `0,1706` und die Within-n-AUC von
`0,830` auf `0,888`; die Verbesserung war in allen sieben Dimensionen positiv.
Bei `+4` war der Zusatzwert klein und instabil. Der Sensor enthält somit echte
lokale Topologieinformation, aber bislang keine belegte Solve-Information.

## 6. Aktoren

| Aktor | Wirkung |
| --- | --- |
| Single-Flip | lokale Basisänderung in einer Spalte |
| Hamming-2-Symbolmove | lokale Bewegung innerhalb einer H4-Basis |
| Hamming-3/4-Symbolmove | weitere lokale Symbolkanäle |
| Tabu | zustandsabhängige, erinnerungsbehaftete Uphill-Exkursion |
| Q-Shell-Walk | explizite Höhenkontrolle |
| Targeted Escape | großer Ziellag-konditionierter Basinsprung |
| Tripleprodukt-Dual `F` | involutiver Fernsprung bei fester Basis/Achse |
| strukturierter Warmstart | Eintritt aus Paley, BS, TT oder Strong Split |

Der Dualsprung negiert alle vier Bits genau an Spalten mit
`beta=a*b*c*d=-1`. Auf den 109 n=52-Q1-Pilotzuständen änderte er im Mittel
etwa 104 Bits. Der direkte Zustand lag bei mittlerem `Q≈298`; ein anschließender
Greedy-Quench endete immer in einem anderen Minimum bei `Q=6..19`, aber nie in
einer Lösung. Das ist ein billiger bestätigter Fernaktor, noch kein
erfolgreicher Navigator.

Ein einzelner Hamming-4-Spaltenmove erreichte bei denselben Zuständen niemals
`Q<=16`. Der lokale Orientierungsunterraum und der globale Dualsprung verhalten
sich damit sehr verschieden.

## 7. Navigationsregelkreis

```text
1. Quench:      m = G(x)
2. Sondieren:   lokale Sensoren bei mehreren möglichen Höhen
3. Steuern:     kleinste plausible Escape-Höhe/Aktor wählen
4. Exkursion:   x' ~ K(m,...)
5. Re-Quench:   m' = G(x')
6. Lernen:      Return, Übergang, Solve und Kosten aktualisieren
```

Der zentrale Optimierungswert ist nicht bloß `Q(x')`, sondern der erreichte
Quench-Endpunkt `G(x')` bei gegebener Arbeit. Ein Aktor ist nützlich, wenn er
bei gleichem Budget häufiger neue oder lösungsnähere operative Basins erreicht.

Ein erster kostenbelasteter Regler verglich fest `+2`, fest `+4` und eine aus
dem Zwei-Schritt-Sensor gewählte Höhe. Der Regler senkte die Rückkehrquote von
`9,8 %` auf `4,6 %`, war bei tieferen Basins pro Million Auswertungen aber
schlechter als fest `+2` (`61` statt `66`; fest `+4`: `50`). Das zeigt
präzise, dass **Escape-Wahrscheinlichkeit und Zielqualität verschiedene
Regelgrößen sind**. Der aktuelle Lidar erkennt eine dünne Basin-Wand, aber noch
nicht die Richtung zu einem besseren Attraktor.

Bei gleichem Candidate-Budget fanden wiederholte Two-Bit-Kick-plus-Quench-
Versuche deutlich häufiger ein tieferes Minimum als eine einzelne `+4`-Shell.
Der Vergleich enthält jedoch einen Best-of-many-Vorteil und etwa dreifache
Walltime. Er belegt einen brauchbaren Mehrlandepunkt-Aktor, noch keine
überlegene Two-Bit-Geometrie.

Die direkte Sensor-Aktor-Kopplung ist stärker: Ein `+4`-Walk sampelt an jedem
Schritt acht erlaubte Moves und wählt den mit dem größten prospektiven
Zwei-Schritt-Grad. Auf je 256 unabhängigen Roots stieg die Zahl tieferer
Zielbasins gegenüber zufälliger Wahl von `98 auf 133` (`n=40`), `101 auf 158`
(`n=44`), `100 auf 155` (`n=48`) und `100 auf 169` (`n=52`). Die gepaarten
exakten p-Werte lagen zwischen `4,9e-4` und `8,2e-13`; das mittlere End-`Q`
sank bei allen vier Größen. Dafür benötigt der unoptimierte Sensor etwa
10- bis 12-mal so viele Kandidatenbewertungen. Es gab noch keine Lösung.

Damit ist erstmals belegt:

```text
lokale Sublevel-Topologie -> geänderter Pfad -> häufiger tieferer Attraktor.
```

Das ist echte Navigation, aber noch kein Solve-Kompass. Der nächste Sensor
muss die Qualität des Zielattraktors beziehungsweise seine Nähe zur
Cancellation-Grenze beschreiben.

## 8. Kanonisches Experimentdesign

Künftige Basin-Experimente müssen berichten:

- mehrere `n` und unabhängig erzeugte lokale Minima;
- Root-Erzeugung und Quench `G` einschließlich Tie-Break;
- Exkursionspolicy `K`, Höhe, Länge, Tabu und Move-Menge;
- exakte und symmetriekanonisierte Return-Rate;
- Übergangsmatrix beziehungsweise Zahl verschiedener Zielbasins;
- Solve-Rate und Candidate-/Wall-Time-Budget;
- gepaarte Seeds und Konfidenzintervalle;
- getrennte Auswertung pro Root, damit Replikate nicht als unabhängige Minima
  gezählt werden.

Der nächste Datensatz verwendet frische Zufallsstarts bei mehreren Größen,
quencht sie zu unabhängigen Minima und misst relative Passhöhen
`Q_root + {2,4,8,16}`. Damit wird geprüft, ob die n=52-Rückkehrkurve eine
allgemeine Dimensionsentwicklung oder Auswahlbias war.

Best- und First-Improvement wurden auf 192 neuen Roots bei `n=32,40,52`
verglichen. Bei `+2` änderte sich die Rückkehr von `10,0 %` auf `6,0 %`, bei
`+4` blieb sie praktisch gleich (`1,37 %` gegen `1,30 %`). Enge operative
Basins hängen sichtbar vom Quench ab; der grobe Escape oberhalb `+4` ist in
diesem Test robuster.

## 9. Einordnung

Die Begriffe sind in der allgemeinen Energielandschaftsliteratur bekannt:
inherent structures, attraction basins, disconnectivity graphs,
communication heights und committor functions. Projektspezifisch ist ihre
Kopplung an das exakte GS4-NAF-Wörterbuch und die verfügbaren strukturierten
Aktoren.

Ein Committor ist erst nach Festlegung eines Markov-Kerns definiert. Für das
Production-Tabu müsste der Zustand um Tabu-Gedächtnis und gegebenenfalls RNG
erweitert werden. Die endliche Shell-Exkursion plus Quench ist verwandt, aber
nicht identisch mit einem Committor.

Primärquellen und die genaue mathematische Abgrenzung stehen im unabhängigen
Audit `experiments/learning/BASIN_DEFINITION_AUDIT.md`. Die Integration mit
Cancellation, F-Dualität und den NAF-Kanälen steht in
`experiments/learning/NAVIGATION_SYNTHESIS.md`.
