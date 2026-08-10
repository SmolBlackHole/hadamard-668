# Audit: Reverse-Curriculum ab exakten GS4-Lösungen

## Kurzfassung

Alle `1.327` im Archiv bis `n=52` verfügbaren, SHA-eindeutigen Lösungen plus
die Golay-Fold-Referenz wurden untersucht. Für jeden Zustand wurden sämtliche
`4n` Single-Nachbarn sowohl aus dem Tracker-Cache als auch durch einen frischen
Tracker berechnet.

Die wichtigsten Ergebnisse sind:

1. Große freie Lösungen besitzen in ihrer Single-Schale keine `Q=1`-Zustände.
   Bei den archivierten freien `n=44..52`-Lösungen waren `0/64.056` Nachbarn
   bei `Q=1`. Das erklärt exakt, warum diese Lösungen nicht direkt aus einem
   `Q=1`-Zustand betreten werden können.
2. Kleine Symbolsolver-Lösungen besitzen solche Eingänge noch: bei `n=28` gab
   es `53`, bei `n=30` `15`, bei `n=32` `8` direkte `Q1 -> Q0`-Kanten.
3. Der vollständige Two-ply-Lookahead erkennt bei Drei-Flip-Schäden in
   `126/136` Fällen mindestens einen bekannten Rückweg als global beste erste
   Aktion. Mit dem realen Achter-Tournament wurde ein Rückweg im Mittel in
   `12,82 %` der Entscheidungen gewählt, gegenüber `1,89 %` bei einem zufälligen
   First-Move: ungefähr Faktor `6,8`.
4. `rho_s`, `u(Fx)` und grobe `D`-Row-Aggregate erkannten die exakte
   Rückwegdistanz nicht. Ihre cross-source Nearest-Neighbor-Genauigkeit lag bei
   `29,7–32,6 %`, also um die Zufallsbasis von `1/3`.

Damit ist der Reverse-Curriculum-Lead substanziell, aber eng: `lookahead_q`
erkennt lokale Rückweggeometrie nahe bekannter Lösungen. Noch nicht gezeigt ist,
dass dieselbe Richtung von unbekannten Production-Minima auf eine Lösung weist.

## 1. Datensatz und Methodik

Aus `data/solutions.json` wurden identische SHA-Zustände über Strategien hinweg
entfernt und nach Suchprovenienz gruppiert:

- `free-bit`: Lösungen des großen GS4-Solvers;
- `free-symbol`: Lösungen der experimentellen Symbolsolver;
- `paley`: direkte Paley-NG-Lösung;
- `golay-fold`: separat erzeugte Golay-Fold-Referenz.

`free-bit` und `free-symbol` sind keine Behauptung mathematischer
Äquivalenzklassen, sondern nur Herkunftslabels.

Für jede Lösung `s` wurde für alle Single-Flips `i` berechnet:

\[
Q(s\oplus i)=\lVert d_i(s)\rVert^2,
\]

weil `u(s)=0`. Jeder dieser Werte wurde durch vollständigen Rebuild des
geflippten Zustands bestätigt.

Für das outward Curriculum wurden pro `(Familie,n)` bis zu vier Lösungen
gewählt. Je Lösung entstanden vier zufällige, nicht bereits gelöste Zustände
in Hamming-Tiefe `1`, `2` und `3`. Insgesamt sind das `408` Zustände. Der
absichtlich verwendete Rückweg wurde durch erneutes Flippen geprüft.

Die minimale Lösungsdistanz bis drei wurde nicht aus diesem bekannten Pfad
angenommen:

- alle Singles wurden auf `Q=0` geprüft;
- falls kein Single löste, wurden alle geordneten Zwei-Flip-Endpunkte geprüft;
- wenn beides fehlschlug, garantierte der bekannte Drei-Flip-Rückweg Distanz
  drei.

Der resultierende Datensatz ist exakt balanciert: jeweils `136` Zustände mit
Lösungsdistanz `1`, `2` und `3`.

## 2. Exakte Single-Schalen

| Provenienz | n | Lösungen | Nachbarn | silent Q0 | Q1 | mittleres Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| free-symbol | 28 | 526 | 58.912 | 3 | 53 | 6,5 |
| free-symbol | 30 | 321 | 38.520 | 0 | 15 | 7,0 |
| free-symbol | 32 | 127 | 16.256 | 0 | 8 | 7,5 |
| free-bit | 44 | 233 | 41.008 | 0 | 0 | 10,5 |
| free-bit | 46 | 13 | 2.392 | 0 | 0 | 11,0 |
| free-bit | 48 | 62 | 11.904 | 0 | 0 | 11,5 |
| free-bit | 50 | 24 | 4.800 | 0 | 0 | 12,0 |
| free-bit | 52 | 19 | 3.952 | 0 | 0 | 12,5 |
| Golay-Fold | 52 | 1 | 208 | 0 | 0 | 12,5 |
| Paley | 52 | 1 | 208 | 2 | 0 | 12,5 |

### Direkte Konsequenz für die Q1-Wand

Der Single-Flip-Graph ist ungerichtet. Ein Zustand bei `Q=1` kann genau dann
mit einem Single in eine konkrete Lösung gehen, wenn diese Lösung einen
Single-Nachbarn bei `Q=1` besitzt.

Für alle archivierten freien Lösungen ab `n=44` ist diese Zahl null. Ein
`Q1 -> Q0`-Single ist für diese Zielpunkte daher nicht nur selten, sondern
unmöglich. Das beobachtete „erst uphill, dann lösen“ ist mit ihrer lokalen
Schale zwingend vereinbar.

Das Resultat ist keine Aussage über sämtliche möglichen GS4-Lösungen dieser
Längen. Es gilt exakt für alle derzeit archivierten Repräsentationen.

### Symmetrie der Histogramme

Alle aggregierten Histogramme sind um

\[
\frac{n-2}{4}
\]

zentriert. Archivweit galt für gegenüberliegende Spalten `c` und `c+n/2`
derselben Folge sogar numerisch exakt

\[
Q_{s,c}+Q_{s,c+n/2}=\frac n2-1.
\]

Beispiele:

- freie `n=52`: Werte `5..20`, paarweise Summe `25`;
- Golay-Fold `n=52`: Werte `8..17`, paarweise Summe `25`;
- Paley `n=52`: zwei silent Moves bei `0`, gepaart mit zwei Moves bei `25`;
- freie Symbolzustände `n=28`: Q1- und Q12-Häufigkeit beide `53`.

Der konstante Mittelwert ist mit der bereits bekannten konstanten mittleren
Zeilennorm des Flip-Wörterbuchs konsistent. Die stärkere antipodale
Paaridentität wurde an allen `1.327` Lösungen bestätigt, in diesem Audit aber
nicht symbolisch bewiesen.

## 3. Welche Features Rückwegdistanz erkennen

Die folgende Rangkorrelation verwendet die exakte Distanz `1/2/3`:

| Feature | Spearman mit Rückwegdistanz | Einordnung |
| --- | ---: | --- |
| `Q` | +0,757 | stark, aber Damage-Tiefe ist ein Confounder |
| bestes Single-Q | +0,886 | Distanz 1 wird per Definition exakt erkannt |
| bestes Two-ply-Q | +0,333 | erkennt Distanz 2 exakt durch Wert null |
| mittlerer prospective degree bei `Q+4` | +0,730 | größtenteils mit Q gekoppelt |
| maximaler prospective degree | -0,123 | nicht monoton, aber bedingt informativ |
| ` | | rho_s | | ²` | +0,036 | kein Signal |
| Verteilung der vier `rho_s`-Energien | +0,096 | sehr schwach |
| ` | | u(Fx) | | ²` | +0,047 | kein Signal |
| Zahl verschiedener D-Zeilen | +0,031 | kein Signal |
| D-Zeilen-Multiplizität/-Normgrenzen | ungefähr 0 | kein Signal |

Nach linearer Kontrolle für `(Q,n)` blieb beim mittleren prospective degree
nur Korrelation `+0,210`. Der maximale prospective degree zeigte `-0,616` und
ist damit der einzige nichttriviale skalare Lead dieser Stichprobe. Er muss auf
Q-gematchten unabhängigen Roots bestätigt werden; die Reverse-Zustände sind
absichtlich lösungsnah und daher kein neutraler Datensatz.

Die vollen Vektoren überzeugten ebenfalls nicht. Ein cross-source
Nearest-Neighbor, der niemals einen Zustand derselben Ursprungslösung als
Nachbarn verwenden durfte, erreichte:

| Vektor | Accuracy für Distanz 1/2/3 |
| --- | ---: |
| volle vier `rho_s` | 32,6 % |
| `u(Fx)` | 29,7 % |
| D-Zeilennorm-Histogramm | 32,2 % |

Die Zufallsbasis ist `33,3 %`. Alle `408` vollständigen D-Row-Multiset-Hashes
waren einzigartig. Ein Hash identifiziert den Zustand, liefert allein aber
keine generalisierbare Distanzordnung.

## 4. Warum Two-ply den bekannten Rückweg sieht

Für einen drei Flips beschädigten Zustand

\[
x=s\oplus\{a,b,c\}
\]

führt ein bekannter Reverse-First-Move, beispielsweise `a`, nach einem zweiten
bekannten Reverse-Move `b` zu

\[
s\oplus\{c\},
\]

also exakt auf die Single-/Cancellation-Grenze der Lösung. Deshalb gilt

\[
V_2(a)=\min_j Q(x\oplus a\oplus j)
\]

höchstens gleich der günstigsten verbleibenden C1-Schalenenergie. Two-ply ist
damit der erste getestete Sensor, der die bekannte Rückwegstruktur nicht nur
korreliert, sondern geometrisch direkt abbildet.

Auf den `136` Zuständen mit exakter Rückwegdistanz drei:

- enthielt die globale Bestmenge in `126/136 = 92,6 %` mindestens einen der
  drei bekannten Reverse-First-Moves;
- bei zufälliger Auswahl innerhalb der globalen Best-Ties lag die mittlere
  Trefferwahrscheinlichkeit bei `81,5 %`;
- ein Achter-Tournament traf einen Reverse-Move in `12,82 %` der Fälle;
- ein vollständig zufälliger First-Move hätte nur `1,89 %` getroffen.

Für Tiefe zwei ist das Resultat tautologisch perfekt: Beide Reverse-Moves
besitzen einen Zwei-Flip-Endpunkt bei `Q=0`. Für Tiefe eins ist Two-ply das
falsche Werkzeug; der Single-Scan erkennt den direkten Solve bereits exakt.

## 5. Was daraus noch nicht folgt

Der Befund beweist nicht, dass `lookahead_q` von einem beliebigen Q1-Minimum
oder einem Production-Root in Richtung irgendeiner unbekannten Lösung zeigt.
Das Curriculum kennt Zustände, die absichtlich nur drei Bits von einer Lösung
entfernt erzeugt wurden. Es zeigt:

> Der Lookahead kann eine vorhandene lokale C1-Rückwegstruktur lesen, wenn sie
> im Radius zwei seiner ersten Aktion sichtbar wird.

Es zeigt noch nicht:

> Tiefe Two-ply-Werte bedeuten allgemein kleine Communication Height oder hohe
> Solve-Hazard.

Der nächste faire Test sollte deshalb Zustände gleicher `(n,Q)` mischen:

1. Reverse-Curriculum-Zustände mit bekanntem Rückweg;
2. Production-Minima ohne bekannten Rückweg;
3. Shell-Zustände aus echten Uphill-Exkursionen.

Dann wird gemessen, ob `V2`, maximaler prospective degree oder ein expliziter
„Nähe zur beobachteten C1-Histogrammform“-Score zwischen diesen Gruppen trennt
und downstream bei gleichem Candidate-Budget mehr Lösungen erzeugt.

## Reproduktion

```text
python -m experiments.learning.reverse_curriculum_audit \
  --output data/reverse-curriculum-audit.json \
  --sources-per-group 4 --paths-per-depth 4 \
  --tournament-replicas 64 --seed 20260815
```

Implementierung:
`experiments/learning/reverse_curriculum_audit.py`

Daten:
`data/reverse-curriculum-audit.json`
