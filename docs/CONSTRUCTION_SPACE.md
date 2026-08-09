# Konstruktionsraum: Paley, Golay, GS4, Turyn und Tensor

Stand: 2026-08-09

## Kurzfassung

Paley/Ito, Golay, der freie Solver und der GS4-Builder lösen nicht vier
verschiedene Probleme. Alle erzeugen Folgen, deren summierte negaperiodische
Autokorrelation verschwindet. Der Unterschied ist, wie sie diesen Zustand
erreichen:

- Paley/Ito konstruiert ein komplementäres Folgenpaar algebraisch.
- Golay liefert komplementäre Paare und strukturtreue Multiplikatoren.
- Der Solver sucht ein allgemeines komplementäres Vierertupel.
- GS4 übersetzt jedes solche Vierertupel in eine Hadamardmatrix.

Der praktisch wichtigste Fund ist eine komprimierte Produktoperation: Ein
kleines GS4-Tupel kann mit einem gewöhnlichen Golay-Paar direkt in ein größeres
GS4-Tupel überführt werden. Die vier Folgen bleiben erhalten; eine große
Kronecker-Matrix muss nicht als Zwischenzustand gespeichert werden.

## 1. Gemeinsame Gleichung

Für die Folgenpolynome `X_s(z)` lautet die GS4-Bedingung

```text
sum_s X_s(z) X_s(z^-1) = 4n  mod (z^n + 1).
```

Dies ist exakt dieselbe Aussage wie `sum_s NAF_s(t)=0` für alle nichttrivialen
Lags. Der Tracker misst die Abweichung von dieser Gleichung, der Builder setzt
eine exakte Lösung in die GS4-Blockmatrix ein.

Ein negaperiodisches Golay-Paar `(a,b)` erfüllt die stärkere Zweierbedingung

```text
NAF_a(t) + NAF_b(t) = 0.
```

Darum ist `(a,b,a,b)` automatisch eine GS4-Lösung. Paley/Ito und die
Golay-26-Faltung bei `n=52` landen beide in dieser kleineren Unterfamilie. Der
freie Solver ist allgemeiner: Seine beiden Teilpaare müssen einzeln nicht
komplementär sein.

## 2. Exakte Spaltenkoordinate: zwei orthonormale Basen

Fasse die vier Folgenwerte an Position `c` als Spalte

```text
x_c = (a_c,b_c,c_c,d_c)^T
```

zusammen und verwende die Hadamardmatrix

```text
H4 = [[1, 1, 1, 1],
      [1,-1, 1,-1],
      [1, 1,-1,-1],
      [1,-1,-1, 1]].
```

Die transformierte Spalte

```text
v_c = H4 x_c / 4
```

hat immer Norm eins. Da `x_c^T x_d = 4 v_c^T v_d`, ist das reduzierte
Trackerresiduum exakt die summierte negaperiodische Vektorautokorrelation:

```text
u_t = sum_c sign(c,t) * <v_c,v_(c+t mod n)>.
```

Das Alphabet der 16 binären Spalten zerfällt dabei in zwei Mengen. Mit

```text
beta_c = a_c b_c c_c d_c
b_c    = (1-beta_c)/2
O      = (J-2I)/2
```

gilt:

- `beta_c=+1`: `v_c` ist einer der acht signierten Koordinatenvektoren
  `+-e_j`;
- `beta_c=-1`: `v_c` ist einer von acht signierten Vektoren mit Einträgen
  `+-1/2`.

Jede Menge besteht aus vier antipodalen Paaren und bildet eine orthonormale
Basis. Zwischen den beiden Basen hat jedes Skalarprodukt Betrag `1/2`; sie
sind also gegenseitig unbiased. Jede GS4-Spalte besitzt damit die exakte
Koordinate

```text
v_c = eta_c * O^b_c * e_(j_c)
```

aus Basiswahl, Achsenindex und Vorzeichen. Diese Darstellung spart allein
keine Bits, trennt aber zwei bisher vermischte Entscheidungen.

Der vorhandene `base_to_gs4`-Pfad ist die inverse Abbildung einer besonders
strukturierten achsenreinen Folge: Seine ternäre Eingabespalte hat genau einen
nichtverschwindenden Eintrag `t_c`, danach gilt `x_c=H4 t_c` und folglich
`v_c=t_c`. Die zusätzlichen Blocksupport- und aperiodischen Bedingungen machen
daraus eine klassische Base-/T-Sequence-Einbettung. Achsenrein allein ist
nicht mit einer klassischen aperiodischen T-Sequence gleichbedeutend.
Paley-/Golay-Tupel `(a,b,a,b)` benutzen sogar nur zwei Achsen.

Die gespeicherten freien GS4-Repräsentationen verhalten sich anders. Bei den
vorhandenen Lösungen für `n=44,46,48,50,52` ist keine achsenrein; die 19
`n=52`-Repräsentationen enthalten jeweils 17 bis 33 Positionen aus der zweiten
Basis. Keine davon wird durch unabhängige negazyklische Verschiebungen und
Reversals der vier Folgen achsenrein; im vollständig geprüften
Repo-Symmetrieorbit blieben mindestens 6 bis 10 Positionen in der
Minderheitsbasis. Das beweist nur, dass sie außerhalb dieser konkreten
Parametrisierung und dieses Orbits liegen. Es beweist weder neue
Hadamard-Äquivalenzklassen noch wissenschaftlich unbekannte Familien.
Auch die bloße Anzahl der Symbole je Basis ist kein Invariant: Ein
unabhängiger Negashift kann sie verändern. Belastbarer ist die vollständige
Prüfung des Shift-Orbits oder die paarweise Residualgröße

```text
J = min over pairings ||rho_i + rho_j||^2.
```

Paley-/Golay-Paare haben `J=0`; die 19 freien `n=52`-Repräsentationen lagen
zwischen 640 und 1344. Auch das trennt nur die beobachtete Zweierfaktorisierung
vom freien Vierertupel, nicht vollständige Hadamard-Äquivalenzklassen.

Algebraisch ist dies die zentrale Differenz der n=52-Wand. Setze

```text
S_n = { NAF(x) : x in {+-1}^n }.
```

Paley/Golay liefern bereits `0 in S_n + S_n`. Der freie GS4-Solver sucht im
allgemeineren Problem `0 in S_n + S_n + S_n + S_n`. Die Existenz einer
dünnen Zweierfaktorisierung macht die lokale Navigation im viel größeren
Vierersummenraum nicht automatisch leicht.

Das Basislabel allein liefert bisher keine verborgene Existenzbedingung. Eine
vollständige Enumeration für `n=2..6` fand unter exakten GS4-Lösungen jedes der
`2^(n-1)` normalisierten Basiswörter. Umgekehrt erzeugte das vollständige
spaltenweise Umschalten fester Paley-Achsenlösungen bei `n=10,12,16` nur die
beiden konstanten Basiswahlen. Ein nichttrivialer Konstruktionsweg muss daher
Basiswahl, Achsenindex und Vorzeichen gemeinsam kontrollieren.

Die neue Koordinate liefert eine natürliche lokale Nachbarschaft. An einer
Spalte existieren nur 15 andere Symbole:

- zwei oder vier Bitänderungen erhalten die Basiswahl;
- eine oder drei Bitänderungen wechseln die Basis.

Da dabei höchstens ein Bit je Folge geändert wird, ist das Residualdelta exakt
die Summe der betroffenen Single-Deltas; Same-Sequence-Paarkorrekturen treten
nicht auf. Alle `15n` Symbolersetzungen lassen sich daher gemeinsam und billig
auswerten. Ob diese kategorielle Nachbarschaft die Skalierungswand verschiebt,
ist noch nicht getestet.

Der kleinste sinnvolle Hybrid muss nicht sofort alle 15 Ersetzungen verwenden.
Die heutigen vier Singles plus die sechs basis-erhaltenden Zwei-Bit-Moves je
Spalte ergeben eine global verbundene `10n`-Nachbarschaft. Sie ergänzt genau
die bisher fehlende direkte Bewegung innerhalb einer Basis. Triple- und
Vierfachflips sind zunächst nur Stagnations-Makromoves; der volle `15n`-Scan
bleibt eine kontrollierte Ablation.

### Paley-Parität

Der aktuelle Sequenzpfad `paley-ng` gilt für gerades `n` mit primem
`q=2n-1`, also `q=3 mod 4`. Er liefert über die zweite Paley/Ito-Reihe ein
negaperiodisches Golay-Paar und damit unmittelbar ein GS4-Tupel.

Für ungerades `n` gilt dagegen `q=1 mod 4`. Die klassische Paley-Type-II-
Konstruktion liefert bei Primzahlpotenz `q` direkt eine Hadamardmatrix der
Ordnung `2(q+1)=4n`. Das schließt die Matrixexistenz konstruktiv, liefert aber
nicht automatisch vier Folgen für den vorhandenen GS4-Tracker. Ein künftiger
Type-II-Pfad gehört deshalb in den Matrixdispatcher, nicht stillschweigend in
`paley-ng`. Referenzen: [Paley-Type-I/II-Einordnung](https://doi.org/10.1007/s10801-021-01033-x)
und [Erweiterung der Paley-Konstruktion](https://arxiv.org/abs/1912.10755).

## 3. Paley und Golay lassen sich direkt mischen

Sind `(p1,p2)` und `(g1,g2)` zwei NG-Paare derselben Länge, dann ist

```text
(p1, p2, g1, g2)
```

ohne Suche eine GS4-Lösung. Für die Paley- und Golay-Lösungen bei `n=52` wurde
dies mit Tracker, Builder und unabhängigem Audit als gültige `H208` geprüft.
Der gemischte Zustand besitzt außerdem einen neuen Projekt-Klassenhash. Das
beweist eine neue Repräsentation im Projekt, aber noch keine Inequivalenz unter
allen Hadamard-Äquivalenzoperationen.

## 4. Turyn-Multiplikation als komprimierter Produktweg

### Literatur

Balonin und Djokovic erweitern Turyns Multiplikation zu

```text
gewöhnliches Golay-Paar der Länge g
× negaperiodisches Golay-Paar der Länge n
→ negaperiodisches Golay-Paar der Länge gn.
```

Quelle: [Negaperiodic Golay pairs and Hadamard matrices](https://arxiv.org/abs/1508.00640),
insbesondere Proposition 3 und Gleichungen (15), (16).

### Exakte Projektfolgerung

Dieselbe Normidentität kann getrennt auf `(x1,x2)` und `(x3,x4)` eines
beliebigen GS4-Tupels angewendet werden. Daher gilt

```text
Golay_g × GS4_n → GS4_(gn).
```

Die beiden Eingangspaare müssen einzeln nicht komplementär sein. Das wurde mit
allgemeinen Solverlösungen geprüft, deren Teilpaare nichtverschwindende
Residualvektoren besitzen.

Für den kleinsten Golay-Faktor `g=2` ist die Abbildung besonders einfach. Aus
einem Paar `(c,d)` der Länge `n` wird:

```text
e[2j]   =  c[j]
e[2j+1] = -d[n-1-j]
f[2j]   =  d[j]
f[2j+1] =  c[n-1-j].
```

Die Formel wird auf beide Paare des GS4-Tupels angewendet. Damit kann der
Solver eine leichte Basisgröße lösen und das Ergebnis anschließend exakt
verdoppeln.

## 5. Verifizierte neue Projektabdeckung

Alle folgenden Ergebnisse bestanden Tracker, GS4-Builder und den unabhängigen
Integer-Audit:

| Ziel-`n` | Basis | Konstruktionsweg | Matrix |
| ---: | ---: | --- | ---: |
| 50 | Solver `n=25`, Seed 0 | Turyn ×2 | H200 |
| 56 | Solver `n=28`, Seed 0 | Turyn ×2 | H224 |
| 56 | Paley-Primzahlpotenz `n=14`, `q=27` | Turyn ×4 | H224 |
| 58 | Solver `n=29`, Seed 0 | Turyn ×2 | H232 |
| 60 | Paley `n=30`, `q=59` | Turyn ×2 | H240 |
| 62 | Solver `n=31`, Seed 0 | Turyn ×2 | H248 |

Der zuvor offene Block `50,56,58,60,62` muss damit nicht mehr in seiner großen
Form durchsucht werden. Der Solver und die algebraische Konstruktion ergänzen
sich tatsächlich: Der Solver liefert kleine, auch unsymmetrische Basen; Turyn
hebt sie exakt in größere GS4-Längen.

`n=167` wird dadurch nicht direkt erreicht. Die Zielgröße ist ungerade, besitzt
keinen nichttrivialen binären Golay-Faktor und `2n-1=333` ist keine
Primzahlpotenz.

## 6. Verhältnis zum bisherigen Tensorpfad

Der Repo-Tensorpfad berechnet nach zwei unabhängigen Suchen die volle Matrix

```text
H = H1 ⊗ H2.
```

Das ist ein universeller Hadamard-Abschluss, verliert aber die vier
GS4-Sequenzen. Der Tracker kann das Ergebnis daher nicht direkt weiterführen.

Eine komponentenweise Sequenzformel `z_s=x_s⊗y_s` ist falsch: Das geprüfte
Beispiel Paley `n=4` mal Paley `n=6` endet bei `Q=80`. Auch
`GS4(a,b,a,b)` ist im Allgemeinen nicht bloß `H2⊗K(a,b)`; exakte
Smith-Normalformen unterscheiden sich bereits bei `n=4` und `n=12`.

Turyn ist deshalb die derzeit korrekte strukturtreue Produktoperation. Sie ist
nicht für zwei beliebige GS4-Faktoren definiert, benötigt aber nur einen
gewöhnlichen Golay-Faktor und liefert wieder vier kompakte Folgen.

## 7. Beschädigen und Schrumpfen

Ein gepaarter `n=52`-Recovery-Test mit 100.000 Schritten ergab:

| beschädigte Bits | Paley | Golay-Fold | freie Solverlösung |
| ---: | ---: | ---: | ---: |
| 1 | 20/20 | 15/20 | 16/20 |
| 2 | 14/20 | 8/20 | 8/20 |
| 4 | 2/20 | 1/20 | 1/20 |
| 8, 16 oder 32 | 0/20 | 0/20 | 0/20 |

Paley zeigt bei ein bis zwei Fehlern einen positiven, mit dieser Stichprobe
noch nicht signifikanten Trend. Ein außergewöhnlich großes algebraisches
Einzugsgebiet ist nicht belegt.

Beim Schrumpfen der drei `n=52`-Familien auf `n=51` wurden 0/180 Zustände
repariert, auf `n=50` nur 3/180. Die neue Länge ändert den Modulus
`z^n+1` und damit alle Residuen gleichzeitig. Schrumpfen bleibt eine
Warmstart-Hypothese, ist aber der exakten Produktkonstruktion klar unterlegen.

## 8. Produktionsdispatcher

Die Strategie `construct` arbeitet in dieser Reihenfolge:

1. direkte Paley-/Primzahlpotenz-Konstruktion prüfen;
2. Golay-Teiler des Ziel-`n` suchen und nur die kleinere GS4-Basis lösen;
3. die Basis mit Turyn exakt liften;
4. erst wenn kein Konstruktionsweg greift, den großen freien Solver starten.

Die Primzahlpotenz-Erweiterung ist noch offen. Interessante Solver-Hypothesen
danach sind `lift-and-repair` für noch nicht
exakte Basen und Turyn-Makromoves, bei denen ein Basisflip als strukturierter
Mehrbit-Move im Zielraum erscheint. Beides ist noch ungetestet.

## 9. Verifikation

Die produktiven Konstruktionsidentitäten sind durch Tracker-, Builder- und
unabhängige Integer-Audits in `tests/` abgesichert. Wegwerfprüfer und ihre
abgeleiteten Rohdaten wurden nach Übernahme der Resultate entfernt.

## 10. Ein gemeinsamer aperiodischer Suchraum für Base und TT

Base Sequences und Turyn-Type-Sequences lassen sich mit demselben ganzzahligen
Residualmodell schreiben:

```text
r_t = sum_s w_s * AF_s(t)
Q   = sum_t r_t^2.
```

Nur Längen und Gewichte unterscheiden sich:

| Raum | Längen | Gewichte | Bits | Residuen |
| --- | --- | --- | ---: | ---: |
| `BS(84,83)` | `(84,84,83,83)` | `(1,1,1,1)` | 334 | 83 |
| `TT(56)` | `(56,56,56,55)` | `(1,1,2,2)` | 223 | 55 |

Der produktive `BaseTracker` bildet dieses gewichtete Modell exakt ab. Sein
Single-Flip-Cache wurde für beide Gewichtsvarianten gegen vollständige
Neuberechnung geprüft. Ein Solver für diesen Raum ist noch nicht integriert.

Nach Best, Djokovic, Kharaghani und Ramp führt ein `TT(56)` über
`BS(111,56)` zu einer Hadamardmatrix der Ordnung 668. Dies ist ein bekannter
offener Konstruktionsweg, kein Existenzbeweis für `TT(56)`:
[Turyn-type sequences: Classification, Enumeration and Construction](https://arxiv.org/abs/1206.4107).

Base Sequences besitzen additive und multiplikative Kompositionen, aber nicht
als beliebige Addition zweier fertiger Base Sequences:

- Zwei gewöhnliche Golay-Paare der Längen `m` und `n` ergeben unmittelbar ein
  `BS(m,n)`. Die Gesamtlänge ist dann additiv `m+n`.
- Yang-/Lagrange-Kompositionen erzeugen größere Base- oder T-Sequences mit
  produktartigen Längenformeln und zusätzlichen Voraussetzungen.
- Für das Primziel `m+n=167` liefert die bekannte Golay-Längenfamilie derzeit
  keine passende additive Zerlegung; eine nichttriviale reine
  Produktzerlegung wird durch die Primzahl 167 blockiert.

Der aperiodische Raum bleibt deshalb ein echter Suchraum und nicht bloß ein
bereits gelöstes Stecksystem.

## 11. Warum der Moduluswechsel fraktalartig aussieht

Die Quotienten modulo `z^n+1` und `z^(n+1)+1` sind nicht ineinander
verschachtelt. Die beiden Polynome haben keine gemeinsame komplexe Nullstelle:
Aus `z^n=-1` und `z^(n+1)=-1` würde `z=1` folgen, was der ersten Gleichung
widerspricht. Ein Padding-Schritt `n -> n+1` verändert daher das gesamte
Spektralgitter und alle Wrap-around-Beziehungen.

Eine echte rekursive Selbstähnlichkeit existiert dagegen bei multiplikativer
Skalierung. Unter der Turyn-Substitution `z -> z^g` wird

```text
z^n + 1  ->  z^(g*n) + 1.
```

Beim Lift liegt das neue Residuum nur auf den mit `g` teilbaren Lags. Das ist
die mathematisch belastbare Version der Fraktalintuition: nicht
`n -> n+1`, sondern ein skalengetreuer Lift `n -> g*n`.

## 12. TT ist eine exakte Unterparametrisierung von GS4(167)

Der stärkste Befund des TT-Audits ist nicht nur die bekannte Zielkonstruktion.
Die vollständige Kette lässt sich bis in die Koordinaten des vorhandenen
Trackers verfolgen:

```text
TT(56)
  -> BS(111,56)
  -> vier disjunkte T-Sequences der Länge 167
  -> Hadamard-4-Mischung
  -> vier binäre GS4-Folgen der Länge 167
  -> vorhandener Builder erzeugt H668.
```

Für jede TT-Belegung, nicht nur für Lösungen, gilt exakt

```text
u_GS4 = (R_TT / 2, 0, ..., 0).
```

Der variable Teil besitzt 55 Koordinaten; die übrigen 28 der insgesamt 83
GS4-Koordinaten sind konstruktionsbedingt null. Der TT-Raum ist damit eine
223-Bit-Unterfamilie des bestehenden 668-Bit-GS4-Raums und verwendet dieselbe
Zielfunktion bis auf die feste Normierung.

Ein TT-Flip entspricht nach der Einbettung keinem gewöhnlichen GS4-Single:

- ein Bit in `A` oder `B` wird zu einem strukturierten 2-Bit-GS4-Move;
- ein Bit in `C` oder `D` wird zu einem strukturierten 4-Bit-GS4-Move.

Das eröffnet eine Hybridstrategie auch dann, wenn `TT(56)` nicht existiert:
im kompakten TT-Raum einen niedrigen Residualzustand suchen, exakt nach
GS4(167) einbetten und anschließend den freien Solver die TT-Unterfamilie
verlassen lassen. Ob dieser Start besser ist als ein freier Zufallsstart, ist
noch nicht gemessen.

Eine Grenze ist bereits exakt bekannt: Im gewichteten aperiodischen Raum wird
`D^T D` nicht allein vom Residuum bestimmt. Die Tight-Frame-Negativergebnisse
des NAF-Trackers dürfen daher nicht blind auf TT oder Base Sequences übertragen
werden; deren Delta-Wörterbuch kann zusätzliche Basin-Information enthalten.

## 13. Erster gewichteter Greedy-/Tabu-Pilot

Der separate aperiodische Solver verwendet vollständig vektorisierte
Single-Flip-Scores und exakte Tabu-Snapshots. Die TT-Initialisierung erzwingt
den höchsten Lag sowie die notwendigen Summenbedingungen bei `z=1` und
`z=-1`. Die Messzeiten enthalten Initialisierung und Suche.

20 feste Seeds, 100.000 Kandidatenevaluierungen:

| TT-n | gelöst | bestes rohes Q | Zeit/Run |
| ---: | ---: | ---: | ---: |
| 2 | 20/20 | 0 | 0,009 s |
| 4 | 20/20 | 0 | 0,010 s |
| 6 | 14/20 | 0 | 0,071 s |
| 8 | 12/20 | 0 | 0,081 s |
| 10 | 11/20 | 0 | 0,068 s |
| 12 | 2/20 | 0 | 0,088 s |
| 14 | 1/20 | 0 | 0,082 s |
| 16 | 0/20 | 4 | 0,074 s |

Die bekannte Treppe `TT(36)..TT(44)` wurde mit 20 Seeds und 200.000
Evaluierungen getestet:

| TT-n | gelöst | bestes rohes Q | Zeit/Run |
| ---: | ---: | ---: | ---: |
| 36 | 0/20 | 92 | 0,125 s |
| 38 | 0/20 | 152 | 0,124 s |
| 40 | 0/20 | 164 | 0,114 s |
| 42 | 0/20 | 192 | 0,103 s |
| 44 | 0/20 | 176 | 0,097 s |

Ein TT(16)-Parameterpilot mit 100 Seeds und 500.000 Evaluierungen erreichte
je nach Walk-Länge 0 bis 3 Lösungen. Greedy ohne Tabu löste 0/100. Der exakte
`z=+-1`-Seedfilter änderte die Rate bei Defaultparametern nicht (je 2/100).

Das Negativergebnis ist klar: Der Tracker und der Solver funktionieren, aber
der direkte Transfer von GS4-Singles und Tabu reproduziert die bekannte
TT-Treppe noch nicht. Mehr Budget oder einfache Tabu-Parameter reichen nicht.
Der nächste sinnvolle TT-Schritt sind Moves, die TT-Nebenbedingungen erhalten,
beispielsweise gekoppelte Flips für die Endpunkt- und Summenbedingungen. Ein
`TT(46+)`-Sweep wäre vor dieser Änderung nicht aussagekräftig.

## 14. Erster eigenständiger Base-Sequence-Solver

Der gewichtete Tracker und derselbe Greedy-/Tabu-Kern wurden als separater
`BS(m,n)`-Suchpfad verdrahtet. Die Initialisierung erzeugt vier binäre Folgen
der Längen `(m,m,n,n)`, entfernt die vier globalen Vorzeichenfreiheiten und
kann die notwendigen Endpoint- und `z=+-1`-Bedingungen vorfiltern. Jede
gefundene Nullstelle wird vollständig über

```text
BS(m,n) -> GS4(m+n) -> H(4(m+n))
```

mit frischem Aperiodic-Tracker, bestehendem NAF-Tracker und unabhängigem
Hadamard-Audit geprüft.

Der kleine Fixed-Seed-Pilot mit 20 Seeds und 100.000
Kandidatenevaluierungen reproduziert die bekannte leichte Treppe:

| BS-Form | gelöst | bestes rohes Q | Zeit/Run, gefiltert |
| --- | ---: | ---: | ---: |
| `BS(2,1)` bis `BS(9,8)` | überwiegend 20/20 | 0 | 0,006-0,026 s |
| `BS(10,9)` | 13/20 | 0 | 0,067 s |
| `BS(12,11)` | 3/20 | 0 | 0,085 s |
| `BS(16,15)` | 0/20 | 4 | 0,084 s |

Ungefilterte Starts waren in dieser kleinen Stichprobe vergleichbar
(`BS(10,9)` 16/20, `BS(12,11)` 2/20, `BS(16,15)` 0/20). Der exakte
Summenfilter ist damit kein belegter Standardgewinn.

Für die beiden 334-Bit-Zielräume mit `m+n=167` ergaben 20 feste Seeds und
2.000.000 Evaluierungen:

| Raum | Modus | gelöst | bestes BS-Q | bestes eingebettetes GS4-Q |
| --- | --- | ---: | ---: | ---: |
| `BS(84,83)` | Greedy+Tabu, ungefiltert | 0/20 | 484 | 121 |
| `BS(84,83)` | Greedy+Tabu, gefiltert | 0/20 | 492 | 123 |
| `BS(111,56)` | Greedy+Tabu, ungefiltert | 0/20 | 588 | 133 |
| `BS(111,56)` | Greedy+Tabu, gefiltert | 0/20 | 600 | 132 |
| `BS(84,83)` | nur Greedy, ungefiltert | 0/20 | 712 | - |
| `BS(111,56)` | nur Greedy, ungefiltert | 0/20 | 1176 | - |

Tabu verbessert die Restenergie deutlich, löst den großen BS-Raum aber
nicht direkt. Der Summenfilter ist neutral bis teuer. Für `BS(n+1,n)` gilt
bei jeder Belegung exakt `Q_GS4=Q_BS/4`; deshalb ist der beste
`BS(84,83)`-Endzustand ein exakt interpretierbarer GS4(167)-Start mit
`Q=121`. Bei `BS(111,56)` sind BS- und GS4-Ranking wegen des Spiegelterms
nicht identisch; der Sweep speichert daher beide Energien und alle
Endsequenzen für eine gepaarte Hybridablation.

## 15. Fast-BS als GS4(167)-Start

Die ungefilterten Endzustände des 2M-BS-Piloten wurden paarweise gegen
zufällige GS4(167)-Starts getestet. In jedem Paar waren Solverbudget,
Solverkonfiguration und RNG-Stream identisch; nur der Startzustand wechselte.
Targeted Escape blieb aus, damit keine hunderte zusätzlichen Quenches den
reinen Basintransfer überlagern.

| Budget | Quelle | Median Start-Q | Median End-Q | BS besser / gleich / schlechter |
| ---: | --- | ---: | ---: | ---: |
| 2.000 | `BS(84,83)` | 140 | 81 | 20 / 0 / 0 |
| 2.000 | Zufall | 3.204 | 1.711 | - |
| 200.000 | `BS(84,83)` | 140 | 64 | 11 / 3 / 6 |
| 200.000 | Zufall | 3.204 | 66 | - |
| 1.000.000 | `BS(84,83)` | 140 | 62 | 7 / 2 / 11 |
| 1.000.000 | Zufall | 3.204 | 61 | - |
| 1.000.000 | `BS(111,56)` | 152 | 61 | 9 / 1 / 10 |
| 1.000.000 | Zufall | 3.204 | 61 | - |

Das Mapping liefert einen enormen kurzfristigen Warmstart, aber keinen
nachgewiesenen besseren Lösungsbasin. Der freie GS4-Solver holt den
Startunterschied innerhalb von 200k bis 1M Evaluierungen nahezu vollständig
auf. Keine der 80 Läufe bei 1M wurde gelöst. Ein einzelner niedrigster
BS-Q-Kandidat ist damit kein ausreichendes Transferkriterium. Ein späterer
Portfolio-Versuch müsste viele strukturell verschiedene Fast-BS-Zustände
billig downstream screenen, statt nur nach Upstream-Q auszuwählen.
