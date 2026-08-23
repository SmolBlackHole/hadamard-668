# Kanonische Suchbefunde

Stand: 2026-08-09

Dieses Dokument enthält nur reproduzierte Aussagen über die Suchdynamik. Die
exakte Mathematik steht in `TIGHT_FRAME_CHARACTERIZATION.md`, Konstruktionen
in `CONSTRUCTION_SPACE.md`, offene Vermutungen in `HYPOTHESES.md`.

## Aktueller Suchpfad

```text
greedy Singles -> Q-Window -> Tabu
               -> einmaliger Targeted Escape oder zufälliger Zwei-Bit-Kick
```

`Q=||u||²` ist der exakte aktuelle GS4-Fehler. Es ist aber kein zuverlässiges
Maß dafür, wie gut der Zustand später eine Lösung erreicht. Tabu ist der
stärkste gemessene Einzelhebel: Er erlaubt vorübergehende Verschlechterungen
und überwindet Plateaus, an denen Greedy und zufällige Kicks hängen bleiben.

## Targeted Escape

Für einen verletzten Lag `k` mit `u_k=sigma` wählt der historische Kick je
einen Flip aus allen vier Folgen mit `d_{s,c,k}=-sigma`. Da die Flips in
verschiedenen Folgen liegen, addieren sich ihre Deltas exakt:

```text
u'   = u + d_1 + d_2 + d_3 + d_4
u'_k = sigma - 4 sigma = -3 sigma.
```

Der Kick repariert den Ziellag nicht. Sein Beitrag zu `Q` steigt dort von 1
auf 9. Der Nutzen besteht darin, einen anderen Basin zu betreten; erst der
anschließende vollständige ILS-Quench findet gelegentlich eine Lösung.

Reproduzierter n=52-Vergleich, 120 gepaarte Seeds und 100k Hauptschritte:

| Policy | gelöst | CPU/Run |
| --- | ---: | ---: |
| entferntes Dominant-Lag-Portfolio, Quench 10k | 3/120 | 22,1 s |
| Support-Lag-Portfolio, Quench 10k | 6/120 | 21,9 s |
| Support-Lag-Portfolio, Quench 5k | 3/120 | 10,4 s |
| Support-Lag-Portfolio, Quench 2k | 1/120 | 4,3 s |

Der Unterschied 3/120 gegen 6/120 ist nicht signifikant. Die beiden Policys
öffnen aber verschiedene Proposalräume. Der GPU-Top-Q-Selector ist verworfen:
historische Gewinner lagen auf Post-Kick-Q-Rängen 103 bis 606 von 625 und
wären vollständig aussortiert worden.

Weitere exakte Grenzen:

- Ein Target-Kick erzwingt für jeden direkten Single-Nachbarn `Q>=4`.
- Nicht jeder Q=1-Zustand besitzt in jeder Folge einen passenden Target-Flip;
  kleine Gegenbeispiele existieren.
- Auf 494 ungelösten n=52-Endzuständen existierte keine direkte exakte
  Cross-Sequence-Annullierung mit ein bis vier Flips.
- Gewinner-Quenches starteten oft bei deutlich höherem Q und erreichten die
  Lösung erst über Greedy, Random-Kick und einen finalen Tabu-Walk.

## Cancellation-Manifold

Ein Single-Flip löst genau dann unmittelbar, wenn `d_i=-u`. Die erfolgreichen
finalen Moves wurden in allen instrumentierten Läufen innerhalb eines
Tabu-Walks beobachtet. Vor dem lösenden Flip lag Q typischerweise zwischen 2
und 11; eine vorherige Aufwärtsexkursion war meistens notwendig. Gesammelte
n=44- und n=52-Escapezustände lagen statisch mindestens zwei Flips von dieser
Manifold entfernt. Lokale C-Nähe war kein brauchbarer Erfolgsprädiktor.

## Verworfene oder negative Suchideen

- **Pair-Rescue:** langsamer und signifikant schlechter bei n=36/38; entfernt.
- **Greedy `10n`-Symbolnachbarschaft:** exakt vektorisierte 4n Singles plus 6n
  basis-erhaltende Spaltenpaare. Bei 200k Schritten ergaben 50 gepaarte Seeds
  n=44: 50/50 gegen 47/50 und n=52: 1/50 gegen 0/50. Die Paare wurden nur 38
  beziehungsweise 31 Mal akzeptiert und liefern als monotone Phase keinen
  Solve-Gewinn. Der optionale Forschungs-Schalter bleibt standardmäßig aus.
- **Monotone Gray-Suche:** viele echte Q-Verbesserungen, aber weniger Lösungen.
- **Triples:** nach korrekter Vektorisierung kein signifikanter Solve-Gewinn.
- **Einzelne Intervallflips:** auf Low-Q-Traces ohne Durchbruch.
- **Cyclotomischer Startfilter:** kein signifikanter additiver Effekt zu Tabu.
- **Post-Kick-Q, frühe Greedy-Drops und Delta-Normen:** kein verlässliches
  Ranking erfolgreicher Escape-Kandidaten.
- **Quadratische Frame-/Spektralgrößen:** Funktionen des bekannten Residuums,
  daher keine unabhängige Basin-Information.
- **Vollständige Delta-Grammatrix:** selbst gleiches `u` und gleiches `D^T D`
  können null gegen sechs unmittelbar lösende Single-Zeilen besitzen. Die
  konkrete Wörterbuchfaktorisierung darf nicht wegaggregiert werden.
- **ML auf Zustandsfeatures:** nur schwaches AUC-Signal um 0,6.
- **Basislabel allein:** vollständig bis `n=6` ohne nichttriviale notwendige
  Bedingung; simples Paley-Spaltenswitching bei `n=10,12,16` liefert keine
  gemischte Konstruktion.
- **Festes Paar-Residualtemplate:** 1.053 Paarvektoren aus allen drei Paarungen
  der freien Archivlösungen waren bis auf Vorzeichen sämtlich verschieden.

## Strukturierte Starts

Der BS->GS4-Warmstart wurde bei n=167 gepaart getestet:

| Budget | Fast-BS Start-Q | Zufalls-Start-Q | Fast-BS End-Q | Zufalls-End-Q |
| ---: | ---: | ---: | ---: | ---: |
| 2k | 140/152 | 3204 | 81/107 | 1711 |
| 200k | 140/152 | 3204 | 64/65 | 66 |
| 1M | 140/152 | 3204 | 61/62 | 61 |

Fast-BS ist ein starker kurzfristiger Warmstart, aber kein nachgewiesen
besserer Basin. Der aktuelle Targeted Escape war dagegen in dünnen ternären
Residualzonen ungefähr bei Q=3 bis 6 nützlich. Ein Start bei Q≈120 ist noch
zu weit von seiner bekannten Einsatzregion entfernt; beim Abstieg geht die
ursprüngliche BS-Struktur offenbar verloren.

## Q1-Wand und kontrollierte Shell-Navigation

Für `Q=1` gilt `u=sigma e_k`. Ein Single-Flip kann dann nur verbessern, wenn
sein exaktes Delta `d=-u` ist und damit direkt löst. Ein neutraler `Q1 -> Q1`-
Move ist entweder ein stiller Flip `d=0` oder transportiert den einzelnen
Defekt mit `||d||²=2` auf einen anderen Lag. Diese Aussage ist exakt.

Auf 109 realen n=52-Q1-Endzuständen wurde der lokale Graph vollständig
ausgewertet:

- alle 109 Q1-Komponenten unter Singles sind Singletons;
- die vollständigen Sublevel-Komponenten bis einschließlich `Q=8` und `Q=9`
  sind ausgeschöpft und enthalten keine Lösung;
- kein Endpunkt aus beliebigen ein, zwei oder drei Bit-Flips besitzt `Q=0`
  oder `Q=1`;
- für diese Daten gilt daher Hamming-Abstand mindestens vier und eine
  Single-Flip-Passhöhe von mindestens `Q=10`.

Das ist keine universelle Kongruenzbarriere. Vollständige kleine Beispiele
besitzen neutrale Q1-Transportpfade; ein isolierter n=12-Q1-Zustand löst schon
über Passhöhe zwei. Die n=52-Wand ist eine dünne, zustandsabhängige
Erreichbarkeitsstruktur des konkreten Delta-Wörterbuchs.

Ein bandbegrenzter Shell-Walk kann gezielt weit entfernte Zustände bei
kontrolliertem Q erreichen. Ein anschließender identischer Greedy-Quench löste
in den bisherigen n=52-Tests jedoch keinen Zustand. Der exakte prospektive
Zwei-Schritt-Grad

```text
b_H(i) = Anzahl zweiter Flips j mit Q(x flip i,j) <= H
```

korreliert stark mit der lokalen Sublevel-Komponentengröße und ist damit ein
echter Topologiesensor. Bei gleichem Evaluationsbudget war permanentes Scoring
aber zu teuer. Ein adaptiver K4/K8-Scan nur bei kleinen lokalen Graden öffnete
bei `H=12` mehr Wege als Zufall, brachte über 3.488 gepaarte Läufe bei
`H=12/16` aber ebenfalls keine Lösung. Der Sensor kartiert lokale
Verzweigung; er misst noch keine Lösungsrichtung.

## Paarresiduum als Separator

Für eine Paarung gilt exakt `w=rho_a+rho_b`; das zweite Folgenpaar muss `-w`
erzeugen. Das liefert die Meet-in-the-Middle-Zerlegung

```text
Anzahl GS4 = Summe_w p_n(w) p_n(-w),
```

wobei `p_n(w)` die Anzahl erzeugender Folgenpaare ist. Die Parität erzwingt
für ungerades n `w_t = 2 mod 4`; `w=0` ist dort unmöglich. Für gerades n sind
alle Koordinaten durch vier teilbar.

Vollständige Enumeration bis n=16 und 19 freie n=52-Lösungen widerlegen die
naheliegende Minimalnorm-Heuristik: Die größte Kollisionsmasse liegt schon bei
n=16 nicht bei minimalem `||w||`, und keine archivierte freie n=52-Lösung
liegt in der Minimaldefektschale. Der relevante unbekannte Wert ist die
Komplementdichte `p_n(-w)`, nicht die Länge von `w`.

## Separater 10n-Symbolsolver

`gs4-symbol` ist ein kleiner Best-Move-Tabu-Walk ohne Sonderlogik. Anders als
die negative monotone 10n-Ablation verwendet er Singles und basis-erhaltende
Spaltenpaare in derselben nichtmonotonen Nachbarschaft. Ein reproduzierter
Klein-n-Sweep ergab:

| n | Seeds | Budget | gelöst |
| ---: | ---: | ---: | ---: |
| 5 bis 22 | je 20 | 50k beziehungsweise 100k | jeweils 20/20 |
| 24 | 50 | 200k | 50/50 |
| 26 | 50 | 200k | 45/50 |
| 28 | 50 | 200k | 24/50 |

Aufwärts- und Plateaumoves traten ab n=16 tatsächlich auf und waren in
einzelnen reproduzierbaren Lösungswegen notwendig. Alle ungelösten n=26/28-
Läufe endeten bei Q=1. Der Produktionssolver bleibt bei gleichem Budget ab
n=24 stärker; der Symbolsolver ist daher ein isoliertes Forschungsinstrument,
kein neuer Default.

Ein fairer interner Ablationswalk mit exakt 200k bewerteten Kandidaten zeigte,
dass die zusätzlichen Moves allein nicht den Gewinn tragen:

| n | 4n Singles | 6n Spaltenpaare | 10n Hybrid |
| ---: | ---: | ---: | ---: |
| 26 | 50/50 | 9/50 | 45/50 |
| 28 | 42/50 | 3/50 | 24/50 |
| 30 | 25/50 | 2/50 | 11/50 |
| 32 | 10/50 | 2/50 | 4/50 |

Bei gleichem Candidate-Budget war 4n auch schneller. Die H4-Moves sind exakt,
aber ihre ungefilterte Aufnahme verschlechtert diesen einfachen Walk.

Der Zwei-Kanal-Scorer verwendet die drei Paarungen `01|23`, `02|13` und
`03|12`. Bei n=28 verbesserte er den tatsächlichen 10n-Solver auf denselben
200 Seeds von 103 auf 125 Lösungen (33 verlorene, 55 gewonnene Seeds,
McNemar-exakt p=0,0246). Der unabhängige Trend war bei n=30 positiv
(21/100 auf 33/100, p=0,081), bei n=32 aber neutral (12/100 auf 11/100).
Darum bleibt `gs4-symbol-channel` eine explizite Forschungsstrategie und wird
nicht zum Default erklärt.

## Repräsentation und Solversteuerung

Die exakte H4-Spaltenkoordinate aus `CONSTRUCTION_SPACE.md` zeigt, dass der
freie Solver eine Folge über zwei orthonormalen Basen durchsucht. Direkte
Paley-/Golay- und BS-/TT-Einbettungen verwenden besondere achsenreine
Unteralphabete; die gespeicherten freien Lösungen sind in ihrer aktuellen
Repräsentation gemischt.

Zwei vorhandene Steuerungsentscheidungen sind deshalb neue Ablationsziele:

- Tabu gibt nur einen strikt niedrigeren Bestzustand zurück. Gleiche-Q-Zustände
  mit einer anderen konkreten Delta-Faktorisierung werden nach dem Walk
  verworfen.
- Greedy nimmt den ersten verbesserten Flip in fester Reihenfolge und beginnt
  danach wieder bei Index null. Damit wird die mathematische
  Verschiebungs-/Permutationssymmetrie algorithmisch nicht symmetrisch
  behandelt.

Das sind plausible Ursachen für schlechte Navigation, noch keine gemessenen
Fehler. Beide müssen mit gleichem Kandidatbudget gepaart getestet werden.

## Basin-Navigation

Ein deterministischer Quench definiert operative Basins exakt. Auf 896
unabhängigen Greedy-Minima über `n=28..52` fiel die Rückkehrquote eines
64-Schritt-Shell-Walks von rund `8,0 %` bei `Q_root+2` auf `2,0 %` bei `+4`
und praktisch null bei `+8/+16`. Ein separater Höhen-/Längentest zeigte, dass
nicht Höhe allein, sondern Höhe plus mindestens wenige laterale Schritte den
Basinwechsel auslöst.

Der exakte prospektive Zwei-Schritt-Grad enthält bei `+2` robuste
Return-Information über `n`, Root-`Q` und Entry-Count hinaus. Ein
kostenbelasteter `+2/+4`-Regler halbierte damit die Return-Rate, erhöhte aber
nicht die Zahl tieferer Basins pro Candidate-Arbeit. Der Sensor misst damit
nachweislich **Basin-Stickiness**, noch nicht Ziel- oder Solve-Nähe.

Eine direkte topologiegeführte `+4`-Shell wählte pro Schritt den besten aus
acht gesampelten Moves nach prospektivem Zwei-Schritt-Grad. Auf je 256 neuen
Roots erhöhte sie bei `n=40,44,48,52` die Rate tieferer Zielbasins von
`38-39 %` auf `52-66 %`; alle vier gepaarten Tests waren signifikant. Das ist
der erste belegte Sensor-Aktor-Effekt des Projekts. Die aktuelle Umsetzung ist
10- bis 12-mal teurer und erzeugte noch keine Lösung, daher bleibt die Aussage
auf gezielte Attraktornavigation beschränkt.

Die Basinpartition ist quenchabhängig: Best- und First-Improvement ergaben bei
`+2` etwa `10,0 %` beziehungsweise `6,0 %` Return. Bei `+4` waren beide mit
etwa `1,3 %` nahezu gleich. Der grobe Escape ist robuster als die enge
Zuordnung eines Zustands zu einem Attraktor.

Die kanonischen Definitionen, Daten und Grenzen stehen in
[`BASIN_NAVIGATION.md`](BASIN_NAVIGATION.md).

### Gerichteter Zwei-Schritt-Lookahead und Reverse-Curriculum

Der Score `min_j Q(x flip i flip j)` ist ein Horizon-2-Rollout: Er bewertet
einen hypothetischen zweiten Flip, führt aber nur den ersten aus und plant im
neuen Zustand erneut. Auf absichtlich zwei bis fünf Schritte von archivierten
Lösungen entfernten Zuständen enthielt die global beste Menge bei Distanz drei
in `126/136` Fällen einen bekannten korrekten ersten Rückweg. Ein gesampeltes
K8-Turnier traf ihn `12,82 %` statt `1,89 %` bei zufälliger Wahl.

Ein linearer Ranker aus Q-, Kanal-, Delta-, Kanten- und Symbolmerkmalen brachte
keinen Zusatznutzen und war downstream signifikant schlechter als der reine
Lookahead (`22/96` statt `37/96`, gepaart `p=0,0081`). Statische
Einzel-NAF-Kanäle und `u(Fx)` enthalten zwar Information jenseits von `u`,
verbesserten Return, tieferes Basin und finales Q aber nicht robust. Das
brauchbare Signal ist damit aktionsbezogen, nicht bloß ein Zustandsvektor.

### Integrabilität und Symbol-Makromoves

Die GF(2)-Kantenrekonstruktion kann für eingefrorenes `u` eine gewünschte
Delta-Zeile `d=-u` exakt erzeugen. Nach Einsetzen der rekonstruierten Folge
ändert sich jedoch das Residuum selbst; exakt bleibt

```text
Q_neu = ||rho_neu - rho_alt||^2 / 16.
```

Ohne zusätzliche Erhaltung des Einzelkanals ist das Reverse-Wörterbuch daher
nicht selbstkonsistent. Die getesteten affinen Kandidaten waren ab `n=24`
nicht besser als zufälliger Folgeneratz.

`10n`/`15n`-Symbolmoves waren bei gleichem Candidate-Budget insgesamt neutral.
Bei spaltengeclusterten Defekten sehen sie jedoch exakte kurze Pfade, die `4n`
nicht sieht: `10n` erkennt Pair-plus-Single, `15n` zusätzlich
Triple-plus-Single. Sie bleiben deshalb als bedingt aktivierter Makro-Aktor
interessant, nicht als dauerhafte vollständige Nachbarschaft.

### QWindow-Skalierung

Die mittlere Energie der rückwärts von freien Lösungen erzeugten
Single-Flip-Grenze ist `(n-2)/4`. Daraus folgt aber kein guter globaler
Schaltpunkt für den Solver. Eine gepaarte Ablation mit 200k Steps ergab:

| n | fest 9 | `ceil((n-2)/4)` | natürlich +2 |
| ---: | ---: | ---: | ---: |
| 40 | 50/50 | 50/50 | 50/50 |
| 44 | 99/100 | 96/100 | 99/100 |
| 48 | 19/50 | 13/50 | 9/50 |

Bei `n=48` war `+2` gepaart schlechter als fest 9 (`p=0,044`). Die
Lösungsrand-Schale beschreibt mögliche Killer-Zustände, aber ein höheres
Fenster schwächt offenbar den Selektionsdruck auf die seltenen günstigen Teile
dieser Schale. Der Produktionsdefault bleibt unverändert.

## Nächste belastbare Experimente

1. Support-Lag-Targeting gegen deaktivierten Target-Escape unter gleichem
   Candidate-Evaluationsbudget vergleichen.
2. Viele strukturell verschiedene Fast-BS-Zustände nur kurz downstream
   quenchen; nach GS4-Reaktion statt Upstream-Q auswählen.
3. Messen, ob strukturierte Starts die ternäre Q<=8-Zone häufiger oder früher
   erreichen. Erst dort Targeted Escape vergleichen.
4. Den exakt gefalteten QPSK-Raum für neue strukturtreue Moves verwenden.
5. Die vorhandenen 6n basis-erhaltenden Spaltenpaare nichtmonoton in Tabu oder
   auf neutralen Q-Fasern testen; erst danach die volle `15n`-Nachbarschaft.
6. Einen Zielqualitätssensor entwickeln: Return/Verzweigung kartiert die Wand,
   sagt aber noch nicht, welches Zielbasin mehr Cancellation-/Solve-Hazard hat.

## Evidenzbasis

Die dauerhaft relevanten Rohdaten bleiben `data/benchmark-n52.json`,
`data/benchmark.json` und `data/solutions.json`. Sessionbezogene
`temporary-*`-Ausgaben und Wegwerfskripte wurden nach Übernahme ihrer Resultate
entfernt.
