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
| `legacy625`, Quench 10k | 3/120 | 22,1 s |
| `support_lag625`, Quench 10k | 6/120 | 21,9 s |
| `support_lag625`, Quench 5k | 3/120 | 10,4 s |
| `support_lag625`, Quench 2k | 1/120 | 4,3 s |

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

## Nächste belastbare Experimente

1. `legacy625` und `support_lag625` mit kandidatenstabilen RNG-Streams als
   echtes Portfolio vergleichen.
2. Viele strukturell verschiedene Fast-BS-Zustände nur kurz downstream
   quenchen; nach GS4-Reaktion statt Upstream-Q auswählen.
3. Messen, ob strukturierte Starts die ternäre Q<=8-Zone häufiger oder früher
   erreichen. Erst dort Targeted Escape vergleichen.
4. Den exakt gefalteten QPSK-Raum für neue strukturtreue Moves verwenden.
5. Die vollständige `15n`-Symbolnachbarschaft und danach Tabu-Fortsetzung auf
   neutralen Q-Fasern testen.

## Evidenzbasis

Die dauerhaft relevanten Rohdaten bleiben `data/benchmark-n52.json`,
`data/benchmark.json` und `data/solutions.json`. Sessionbezogene
`temporary-*`-Ausgaben und Wegwerfskripte wurden nach Übernahme ihrer Resultate
entfernt.
