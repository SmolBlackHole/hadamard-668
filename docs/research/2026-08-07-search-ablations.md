# Konsolidierte Suchablationen

Stand: 2026-08-07

Dieses Dokument enthält die entscheidungsrelevanten Resultate der
abgeschlossenen Pair-, Gray-, Intervall-, Start- und Trace-Experimente. Die
zugehörigen Rohdaten und Einmal-Prototypen wurden nach der Konsolidierung
entfernt.

## 1. Pair-Rescue

Verglichen wurden identische Seeds mit dem Pfad

```text
Singles -> Pair -> Tabu -> Kick
```

gegen

```text
Singles -> Tabu -> Kick.
```

| n | Pair an | Pair aus | McNemar p | CPU/Lösung an | CPU/Lösung aus |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 905/1000 | **938/1000** | 0,010161 | 0,23 s | **0,15 s** |
| 38 | 532/1000 | **619/1000** | 0,000101 | 0,68 s | **0,42 s** |
| 40 | 221/1000 | **254/1000** | 0,093042 | 2,12 s | **1,44 s** |
| 52 | 2/150 | 1/150 | 1,000000 | 1975 s | 3278 s |

Bei `n=36` und `n=38` war Pair aus signifikant besser. `n=52` war mit drei
discordanten Lösungen hinsichtlich Solve-Rate unentscheidbar, aber Pair aus
verkürzte die mittlere Laufzeit von 26,3 auf 21,9 Sekunden. Pair-Rescue wurde
deshalb vollständig aus dem Solver entfernt.

## 2. Statischer Gray-Code bei Low-Q

Auf Pair- und pair-freien `n=52`-Zuständen wurden Kandidatenmengen der Breite
`k=11,12,14,16` untersucht. Die Auswahl erfolgte sowohl über die besten
Single-Scores als auch über gute Pair-Endpunkte.

- 219 gespeicherte Pair-Low-Q-Snapshots: keine Verbesserung;
- 226 pair-freie Low-Q-Snapshots, davon 225 verschieden: keine Verbesserung;
- alle geprüften Zustände hatten `Q<=2`;
- `combo_energy()` wurde gegen direkte NAF-Neuberechnung geprüft.

Der statische Gray-Code findet in diesen isolierten Zuständen keinen direkten
Abstieg.

## 3. Gray-Code über alle lokalen Minima

Auf 11.639 gespeicherten pair-freien Single-Minima bei `n=52` ergab `k=11`:

- 888 Zustände mit irgendeinem Gray-Abstieg;
- 697 beste Treffer waren Pairs;
- 185 beste Treffer waren Triples;
- 6 beste Treffer waren Vierer;
- kein direkter Lösungstreffer;
- alle Verbesserungen lagen bei `Q>=3`.

Gray findet also reale lokale Abstiege, löst aber den eigentlichen
`Q=1/2`-Engpass nicht.

## 4. Online-Gray im Solver

`n=38`, 100 identische Seeds, 200.000 Steps:

| Variante | Gelöst | Ergebnis gegen Baseline |
| --- | ---: | --- |
| Baseline S+TB+K | 74/100 | Referenz |
| Gray vor Tabu | 53/100 | p=0,0060, signifikant schlechter |
| Gray vor Tabu, nur 3+ Bits und Q>=3 | 57/100 | p=0,0147, schlechter |
| Gray nach Tabu | 70/100 | p=0,3865, kein Gewinn und teurer |
| pair-geführter Gray vor Tabu | 46/100 | p=0,0001, deutlich schlechter |

Gray vor Tabu akzeptierte 10.651 exakte Verbesserungen und senkte trotzdem die
Solve-Rate. Bei einem `n=50`-Pilot mit 20 Seeds und 3,2 Millionen Steps löste
keine Variante; häufigeres Erreichen von `Q=1` erzeugte keinen letzten Schritt.

Schlussfolgerung:

> Ein unmittelbar kleineres Q ist keine zuverlässige Aussage darüber, ob der
> neue Zustand in einem besser lösbaren Einzugsgebiet liegt.

Der Online-Gray-Code wurde wieder entfernt.

## 5. Intervallmoves im Kantenraum

Für jede der vier Folgen wurden alle zusammenhängenden Intervalle geprüft,
insgesamt 5.304 Moves pro `n=52`-Zustand.

- 150 Pair-Low-Q-Zustände: keine Verbesserung;
- 150 pair-freie Low-Q-Zustände: keine Verbesserung;
- 795.600 exakte Auswertungen pro Population.

Ein einzelner Intervallflip reicht damit nicht als Low-Q-Rescue. Mehrere
Intervalle oder zustandsabhängige Randwahl bleiben offen.

## 6. Kantenbalancierter Start

Starts mit exakt erfülltem `r_1=0` wurden isoliert und gemeinsam mit Gray
getestet. Die Solve-Rate unterschied sich nicht robust von zufälligen Starts.
Die Nebenbedingung wird durch die späteren Moves nicht erhalten und genügt
allein nicht als Initialisierungsstrategie.

## 7. Scoring-Audit

Die negativen Resultate sind kein bekannter Fehler der Energieberechnung:

- `combo_energy()` wurde exhaustiv bei `n=3` geprüft;
- zufällige Fälle bei `n=38,49,50,51,52` wurden direkt neu aufgebaut;
- Flipbreiten bis 20 wurden geprüft;
- Same-Sequence-Korrekturen stimmen auch bei ungeradem `n`;
- NAF ist quadratisch, daher existieren keine zusätzlichen Triple- oder
  Quadruple-Energieterme.

Höherordentliches Verhalten entsteht dynamisch, weil sich das Wörterbuch `D`
nach jedem Flip ändert, nicht durch fehlende statische Terme in `Q`.

## 8. Verbleibende Suchhypothese

Der aktuelle Tabu-Walk darf intern schlechter werden, übernimmt aber nur einen
Zustand, der bereits besser als sein Start ist. Der zufällige Kick übernimmt
einen schlechteren Zustand und lässt Singles anschließend erneut absteigen,
ist aber ungerichtet.

Noch nicht geprüft wurde der gezielte Ablauf:

```text
lokales Minimum x
-> strukturierter schlechterer Zustand y
-> greedy Singles bis zum Minimum L(y)
-> erst L(y) bewerten.
```

Gray-Subsets oder der finale statt beste Tabu-Zustand könnten dabei als
Proposal dienen. Dies ist eine neue Basin-Hopping-Hypothese und wird durch die
negativen monotonen Gray-Tests nicht widerlegt.
