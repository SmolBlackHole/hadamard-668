# Audit: NAF-Kanäle, `u(Fx)` und Attraktorqualität

## Ergebnis in einem Satz

Die vier signierten Einzel-NAF-Kanäle enthalten nachweislich Information, die
im Summenresidual `u` verloren geht. Sie verbessern aber weder Basinwechsel,
Abstieg in ein niedrigeres Minimum noch finales `Q` in den vorhandenen
Shell-Daten. Für den späteren 50k-Solve bei `n=40` erscheint erstmals ein
schwaches Ranking-Signal; `u(Fx)` bleibt auch dort negativ.

Das ist kein Widerspruch. „Zusätzliche Zustandsinformation“ und „nützlicher
Navigationssensor“ sind zwei verschiedene Aussagen.

## 1. Geprüfte Größen

Für die vier Folgen wurden die signierten Einzelresiduen

\[
\rho_0,\rho_1,\rho_2,\rho_3
\]

frisch aus den Sequenzen berechnet. Für alle `896` Roots wurde exakt geprüft:

\[
\rho_0+\rho_1+\rho_2+\rho_3=4u.
\]

Zusätzlich wurde die bereits untersuchte Transformation `F` angewandt und der
vollständige Residualvektor `u(Fx)` neu aufgebaut. Seine Energie stimmte bei
allen Roots mit dem gespeicherten `dual_q` überein.

Die getesteten Zusatzmerkmale waren:

- alle vier `rho_s`, kanonisch nach ihren Vektoren sortiert;
- permutationsinvariante Kanalmerkmale: zentrierte Kanalnormen,
  Gram-Momente, alle sechs Paar-Summennormen und niedrige Momente;
- vollständiges `u(Fx)`;
- eine kleine invariante Zusammenfassung von `u(Fx)` und seinem Verhältnis zu
  `u`.

Die Baseline enthielt bereits `n`, `Q`, den vollständigen signierten
Residualvektor `u`, Entry-Count sowie mittleren und maximalen prospective
degree. Damit wurde nicht lediglich erneut getestet, ob die Kanäle `Q` oder
`u` indirekt rekonstruieren können.

## 2. Invarianz und Redundanz

Die verwendete volle Kanaldarstellung und ihre Zusammenfassung sind invariant
unter:

- globalem Vorzeichenwechsel einer einzelnen Folge;
- Permutation der vier Folgen.

`u(Fx)` und seine Zusammenfassung besitzen dieselben beiden Invarianzen. Die
Kanal-Gram-Spektren wurden nicht direkt als Float-Eigenwerte verwendet, sondern
durch die exakten polynomialen Momente `tr(G)`, `tr(G²)` und `tr(G³)` ersetzt.

### Die Zusatzinformation ist algebraisch echt

Eine vollständige Enumeration der `65.536` normalisierten Zustände bei `n=5`
zeigt:

- `31` verschiedene `u`-Fasern;
- `4/31` dieser Fasern enthalten mehr als eine kanonische
  Einzelkanal-Konfiguration;
- `256` vollständige signierte Kanalfasern;
- `243/256` dieser Fasern enthalten mehr als ein `u(Fx)`.

Also gilt in beide Richtungen:

\[
u\not\Rightarrow (\rho_0,\ldots,\rho_3),
\qquad
(\rho_0,\ldots,\rho_3)\not\Rightarrow u(Fx).
\]

Die Größen sind damit nicht bloß umbenannte Kopien voneinander.

### Exaktes Matching im Production-Datensatz ist nicht möglich

Die `896` Roots besitzen fast alle ein einzigartiges `u`:

- bei `n=28`: `127/128` verschiedene `u`;
- bei jedem `n=32,36,40,44,48,52`: `128/128` verschiedene `u`.

Es gibt daher nur eine einzige `(n,Q,u)`-Kollision. Die beiden `n=28`-Roots
unterscheiden sich in den Kanälen und in `dual_q`, aber nicht belastbar in ihren
acht Shell-Replikaten. Eine echte within-`(n,Q,u)`-Matched-Analyse braucht
einen absichtlich so erzeugten Datensatz.

## 3. Shell-Ziele: klares Negativergebnis

Verwendet wurden alle `896` exakten und klassen-eindeutigen Greedy-Roots aus
`data/basin-return-map-multin-sensors.json`, jeweils mit acht Replikaten pro
Passhöhe. Die Modelle wurden innerhalb jedes `n` vierfach kreuzvalidiert; eine
Root-Einheit erscheint nie gleichzeitig in Train und Test. Unsicherheit wurde
durch einen nach `n` geschichteten Root-Bootstrap geschätzt.

Die folgende Tabelle verwendet jeweils die Ridge-Stärke, bei der die Baseline
ihren besten Brier-Score beziehungsweise RMSE erreicht:

| Ziel | Baseline | + Kanal-Summary | + `u(Fx)`-Summary | Ergebnis |
| --- | ---: | ---: | ---: | --- |
| neuer Basin, H2 (Brier) | 0,02682 | 0,02980 | 0,02889 | beide schlechter |
| niedriger Basin, H2 (Brier) | 0,03197 | 0,04049 | 0,03514 | beide schlechter |
| niedriger Basin, H4 (Brier) | 0,03095 | 0,03713 | 0,03431 | beide schlechter |
| mittleres finales Q, H2 (RMSE) | 0,78902 | 0,80436 | 0,79628 | beide schlechter |
| mittleres finales Q, H4 (RMSE) | 0,67751 | 0,67924 | 0,67844 | kein Gewinn |

Für die binären Ziele liegen die Root-Bootstrap-Intervalle der Verschlechterung
vollständig auf der negativen Seite. Die vollen hochdimensionalen Vektoren
überfitten noch stärker. Keine der vier vorab festgelegten Ridge-Stärken macht
aus den Kanal- oder Dualmerkmalen einen robusten Shell-Sensor.

Das Negativergebnis ist inhaltlich nützlich: `lower_basin` und `final_q` messen
offenbar vor allem Tiefe, nicht die spätere Erreichbarkeit einer Lösung.

## 4. Solve-Hazard: ein schwacher Kanal-Lead

Der spätere Datensatz `data/directed-probe-downstream-50k.json` enthält bei
`n=40` genug positive 50k-Solves:

| Policy | Solves |
| --- | ---: |
| random | 48/128 |
| prospective-degree | 36/128 |
| lookahead-Q | 30/128 |

Die `384` gequenchten Endpunkte wurden aus Seeds und Policy exakt regeneriert;
alle `384` Endpunkt-Hashes stimmen. Für jeden Endpunkt wurden `Q`, volles `u`
und die lokale H4-Topologie neu berechnet. Alle drei Endpunkte desselben Roots
bleiben im selben CV-Fold.

Bei Ridge `1000` ergibt sich:

| Modell | Brier | AUC |
| --- | ---: | ---: |
| Baseline | 0,21281 | 0,4421 |
| + Kanal-Summary | 0,21002 | 0,5198 |
| + `u(Fx)`-Summary | 0,21300 | 0,4452 |
| + beide | 0,21010 | 0,5144 |

Für die Kanal-Summary beträgt die Root-Bootstrap-Differenz:

- AUC: `+0,077`, 95%-Intervall `[+0,027; +0,129]`;
- Brier: `+0,00279`, 95%-Intervall `[-0,00053; +0,00629]`.

Die AUC-Verbesserung gegenüber derselben Baseline erscheint bei allen vier
festen Ridge-Stärken und ist im Root-Bootstrap positiv. Trotzdem ist das
Resultat nur ein Lead:

1. Die absolute AUC `0,520` liegt nur knapp über Zufall.
2. Der kalibrierte Brier-Gewinn schließt null ein.
3. Es handelt sich um genau einen `n=40`-Datensatz und einen 50k-Solver ohne
   Targeted Escape.
4. Bei `n=44` existieren nur zehn positive Policy-Endpunkte; bei `n=48,52`
   gar keine. Eine held-out-`n`-Bestätigung ist daher noch nicht möglich.

`u(Fx)` zeigt kein inkrementelles Solve-Signal. Die volle Kanalrepräsentation
überfitten die `384` Beispiele; nur ihre kleine invariante Zusammenfassung ist
weiter prüfenswert.

## 5. Reverse-Path-Hazard

Der unabhängige Reverse-Curriculum-Audit enthält `408` Zustände mit exakt
bestimmter Lösungsdistanz `1/2/3`, gruppiert nach Ursprungslösung. Dort waren
statische Kanal- und Dualmerkmale bereits negativ:

- volle vier `rho_s`: cross-source Nearest-Neighbor-Accuracy `32,6 %`;
- `u(Fx)`: `29,7 %`;
- Zufallsbasis für die drei Klassen: `33,3 %`.

Der aktionsbezogene Two-ply-Sensor dagegen erkennt bei Tiefe drei einen
gepflanzten Rückweg:

- in `126/136` Zuständen liegt mindestens ein bekannter Reverse-First-Move in
  der global besten Lookahead-Menge;
- ein K8-Tournament wählt ihn in `12,82 %` der Entscheidungen;
- ein zufälliger First-Move nur in `1,89 %`.

Das trennt zwei Arten von Sensoren:

\[
\underbrace{f(x)}_{\text{statischer Zustandssensor}}
\qquad\text{gegen}\qquad
\underbrace{f(x,a)}_{\text{aktionsbezogener Hazard-Sensor}}.
\]

Die Root-Kanäle sind für alle Aktionen desselben Zustands konstant und können
deshalb allein keinen Reverse-Move auswählen. Der nächste faire Kanaltest muss
die **Änderung nach einer Probeaktion** verwenden, beispielsweise

\[
\Delta\Phi_a=Phi(x\oplus a)-\Phi(x),
\]

wobei `Phi` die kleine Kanal-Summary bezeichnet. Trainiert wird auf bekannten
Reverse-Pfaden, gruppiert nach Ursprungslösung; getestet wird anschließend auf
komplett held-out Lösungen. Als Baseline dienen Candidate-Q, prospective degree
und Two-ply-Q derselben Aktion. Das vorhandene JSON speichert die dafür nötigen
Aktionszustände und ursprünglichen Damage-Flips nicht, daher wäre eine
Regeneration beziehungsweise Erweiterung des Reverse-Datensatzes nötig.

Dieser Test ist wichtiger als ein weiteres Modell für `lower_basin`: Die neue
Downstream-Evidenz zeigt bereits, dass tiefere Landungen nicht automatisch
höhere Solve-Hazard bedeuten.

## 6. Kleinster fairer Aktor-Test

Für die unmittelbare Navigation ergibt sich weiterhin ein kleiner, sauber
budgetierter Test:

1. `fixed-H2`: immer bis Passhöhe `Q_root+2`;
2. `fixed-H4`: immer bis `Q_root+4`;
3. `adaptive`: zuerst H2-Sensor messen; nur bei niedrigem geschätztem
   Escape-Hazard zu H4 erweitern.

Alle Sensorbewertungen zählen zum Candidate-Budget. Für den adaptiven Arm darf
nicht der vollständige teure Sensor an allen Entry-Moves verwendet werden;
fair ist ein vorab fixierter K4- oder K8-Sample. Primärziele sind downstream
Solve-Hazard und Reverse-Path-Hazard, nicht `lower_basin`.

Die Kanal-Summary sollte in diesem Aktor-Test zunächst nur als sekundärer
Tie-Breaker mitgeführt werden. Der aktuelle Solve-Lead reicht nicht aus, sie als
primären Controller einzubauen.

## Reproduktion

```text
python -m experiments.learning.basin_channel_analysis \
  --output data/basin-channel-analysis.json

python -m experiments.learning.downstream_channel_analysis \
  --output data/downstream-channel-analysis.json
```

Zugehörige Dateien:

- `experiments/learning/basin_channel_analysis.py`
- `data/basin-channel-analysis.json`
- `experiments/learning/downstream_channel_analysis.py`
- `data/downstream-channel-analysis.json`
