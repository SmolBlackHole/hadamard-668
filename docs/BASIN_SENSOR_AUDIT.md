# Unabhängiger Audit des Basin-Sensors

Stand: 2026-08-09

## Ergebnis in einem Satz

Der prospektive Zwei-Schritt-Grad ist ein echter, vor der Exkursion berechneter
Sensor für die **Rückkehrwahrscheinlichkeit der aktuellen Shell-Policy**. Bei
relativer Höhe `+2` besitzt er robusten inkrementellen Vorhersagewert. Das ist
weder Outcome-Leakage noch bislang ein Signal für Solve-Nähe; außerdem wurde
seine nicht unerhebliche Berechnungsarbeit im Aktorvergleich noch nicht
berechnet.

## 1. Geprüfter Datensatz

`data/basin-return-map-multin-sensors.json` enthält:

- `n = 28, 32, 36, 40, 44, 48, 52`;
- je 128 unabhängige Zufallsstarts, deterministisch zu lokalen Minima
  gequencht;
- 896 verschiedene State-Hashes und auch 896 verschiedene Class-Hashes;
- je Root acht Exkursionsreplikate bei relativer Höhe `+2,+4,+8,+16`;
- insgesamt 28.672 Exkursionsläufe.

Damit behebt dieser Datensatz den wesentlichen Auswahlfehler des alten Piloten
mit 109 selektierten `n=52,Q=1`-Endzuständen. Er repräsentiert jetzt allerdings
die Verteilung von **greedy-gequenchten Zufallsstarts**, nicht die Verteilung
der terminalen Zustände des Production-Solvers.

Die exakten Rückkehrzahlen, konditioniert auf mindestens einen akzeptierten
Exkursionsmove, sind:

| relative Höhe | bewegte Läufe | exakte Rückkehr | Rate |
| ---: | ---: | ---: | ---: |
| `+2` | 7.096 | 567 | 7,99 % |
| `+4` | 7.168 | 144 | 2,01 % |
| `+8` | 7.168 | 2 | 0,028 % |
| `+16` | 7.168 | 0 | 0 % |

Bei `+2` gab es zusätzlich 72 Läufe ohne zulässigen ersten Move. Sie
enden zwangsläufig wieder am Root. Für eine operative Aktorbilanz beträgt die
Rückkehr daher `639/7168 = 8,92 %`; `7,99 %` ist die bedingte Rate
`P(return | moved)`. Beide Größen sind legitim, dürfen aber nicht vermischt
werden.

In keinem Lauf war die symmetriekanonisierte Rückkehr größer als die exakte
Hash-Rückkehr. Der Class-Hash ändert die vorliegenden Zahlen also nicht.

## 2. Definition und Leakage-Audit

Für Root `x`, Cap `H` und einen zulässigen ersten Flip `i` berechnet der Code

```text
b_H(i) = #{j != i : Q(x flip i flip j) <= H}.
```

Die Root-Sensoren sind dann unter anderem

```text
entry_count              = #{i : Q(x flip i) <= H}
prospective_degree_sum   = sum_i b_H(i)
prospective_degree_max   = max_i b_H(i)
prospective_degree_mean  = sum_i b_H(i) / entry_count.
```

Die Berechnung findet in `_root_sensors()` statt, bevor irgendein
Replikat-RNG erzeugt oder `_run_excursion()` aufgerufen wird. Der Sensor kennt
weder den gewählten ersten Move noch Landing, Quench-Endpunkt oder Return-Label.
Es gibt daher **keine direkte Label-Leakage**.

`pair_qs_after_first()` zählt exakt die zulässigen Zwei-Flip-Endpunkte. Der
Rückflip `j=i` wird explizit auf Root-`Q` gesetzt und durch `-1` aus dem Grad
entfernt. Das stimmt mit der Tabu-Regel des zweiten Schrittes überein.

Es gibt jedoch eine absichtliche definitorische Nähe: Sensor und Outcome
verwenden denselben Q-Cap und denselben Single-Move-Graphen. Der Sensor misst
lokale Verzweigung genau der Dynamik, deren Rückkehr er vorhersagen soll. Das
ist kein Leakage, aber der Befund darf nur als **Policy-Topologiesensor** und
nicht als unabhängige GS4-Lösbarkeitsgröße interpretiert werden.

## 3. Replikate und Cross-Validation

Die acht Replikate eines Roots sind keine acht unabhängigen Landschaftspunkte.
Die vorhandene Auswertung macht das Wichtige richtig:

- genau eine Modellzeile pro Root;
- gruppierte Binomial-Likelihood mit `returns` von `trials`;
- kein Root wird zwischen Training und Test geteilt;
- Leave-one-`n`-out hält jeweils eine ganze Dimension zurück.

Die ausgegebene AUC expandiert die acht Outcomes rechnerisch wieder. Als
Punktschätzer für eine zufällige Exkursion ist das bei nahezu gleicher
Replikatzahl in Ordnung. Sie darf aber nicht so gelesen werden, als lägen
7.096 unabhängige Roots vor. Unsicherheiten müssen nach Root und zweckmäßig
stratifiziert nach `n` gebootstrapt werden.

Die globale AUC kann zusätzlich von unterschiedlichen Grundraten zwischen den
Dimensionen profitieren. Deshalb wurden unten auch AUCs innerhalb jedes `n`
berechnet und nach der Zahl positiver/negativer Outcome-Paare gewichtet.

`height_only` ist im Analyseskript irreführend benannt: Das Modell verwendet
`n` und `root_q`, nicht lediglich die Höhe. Außerdem ist Leave-one-`n`-out mit
einem linearen `n`-Term für `n=28` und `n=52` eine Extrapolation. Das ist kein
Leak, aber ein stärkerer Generalisierungstest als Interpolation zwischen
bekannten Größen.

## 4. Unabhängige Reproduktion des inkrementellen Werts

Es wurden Ridge-Binomialmodelle mit Root als Auswertungseinheit gerechnet.
Schwere Zählwerte wurden als `log1p` transformiert. Verglichen wurden:

```text
NR     = n + root_q
ENTRY  = NR + log(1 + entry_count)
TOPO   = ENTRY
         + log(1 + prospective_degree_mean)
         + log(1 + prospective_degree_max)
```

### Relative Höhe `+2`

| Leave-one-n-out-Modell | Log-Loss | globale AUC | innerhalb-n AUC |
| --- | ---: | ---: | ---: |
| `NR` | 0,21166 | 0,8460 | 0,8234 |
| `ENTRY` | 0,20781 | 0,8529 | 0,8302 |
| `TOPO` | **0,17057** | **0,8999** | **0,8877** |

Der gepaarte, nach `n` stratifizierte Root-Bootstrap für

```text
LogLoss(ENTRY) - LogLoss(TOPO)
```

ergab `0,03734`, 95-%-Intervall `[0,02662; 0,04873]`. Der Log-Loss wurde in
jeder der sieben einzeln gehaltenen Dimensionen besser; die Verbesserungen
lagen zwischen `0,0211` und `0,0520`.

Eine zweite achtfache Cross-Validation, die Roots innerhalb jeder Dimension
auf unterschiedliche Folds verteilt, ergab praktisch dasselbe:

```text
ENTRY Log-Loss = 0,20983
TOPO  Log-Loss = 0,17128
Delta           = 0,03852
Root-Bootstrap  = [0,02650; 0,05075].
```

Damit wird der Effekt bei `+2` weder nur von `n`, Root-`Q` noch von der Zahl
zulässiger erster Moves getragen. `prospective_degree_mean` trägt den größten
Teil des Zusatzsignals; `max` verbessert das Modell noch etwas. `sum` ist bei
gegebenem `entry_count` und `mean` algebraisch redundant.

### Relative Höhe `+4`

| Leave-one-n-out-Modell | Log-Loss | globale AUC | innerhalb-n AUC |
| --- | ---: | ---: | ---: |
| `NR` | 0,06821 | 0,9258 | 0,8879 |
| `ENTRY` | 0,06850 | 0,9298 | 0,8977 |
| `TOPO` | **0,06615** | **0,9322** | **0,8997** |

Der Zusatzgewinn ist nur `0,00235`, Bootstrap-Intervall
`[0,00045; 0,00437]`. Bei der Root-Fold-CV ist das Intervall
`[-0,00009; 0,00351]` und schließt null ein. In zwei gehaltenen Dimensionen
(`n=36,48`) wurde der Log-Loss sogar geringfügig schlechter.

Der robuste Befund lautet daher:

> Die lokale Zwei-Schritt-Topologie sagt bei der engen `+2`-Schale Root-
> Stickiness voraus. Bei `+4` ist ihr Zusatzwert klein; bei `+8/+16` fehlen
> positive Returns für eine sinnvolle Modellierung.

Der Bootstrap resampelt die bereits cross-validierten Vorhersagen. Er erfasst
die Variation der gehaltenen Roots, nicht vollständig die Unsicherheit einer
erneuten Modellanpassung. Weil beide CV-Schemata und alle sieben `+2`-
Dimensionen dasselbe Vorzeichen liefern, ändert diese Einschränkung den
qualitativen `+2`-Befund nicht.

## 5. Sensorarbeit fehlt bislang in der Kostenrechnung

Ein voller prospektiver Grad wertet für jeden zulässigen Einstieg weitere
`4n` Endpunkte aus. Für nur eine Höhe ist die nominale Arbeit daher

```text
4n * (1 + entry_count).
```

Bei `+2` beträgt sie im Datensatz je nach `n` im Mittel rund 1.160 bis 1.400
Endpunktbewertungen. Das sind 21 bis 25 Prozent der anschließenden mittleren
Walk-plus-Quench-Arbeit. Alle vier Höhen naiv vollständig zu sondieren kostet
im Mittel ungefähr 27.500 Bewertungen bei `n=28` und 50.200 bei `n=52` und ist
damit deutlich teurer als eine einzelne Exkursion.

Diese Kosten stehen weder in `run["evaluations"]` noch im bisherigen
Sensorvergleich. Der Sensor ist also prädiktiv, aber sein **Nettonutzen als
Aktorsteuerung ist noch nicht gezeigt**. Für einen Produktionsversuch sind
ein gesampelter K4/K8-Grad oder inkrementell wiederverwendete Paar-Q-Vektoren
wahrscheinlicher sinnvoll als ein voller Mehrhöhen-Scan.

## 6. Keine Aussage über Solve-Nähe

Im gesamten Datensatz entstanden nur neun Lösungen:

```text
+2: 4, +4: 3, +8: 1, +16: 1.
```

Sie liegen zudem fast ausschließlich bei den kleinsten Dimensionen. Damit kann
weder ein Solve-Modell gefittet noch ein Zusammenhang zwischen
prospektivem Grad und Lösungswahrscheinlichkeit belastbar geprüft werden.

Besonders wichtig: `+16` wechselt in 100 Prozent der Läufe das exakte Basin,
findet aber nur eine Lösung. **Basinwechsel und Solve-Nähe sind verschiedene
Ziele.** Der neue Sensor ist aktuell ein Sensor für Rückkehr/Stickiness, nicht
für den Committor zur Lösungsmenge.

## 7. Kleinster fairer adaptiver Aktor-Test

Der nächste Test sollte nicht sofort den Production-Solver umbauen. Die
kleinste saubere Ablation vergleicht drei Policies auf frischen Roots:

```text
P2: immer relative Höhe +2
P4: immer relative Höhe +4
PA: +2, wenn ein cross-fitted Sensor geringe Rückkehr vorhersagt,
    sonst +4
```

Minimaler Versuchsaufbau:

1. Neue Root-Kohorte aus denselben `n`, zusätzlich separat eine Kohorte aus
   terminalen Production-Solver-Roots. Die beiden Verteilungen nicht mischen.
2. Modell und Entscheidungsschwelle ausschließlich auf alten Dimensionen oder
   Trainings-Roots festlegen; Test-Root-Outcomes bleiben unsichtbar.
3. Für `PA` zunächst nur einen K4- oder K8-Schätzer des mittleren
   prospektiven Grades berechnen. **Jede Sensorbewertung zählt gegen dasselbe
   Candidate-Evaluation-Budget.**
4. Pro Root und Policy gepaarte, vorab abgeleitete Seeds verwenden. Wegen
   unterschiedlicher Kandidatenmengen sind die Pfade nicht identisch; die
   statistische Einheit bleibt der Root.
5. Primäres Ziel: neue exakte Basin-Hashes pro gesamter Candidate-Arbeit.
   Sekundär: verschiedene Zielbasins, Return-Rate, Hamming-Distanz und
   Quench-End-`Q`.
6. Danach jedem neuen Basin exakt dasselbe kleine Production-Solver-Budget
   geben. Erst diese Stufe misst, ob der gesteuerte Wechsel mehr Solve-Hazard
   erzeugt als bloß irgendein Wechsel.

Eine sinnvolle vorab festgelegte Startregel ist:

```text
wenn entry_count(+2) == 0: +4
sonst wenn geschätztes R(+2) <= 0,35: +2
sonst: +4
```

`0,35` ist kein Naturwert. Er folgt grob aus der beobachteten Kostenrelation:
`+2` ist trotz höherer Rückkehr mit etwa 5.638 mittleren Bewertungen pro Lauf
billiger als `+4` mit etwa 8.808. Die Schwelle muss vor dem Test eingefroren
oder ausschließlich innerhalb des Trainingsfolds optimiert werden.

Als härtere zweite Stufe sollten relative Höhen `0,1,2,3,4` vermessen werden.
Die jetzigen Daten zeigen bei allgemeinen Greedy-Minima den Übergang bereits
zwischen `0` und `+2`; die groben Höhen `2,4,8,16` sind für eine individuelle
Schätzung von `H_esc` beziehungsweise `H50` zu weit auseinander.

## 8. Belastbare Interpretation

Belastbar ist:

1. Der deterministische Quench definiert operative Basins sauber.
2. Enge Sublevel-Topologie trägt messbare Information darüber, ob die aktuelle
   randomisierte Shell-Policy in dasselbe operative Basin zurückkehrt.
3. Der Effekt generalisiert bei `+2` über alle sieben getesteten Dimensionen.

Noch nicht belastbar ist:

1. dass der Sensor Solve-Nähe misst;
2. dass er nach Einrechnung seiner Kosten den Solver verbessert;
3. dass dieselbe Beziehung für Production-Tabu-Endzustände oder Q1-Minima gilt;
4. dass die beobachtete Return Transition eine intrinsische, quenchunabhängige
   Basin-Grenze ist.

Der Fund ist damit kleiner als ein universeller Basin-Kompass, aber größer als
eine weitere Darstellung von `Q`: Er ist ein validierter lokaler Sensor für
die policyabhängige Topologie. Der nächste Erkenntnisschritt ist ein fair
budgetierter Aktortest, nicht ein weiteres Korrelationsmodell.
