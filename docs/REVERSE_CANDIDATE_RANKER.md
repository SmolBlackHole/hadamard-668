# Audit: Candidate-Ranker aus dem Reverse-Curriculum

## Kurzfassung

Ein streng leakage-freier linearer Candidate-Ranker wurde auf bekannten
Reverse-First-Moves trainiert und sowohl mit ganzen zurückgehaltenen
Ursprungslösungen als auch mit vollständig zurückgehaltenen `n` geprüft.

Das Ergebnis ist nützlich gerade weil es kein neuer Sieger ist:

- Der lineare Ranker schlägt den reinen `lookahead_q` nicht.
- Cross-`n` bei Depth 3 erreicht er `84,1 %` Top-1 gegenüber `85,6 %` für
  `lookahead_q`.
- Im Achter-Tournament erreicht er `12,35 %` gegenüber `13,47 %`.
- Auf neuen freien Root-Shells ist der Ranker nach 50k Downstream sogar
  signifikant schlechter als Lookahead (`7` Ranker-only gegen `22`
  Lookahead-only, exakter gepaarter p-Wert `0,0081`).

Der robuste Befund ist deshalb:

> Die lokale Reverse-Geometrie wird fast vollständig durch den exakten
> Two-ply-Wert getragen. Die zusätzlich getesteten Kanal-, Symbol-, Kanten- und
> D-Aggregate verbessern seine Übertragbarkeit nicht.

Der reine Lookahead zeigt im Downstream einen kleinen positiven Solve-Hazard-
Trend, dieser ist mit der Pilotgröße von 96 Roots aber noch nicht signifikant.

## 1. Datensatz und Leakage-Schutz

Aus den archivierten Lösungen wurden pro `(Provenienz,n)` höchstens acht
SHA-eindeutige Ursprungslösungen verwendet. Für jede wurden je zwei zufällige
nichtgelöste Zustände in den Hamming-Tiefen `2,3,4,5` erzeugt:

```text
66 Ursprungslösungen
× 4 Tiefen
× 2 Zustände
= 528 Ranking-Zustände
```

Trainiert wurde ausschließlich auf den `132` Zuständen der Tiefe drei. Ein
Candidate ist positiv, wenn sein Index zu den drei absichtlich geflippten Bits
gehört und daher ein korrekter erster Schritt auf einem garantierten Rückweg
ist.

Der Reverse-Index wurde ausschließlich zur Labelbildung verwendet. Kein
Feature enthält:

- den bekannten Pfad,
- die Hamming-Differenz zur Ursprungslösung,
- den Ursprungslösungs-Hash,
- oder eine Zielsequenz.

Jeder beschädigte Zustand und seine Wiederherstellung wurden mit einem frischen
Tracker geprüft.

### Cross-Validation

Zwei getrennte Prüfungen wurden verwendet:

1. **Leave-one-n-out:** Ein komplettes `n` einschließlich sämtlicher seiner
   Ursprungslösungen bleibt beim Training unsichtbar.
2. **Source-group 5-fold:** Alle Zustände derselben Ursprungslösung liegen
   vollständig entweder in Train oder Test.

Damit kann der Ranker weder Repräsentationen derselben Lösung noch dieselbe
Ordnung als Abkürzung wiedererkennen.

## 2. Candidate-Features

Alle Features sind vor Ausführung des ersten Moves aus dem aktuellen Zustand
berechenbar:

- unmittelbares `Q_i`;
- `min_j Q_ij`;
- Zahl der Zwei-Flip-Endpunkte unter `Q`, `Q+2`, `Q+4`;
- `||d_i||²`, `<u,d_i>` und normierte Gegenrichtung zu `u`;
- Norm des betroffenen Einzelresiduums `rho_s` vor/nach dem Kandidaten;
- Spaltenparität `beta`, Kandidatenbit und linkes/rechtes Kantenprodukt;
- H4-Spaltentyp über Maximum und Support der transformierten Spalte;
- gleiche beziehungsweise entgegengesetzte D-Zeilen;
- exakte und beste angenäherte cross-sequence Cancellation-Zeilen;
- Sequenz-ID als vier One-Hot-Komponenten.

Ein candidate-spezifischer vollständiger Rebuild von `u(Fx)` wurde bewusst
nicht verwendet: Das wäre für jeden der `4n` Kandidaten wesentlich teurer und
kein billiger Online-Score. Die candidate-spezifische Einzelkanaländerung ist
über `rho_s + 4d_i` enthalten.

Der Ranker ist ein gewichtetes lineares Ridge-Modell. Positive und negative
Candidates tragen gleiches Gesamtgewicht. Das Modell ist daher leicht
inspizierbar und keine versteckte nichtlineare Pipeline.

## 3. Ranking-Ergebnis bei Depth 3

Aggregiert über die vollständig zurückgehaltenen `n`:

| Policy | Top-1-Hit | K8-Hit |
| --- | ---: | ---: |
| Random | 1,52 % | 1,94 % |
| `lookahead_q` | **85,61 %** | **13,47 %** |
| linearer Ranker | 84,09 % | 12,35 % |

Source-group 5-fold ergibt dasselbe Bild:

| Policy | Top-1-Hit | K8-Hit |
| --- | ---: | ---: |
| Random | 5,30 % | 1,80 % |
| `lookahead_q` | **86,36 %** | **13,59 %** |
| linearer Ranker | 84,09 % | 13,14 % |

Die Top-1-Randomrate schwankt wegen nur 132 Zuständen; die über 8.448
Tournament-Wiederholungen gemessene K8-Rate liegt stabil nahe der theoretischen
Reverse-Dichte.

### Leave-one-n-out im Detail

| zurückgehaltenes n | Ranker Top-1 | Lookahead Top-1 |
| ---: | ---: | ---: |
| 28 | 68,8 % | 75,0 % |
| 30 | 50,0 % | 56,3 % |
| 32 | 68,8 % | 75,0 % |
| 44 | 93,8 % | 100 % |
| 46 | 93,8 % | 87,5 % |
| 48 | 100 % | 93,8 % |
| 50 | 100 % | 100 % |
| 52 | 95,0 % | 95,0 % |

Der Ranker extrapoliert bei den großen Ordnungen gut, aber nicht besser als die
bereits exakte Two-ply-Geometrie. Besonders bei den kleinen Symbolsolver-
Lösungen ist die Übertragung schwächer.

## 4. Was das lineare Modell tatsächlich lernt

Die betragsmäßig größten standardisierten Gewichte sind:

| Feature | Gewicht |
| --- | ---: |
| `min_two_ply_Q` | -0,432 |
| `<u,d_i>` | +0,348 |
| Pair-Count unter `Q+4` | +0,296 |
| negative Alignment-Richtung | +0,276 |
| unmittelbares `Q_i` | +0,200 |
| Pair-Count unter `Q+2` | -0,138 |
| ` | | d_i | | ²` | +0,101 |

Die übrigen Einzelresidual-, Symbol-, Kanten- und D-Features besitzen nur sehr
kleine Gewichte. Wegen starker Kollinearität sollten die Vorzeichen nicht als
einzelne mathematische Gesetze gelesen werden.

Der strukturelle Befund ist dennoch klar: Der Ranker ist im Wesentlichen

```text
Two-ply-Q + lokale Pair-Dichte + kleine lineare Korrekturen.
```

Diese Korrekturen helfen auf den Trainingsnachbarschaften gelegentlich, senken
aber die robuste cross-`n`-Leistung.

## 5. Andere Damage-Tiefen

Das auf Depth 3 trainierte Endmodell wurde deskriptiv auch auf anderen Tiefen
derselben Source-Stichprobe geprüft:

| Tiefe | Ranker Top-1 | Lookahead Top-1 | Ranker K8 | Lookahead K8 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 100 % | 100 % | 10,3 % | 9,5 % |
| 3 | 84,8 % | 79,5 % | 12,6 % | 13,3 % |
| 4 | 66,7 % | 69,7 % | 15,0 % | 15,5 % |
| 5 | 54,5 % | 53,8 % | 16,1 % | 15,3 % |

Depth 2 ist konstruktionsbedingt trivial: Zwei bekannte Reverse-Moves ergeben
direkt `Q=0`. Bei wachsenden Tiefen sinkt die volle Top-1-Erkennung, während
die K8-Trefferrate wegen mehr positiver Reverse-Candidates steigt. Diese
Auswertung ist keine unabhängige Source-CV für die anderen Tiefen und daher nur
eine Geometriekontrolle.

## 6. Offline-Shell plus 50k Downstream

Der endgültige Ranker wurde ausschließlich offline als Move-Selector in einer
`Q_root+4`-Shell verwendet. Production-Code wurde nicht verändert.

Für `n=38,40,42` entstanden je 32 neue zufällige Zustände, die zunächst mit
dem festen Best-Improvement-Quench zu freien Roots gebracht wurden. Verglichen
wurden:

- direkter 50k-Downstream vom Root;
- zufällige 64-Move-Shell;
- `lookahead_q`-Shell mit Achter-Tournament;
- Ranker-Shell mit Achter-Tournament.

Alle Downstream-Läufe verwendeten `SolverConfig(targeted_escape=False)`. Für
alle vier Policies eines Roots wurde derselbe Solver-RNG-Seed verwendet. Jede
zurückgegebene Lösung beziehungsweise Endenergie wurde frisch rebuilt.

### Solve-Hazard

| n | Direkt | Random-Shell | Lookahead-Shell | Ranker-Shell |
| ---: | ---: | ---: | ---: | ---: |
| 38 | 23/32 | 20/32 | **25/32** | 16/32 |
| 40 | 6/32 | **10/32** | 9/32 | 6/32 |
| 42 | 2/32 | 2/32 | **3/32** | 0/32 |
| gesamt | 31/96 | 32/96 | **37/96** | 22/96 |

Gepaarte exakte Vergleiche über alle 96 Roots:

- Lookahead gegen direkt: `15` Lookahead-only, `9` Direct-only,
  `p=0,307`;
- Lookahead gegen Random: `18` Lookahead-only, `13` Random-only,
  `p=0,473`;
- Ranker gegen Lookahead: `7` Ranker-only, `22` Lookahead-only,
  `p=0,0081`.

Damit gibt es einen konsistent positiven, aber noch unbewiesenen Trend für den
reinen Lookahead. Der lineare Ranker ist in dieser Form klar falsifiziert: Er
verschlechtert die relevante Solve-Hazard signifikant, obwohl er bekannte
Reverse-Candidates offline sehr gut erkennt.

Die Shell-Kosten sind in diesen Solve-Zahlen noch nicht gegen den 50k-Budget
gerechnet. Das begünstigt Lookahead und Ranker. Ohne signifikanten
Solve-Hazard-Gewinn besteht daher erst recht kein Performance-Argument für den
Ranker.

## 7. Konsequenz

Der Ranker sollte nicht in den Solver übernommen werden. Er beantwortet die
falsche Verallgemeinerungsfrage zu gut:

> „Sieht dieser Move wie ein Reverse-Move in einer künstlich beschädigten
> bekannten Lösung aus?“

Das reicht nicht für:

> „Erhöht dieser Move von einem freien Production-Root die Chance, innerhalb
> von 50k eine Lösung zu erreichen?“

Der kleine weiterzuverfolgende Lead ist ausschließlich `lookahead_q` selbst.
Der nächste Test sollte dessen positiven `37/96`-Trend mit mehr gepaarten Roots
und candidate-kostenfairer Budgetierung prüfen. Zusätzliche lineare
Reverse-Features sind bis dahin verworfen.

## Reproduktion

```text
python -m experiments.learning.reverse_candidate_ranker \
  --output data/reverse-candidate-ranker.json \
  --sources-per-group 8 --paths-per-depth 2 \
  --downstream-ns 38 40 42 --downstream-roots 32 \
  --downstream-steps 50000 --seed 20260816
```

Dateien:

- `experiments/learning/reverse_candidate_ranker.py`
- `data/reverse-candidate-ranker.json`
