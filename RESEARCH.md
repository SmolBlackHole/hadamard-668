# Solver-Suchdynamik — Systematische Verbesserung

## Ausgangslage

Sweep mit 10 Seeds, 200k Steps zeigt steilen Abfall ab n=30:

| n | Ordnung | Gelöst |
| ---: | :---: | ---: |
| 40 | 160 | 0/10 |
| 38 | 152 | 0/10 |
| 36 | 144 | 2/10 |
| 34 | 136 | 0/10 |
| 32 | 128 | 4/10 |
| 30 | 120 | 9/10 |
| 28 | 112 | 10/10 |
| 26 | 104 | 10/10 |
| 24 | 96 | 10/10 |

n=28 zuverlässig, danach Klippe. n=36 mit 2/10 wirkt wie Glückstreffer.

Vermutung: kein reines Budget-Problem, sondern Suchdynamik-Problem — Solver
findet lokales Minimum, aber Kick/Restart kommen nicht ins richtige Becken.

---

## Referenz: pzinn/hadamard

Öffentliches Repository: `pzinn/hadamard` auf GitHub. Das zugehörige Paper
(arXiv:2604.11101, "Generating Hadamard matrices with transformers") verlinkt
Code und Laufdaten ausdrücklich.

Deren **lokaler Solver** (`improve.py`) erreicht n=50 in ~6h auf zirkulanten
Konstruktionen. Die vier Kernkomponenten:

### 1. Gray-Code Subset-Rescue

Statt wie wir nur Pairs und Triples zu testen, bewertet deren Optimierer
zunächst alle Single-Flips und nimmt die besten

```python
k = min(11, na)
```

Kandidaten. Dann testet er per Gray-Code **jede nichtleere Teilmenge** dieser
Bits: `2^11 - 1 = 2047` Kombinationen.

Unser KFlip beschränkt sich auf Breite 2 und 3. Deren Variante fragt dagegen:
"Unter den 11 vielversprechendsten Bits, hilft irgendeine beliebige
Kombination?"

**Gray-Code-Mechanismus**: Zwischen zwei aufeinanderfolgenden Kombinationen
ändert sich genau ein Bit. Dadurch kann das Spektrum (bzw. die Energie)
inkrementell aktualisiert werden:

```text
Kombination 1:  {Bit 3}
Kombination 2:  {Bit 3, Bit 7}   ← Bit 7 hinzu
Kombination 3:  {Bit 7}          ← Bit 3 weg
...
```

Pro Schritt wird nur ein Band hinzugefügt oder entfernt — O(b) statt O(b²)
für die Cross-Terms. Das ist architektonisch sehr nah an unserem
`_combo_delta`, das ebenfalls Bänder einzeln verarbeitet.

**Empfohlene Startgrößen für uns**:

```text
k=8  → 255 Kombinationen
k=9  → 511 Kombinationen
k=10 → 1023 Kombinationen
```

Nicht sofort k=11 nehmen. Erst bei n=32,36 testen.

**Implementierungsstufen**:

1. Naiv mit `_combo_delta()` pro Teilmenge (einfach, korrekt)
2. Gray-Code mit inkrementellem `_single`-Add/Remove (schnell, komplex)

Erst Wirkung nachweisen, dann optimieren.

### 2. Tabu-Walk

Bei einem lokalen Minimum (kein Single-Flip verbessert, Rescue hilft nicht):

```text
→ Bestes erlaubtes Single-Bit flippen (auch wenn es schlechter wird)
→ Kürzlich verwendete Bits für Tenure ~n/4 Schritte tabu setzen
→ Besten unterwegs gefundenen Zustand behalten
→ 50-200 Schritte laufen
→ Danach entweder zum besten Zustand zurück oder Hard-Restart
```

Der Tabu-Walk nimmt bei jedem Schritt den besten momentan erlaubten
Single-Flip, **selbst wenn dieser die Energie verschlechtert**. Kürzlich
verwendete Bits erhalten eine Strafe, damit der Walker nicht sofort hin und
her flippt.

Das passt extrem gut zu unserer derzeitigen Schwachstelle:

```text
Single-Minimum erreicht
→ Pair/Triple-Rescue hilft nicht
→ derzeit: 3 Kicks, dann kompletter Neustart (alle 4 Sequenzen neu)
```

Stattdessen:

```text
Single-Minimum
→ 50-150 Tabu-Schritte
→ besten unterwegs gefundenen Zustand übernehmen
→ erst danach Restart (wenn immer noch kein besserer Zustand)
```

Vermutlich der **stärkste Einzelkandidat** — unser Problem ist nicht "kein
Weg existiert", sondern "der Solver findet das richtige Becken nicht". Der
Tabu-Walk ist ein gezielter Spaziergang aus dem falschen Becken heraus,
während unser Kick ein zufälliger Tritt mit verbundenen Augen ist.

Testparameter: 50, 100, 200 Schritte. Tenure: n/4.

### 3. Ganzfolgen-Spektralreparatur

Die stärkste nichtlokale Bewegung in deren Code ersetzt eine komplette
GS-Folge:

1. Spektren von drei Folgen festhalten
2. Berechnen, welches Spektrum die vierte Folge idealerweise haben müsste
3. Zufällige Phasenstörung anwenden
4. Inverse FFT
5. Auf ±1 runden
6. Übernehmen, falls der Score besser wird

Das Paper beschreibt diesen Blocksprung sowie Parallel Tempering zusätzlich
zur lokalen Bitflip-Suche.

**Nicht direkt übertragbar**: Deren Builder ist **zirkulant**, unserer ist
**negazyklisch**. Die Fourier-Auswertung müsste an die negazyklischen
Frequenzen angepasst werden — also an die Nullstellen von `z^n + 1` statt an
die gewöhnlichen n-ten Einheitswurzeln.

Das wäre der **größte algorithmische Hebel**, aber auch die aufwändigste
Änderung. Unser Solver könnte dann zwischen zwei Ebenen wechseln:

```text
Lokal:   Einzelne Bits und kleine Kombinationen (schnell, häufig)
Nichtlokal: Eine komplette Folge spektral rekonstruieren (teuer, selten)
```

Zurückgestellt bis die lokale Suche ausgereizt ist.

### 4. Adaptives K

`K = sqrt(B) * 3`, nach mehreren aufeinanderfolgenden Rescue-Hits erweitert.
Unser Code hat das bereits teilweise (`rescue_mode`), aber die Wirksamkeit
ist unklar (siehe Schritt 7: rescue_mode-Ablation).

---

## Was wir NICHT sofort übernehmen sollten

Diese Komponenten aus pzinn/hadamard sind für uns aktuell nicht sinnvoll:

| Komponente | Grund |
| --- | --- |
| **Transformer** | 1M parallele Kandidaten — Infrastruktur bevor wir wissen welche Suchbewegungen helfen |
| **Parallel Tempering** | Mehrere Temperatur-Ketten — Overkill solange die Basis-Suche nicht stabil ist |
| **Feste Segment-Summen** | Beruht auf dem Fourier-Nullmodus zirkulanter Konstruktionen — nicht auf negazyklisch übertragbar |
| **1M parallele Kandidaten** | Braucht GPU-Infrastruktur — lohnt sich erst wenn die Suchstrategie steht |

---

## Baseline (Schritt 0)

Für n=32,34,36 mit 200k Steps, gleiche Seeds, gleiche Hardware erfassen:

```text
solved, best_energy, evals, elapsed_s,
singles, pairs, triples, kicks, restarts
```

Restenergie bei Fehlschlägen ist die wichtigste Kennzahl.

```bash
PYTHONPATH=src python scripts/bench.py --sweep gs4 32 34 36 --seeds 10 --dataset data/baseline.json
```

---

## Testkaskade (nur eine Änderung pro Lauf)

### 1. Single-Scan-Reihenfolge

Aktuell: Bits immer in derselben Reihenfolge, erstes besseres wird genommen.

**Variante A**: Zufälliger Startindex pro Sweep, dann zyklisch (fast kostenlos,
weniger verzerrt).

**Variante B**: Komplette Reihenfolge pro Sweep mischen.

Datei: `solver.py` — `_positions` und Scan-Loop (Zeile 88)

### 2. Echter Kick

Aktuell: Kick nur übernehmen wenn `_combo_delta < cur_e`. Das ist ein
zufälliger Mehrbit-Move mit falschem Namensschild, kein echter Kick.

**Variante A**: Kick immer übernehmen (best_seq bleibt erhalten, current darf
schlechter werden). Der Solver läuft dann aus dem verschlechterten Zustand
weiter — dadurch kann er Becken wechseln.

**Variante B**: Adaptive Kick-Stärke:

```text
1 Bit pro Sequenz → 2 Bits → 4 Bits → Restart
```

Datei: `solver.py` — Phase 4 (Zeile 136-169)

### 3. Teil-Restart

Aktuell: Nach 3 gescheiterten Kicks werden alle 4 Sequenzen neu gewürfelt.

**Variante A**: Eine zufällige Sequenz neu.
**Variante B**: Zwei Sequenzen neu.
**Variante C**: Schlechteste Sequenz neu (braucht row-energy Metrik).

Zuerst Variante A testen. Vielleicht sind 3 Folgen schon brauchbar und werden
aus Prinzip verbrannt.

Datei: `solver.py` — `kick_streak >= 3` Block (Zeile 154-160)

### 4. Tabu-Walk (aus pzinn/hadamard)

```text
lokales Minimum erreicht
→ bestes erlaubtes Single-Bit flippen (auch wenn schlechter)
→ kürzlich verwendete Bits tabu setzen (Tenure ~n/4)
→ besten unterwegs gefundenen Zustand behalten
→ 50-200 Schritte, dann zurück zum Besten oder Restart
```

Parameter: 50, 100, 200 Schritte.

Vor dem Hard-Restart einsetzen. Vermutlich stärkster Einzelkandidat — das
Problem ist nicht "kein Weg", sondern "falsches Becken".

Datei: neu `src/tabu.py` oder in `solver.py` integrieren

### 5. Top-k Subset-Rescue (aus pzinn/hadamard)

Statt nur Pairs/Triples: ALLE 2^k-1 Teilmengen der besten k Single-Kandidaten
testen.

```text
k=8  → 255 Kombinationen
k=9  → 511 Kombinationen
k=10 → 1023 Kombinationen
```

**Stufe 1**: Naiv mit `_combo_delta()` pro Teilmenge. `itertools.combinations`
für jede Größe 1..k durchiterieren.

**Stufe 2** (nach Wirkungsnachweis): Gray-Code mit inkrementellem
Single-Band-Update. Ein Bit ändert sich pro Schritt → O(b) statt O(b²) für
Cross-Terms. Architektonisch nah an unserem `_combo_delta`-Design.

Zuerst k=8 bei n=32,36 testen.

Datei: `solver.py` — `_rescue` ersetzen/erweitern

### 6. K/K3-Tuning

Aktuell: `K = min(B-1, max(16, int(B**0.5 * 3)))`, `K3 = min(K//2, 10)`.

Testen: K = 3√B, 4√B, 5√B; K3 = 8, 10, 12.

Nicht alle Kombinationen, erst K variieren, dann K3.

Datei: `solver.py` Zeile 73-74

### 7. rescue_mode-Ablation

Im aktuellen Code: `rescue_mode` + `narrowed`-Pool. Prüfen ob `narrowed`
praktisch identisch mit `top_candidates` ist (dann ist der Mode wirkungslos).

**Variante A**: rescue_mode komplett entfernen.
**Variante B**: Best-Mode nur auf `top_candidates[:K//2]`.

Könnte Rechenzeit sparen oder den Deep-Rescue tatsächlich sinnvoll machen.

Datei: `solver.py` — rescue_mode, narrowed, Phase 2/3

### 8. Ablationen (Beitrag jeder Komponente)

```text
Singles only
Singles + Pairs
Singles + Pairs + Triples
ohne Kick
ohne Restart
mit Tabu statt Kick
mit Subset-Rescue statt Pair/Triple
```

Kernfrage: Wie viele erfolgreiche n=36 Runs brauchen ≥1 Triple-Move?
Falls ≈0: Triple-Code ist teurer Ballast.

### 9. Budget-Skalierung

Erst NACHDEM Suchpfade verbessert wurden:

```text
200k → 500k → 1M
```

Für n=32,34,36. Unterscheidet Rechenlimit von Suchdynamiklimit:

```text
Mehr Budget hilft stark → Rechenlimit (einfach mehr Steps geben)
Mehr Budget hilft kaum → Suchdynamiklimit (braucht bessere Moves)
```

### 10. Negazyklische Spektralreparatur (längerfristig)

Aus pzinn/hadamard: Eine ganze Folge aus den Spektren der anderen drei
rekonstruieren.

**Mathematischer Unterschied**:

- Zirkulant: Fourier-Basis = Einheitswurzeln `z^n = 1`
- Negazyklisch: Fourier-Basis = Nullstellen von `z^n + 1 = 0`

Die negazyklischen Frequenzen sind `exp(iπ(2k+1)/n)` statt `exp(i2πk/n)`.
Die Spektralbedingungen für die GS4-Konstruktion müssen an diese Wurzeln
angepasst werden.

Größter algorithmischer Hebel, aber aufwändig. Zurückgestellt bis lokale
Suche ausgereizt ist.

---

## Priorisierte Reihenfolge

```text
0. Baseline einfrieren
1. Zufälliger Scan-Start (billig, testet Verzerrung)
2. Kick immer akzeptieren
3. Teil-Restart (eine Sequenz)
4. Tabu-Walk (stärkster Kandidat)
5. Top-8 Subset-Rescue
6. K/K3-Tuning
7. rescue_mode-Ablation
8. Gewinner kombinieren
9. Größeres Budget
10. Spektralreparatur (langfristig)
```

Immer nur **eine Änderung pro Testlauf**. Kein "SuperSolver" bei dem niemand
weiß ob es am Tabu, am Kick oder am Seed lag.

---

## Referenzen

- `pzinn/hadamard` auf GitHub — `improve.py`: Tabu-Walk + Gray-Code-Rescue +
  Spektralreparatur + Parallel Tempering
- arXiv:2604.11101 — "Generating Hadamard matrices with transformers"
- Lokaler Solver dort: n=50 in ~6h (zirkulant, nicht negazyklisch)
- Unser negazyklischer GS4-Raum ist strukturell anders (z^n+1 statt z^n=1),
  aber die Suchstrategien sind übertragbar

## Dateien

| Datei | Änderung |
| --- | --- |
| `src/solver.py` | Scan-Order, Kick-Logik, Restart, Rescue, K-Parameter |
| `src/tabu.py` (neu) | Tabu-Walk als Escape-Phase |
| `scripts/bench.py` | Baseline-Sweep mit Stats-Export |
| `data/baseline.json` | Baseline-Ergebnisse |
