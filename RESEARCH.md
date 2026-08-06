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

## Baseline (Schritt 0) — ERGEBNISSE

**Erste Baseline war fehlerhaft** (Commit `af00bf0`): `generator.py` baute
bei Fehlschlägen (`best_e > 0`) eine `np.ones()`-Dummy-Matrix statt
`builder.build(best_seq)`. `check_orthogonality()` lieferte daher die Energie
der All-Eins-Matrix — ein Faktor-2-Bug im Reporting.

Gefixt: `metrics.energy` = reale Gram-Energie von `best_seq`, `solver_e` =
interne Tracker-Energie. Baseline wird mit Fix neu gerechnet.

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
ändert sich genau ein Bit. Dadurch kann die Energie inkrementell aktualisiert
werden:

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

Nicht sofort k=11 nehmen und den Rechner wegen menschlicher Ungeduld anzünden.
Erst bei n=32,36 testen.

**Implementierungsstufen**:

1. Naiv mit `_combo_delta()` pro Teilmenge (einfach, korrekt)
2. Gray-Code mit inkrementellem `_single`-Add/Remove (schnell, komplex)

Erst Wirkung nachweisen, dann drei Nächte lang optimieren.

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

Vermutlich der **stärkste Einzelkandidat** — unser Baseline-Befund zeigt ein
degeneriertes Rang-1-Plateau, aus dem der Solver mit reinem Descent nicht
entkommt. Der Tabu-Walk ist ein gezielter Spaziergang aus dem Plateau heraus,
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

| Komponente | Grund |
| --- | --- |
| **Transformer** | 1M parallele Kandidaten — Infrastruktur bevor wir wissen welche Moves helfen |
| **Parallel Tempering** | Mehrere Temperatur-Ketten — Overkill solange die Basis-Suche nicht stabil ist |
| **Feste Segment-Summen** | Beruht auf dem Fourier-Nullmodus zirkulanter Konstruktionen — nicht übertragbar |
| **1M parallele Kandidaten** | Braucht GPU-Infrastruktur — lohnt sich erst wenn die Suchstrategie steht |

---

## Testkaskade (nur eine Änderung pro Lauf)

### 1. Single-Scan-Reihenfolge ✗ (abgeschlossen — keine Verbesserung)

Variante A (zufälliger Startindex pro Sweep, dann zyklisch) getestet mit
n=32,34,36, 10 Seeds, 200k Steps gegen Baseline.

Ergebnis: 4/30 gelöst (vs. 6/30 Baseline). Kein systematischer Gewinn —
die Scan-Reihenfolge verschiebt nur welche Seeds erfolgreich sind.

Code reverted. Schritt 1 ist abgehakt — keine weiteren Scan-Varianten testen.

### 2. Echter Kick

Aktuell wird ein Kick nur übernommen, wenn er sofort besser ist. Das ist eher
ein zufälliger Mehrbit-Move mit falschem Namensschild.

```text
Baseline: Kick nur übernehmen, wenn besser
Variante A: Kick immer übernehmen
Variante B: Kick-Stärke adaptiv erhöhen
```

Adaptive Variante etwa:

```text
1 Bit pro Sequenz
2 Bits pro Sequenz
4 Bits pro Sequenz
danach Restart
```

`best_seq` bleibt erhalten, also darf der aktuelle Zustand ruhig vorübergehend
schlechter werden.

Datei: `solver.py` — Phase 4

### 3. Teil-Restart statt kompletter Restart

Aktuell wird nach mehreren gescheiterten Kicks alles neu gewürfelt.

```text
Baseline: alle vier Sequenzen neu
Variante A: eine zufällige Sequenz neu
Variante B: zwei Sequenzen neu
Variante C: schlechteste Sequenz neu
```

Erst Variante A. Vielleicht sind drei Folgen schon brauchbar und du verbrennst
sie momentan aus Prinzip.

Datei: `solver.py` — `kick_streak >= 3` Block

### 4. Tabu-Walk (aus pzinn/hadamard)

```text
lokales Minimum erreicht
→ bestes erlaubtes Single-Bit flippen
→ auch wenn es schlechter wird
→ kürzlich verwendete Bits tabu setzen (Tenure ~n/4)
→ besten unterwegs gefundenen Zustand behalten
```

Testparameter zunächst:

```text
50 Schritte
100 Schritte
200 Schritte
```

Tabu-Walk vor dem Hard-Restart einsetzen.

Das ist wahrscheinlich einer der stärksten Kandidaten, weil dein aktuelles
Problem offenbar nicht "kein Weg", sondern "falsches Becken" ist. Die
Baseline bestätigt das: ein degeneriertes Rang-1-Plateau, aus dem der Solver
mit reinem Descent nicht entkommt.

Datei: neu `src/tabu.py` oder in `solver.py` integrieren

### 5. Top-k Subset-Rescue (aus pzinn/hadamard)

Dein Solver testet derzeit Pairs und Triples. Deren Solver nimmt die besten k
Single-Kandidaten und testet alle Teilmengen.

Start klein:

```text
k=8  → 255 Kombinationen
k=9  → 511 Kombinationen
k=10 → 1023 Kombinationen
```

Erst `k=8`.

Zunächst naiv mit `_combo_delta()` testen. Falls es hilft, danach Gray-Code
beziehungsweise inkrementelle Updates bauen. Erst Wirkung nachweisen, dann
optimieren. Eine ungewohnte, aber brauchbare Reihenfolge.

Datei: `solver.py` — `_rescue` ersetzen/erweitern

### 6. K/K3-Tuning

Aktuell ungefähr:

```python
K = min(B-1, max(16, int(B**0.5 * 3)))
K3 = min(K // 2, 10)
```

Testen:

```text
K = 3√B, 4√B, 5√B
K3 = 8, 10, 12
```

Nicht alle Kombinationen. Erst K verändern, danach K3.

Datei: `solver.py` — K, K3 Initialisierung

### 7. rescue_mode-Ablation

Dein `narrowed`-Pool ist praktisch identisch mit `top_candidates`.

```text
Baseline: aktueller Modus
Variante A: rescue_mode komplett entfernen
Variante B: Best-Mode nur auf top_candidates[:K//2]
```

Das könnte Rechenzeit sparen oder den Deep-Rescue tatsächlich sinnvoll machen.

Datei: `solver.py` — rescue_mode, narrowed, Phase 2/3

### 8. Ablationen (Beitrag jeder Komponente)

Damit findest du heraus, welche Teile wirklich etwas beitragen:

```text
Singles only
Singles + Pairs
Singles + Pairs + Triples
ohne Kick
ohne Restart
mit Tabu statt Kick
mit Subset-Rescue statt Pair/Triple
```

Besonders wichtig:

> Wie viele erfolgreiche (n=36)-Runs benötigen mindestens einen Triple-Move?

Die Baseline sagt ≈0 — der Triple-Code ist vermutlich teurer Ballast.

### 9. Budget-Skalierung

Erst nachdem die Suchpfade verbessert wurden:

```text
200k
500k
1M
```

für n=32,34,36.

Dann siehst du:

```text
mehr Budget hilft stark → Rechenlimit
mehr Budget hilft kaum → Suchdynamiklimit
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
0. Baseline einfrieren ✓ (Commit af00bf0, data/baseline.json)
1. Random-Scan-Start ✗ (kein Gewinn, reverted)
2. Kick immer akzeptieren
3. Teil-Restart (eine Sequenz)
4. Tabu-Walk (stärkster Kandidat gegen Plateau)
5. Top-8 Subset-Rescue
6. K/K3-Tuning
7. rescue_mode-Ablation
8. Gewinner kombinieren
9. Größeres Budget
10. Spektralreparatur (langfristig)
N. AutocorrTracker (NAF-Reduktion, siehe HYPOTHESIS.md) — 3-5x schneller
    für GS4, verifiziert korrekt gegen GramTracker.  Test-Sweep läuft.
```

Immer nur **eine Änderung pro Testlauf**. Kein "SuperSolver" bei dem niemand
weiß, ob es am Tabu, am Kick oder am zufällig günstigeren Seed lag. Die
Wissenschaft nennt das dann gern "komplexes Zusammenspiel", weil "wir haben
den Überblick verloren" weniger elegant klingt.

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
| `run.py` | --sweep mit ProcessPoolExecutor (fertig) |
| `src/output.py` | save_run Auto-Solutions +_write_entry (fertig) |
| `data/baseline.json` | Baseline-Ergebnisse Commit af00bf0 (fertig) |
