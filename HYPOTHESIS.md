# GS4-Energie via negaperiodische Autokorrelation

## Die Entdeckung

Der `GramTracker` berechnet die Orthogonalitäts-Energie einer 4nx4n-Matrix H.
Durch die Goethals-Seidel-Blockstruktur reduziert sich das Problem drastisch.

## Mathematische Reduktion

Die GS4-Matrix hat Blockstruktur:

```txt
     ┌               ┐
     │ A   BR  CR  DR│
H  = │-BR  A  -DᵀR CᵀR│
     │-CR  DᵀR  A  -BᵀR│
     │-DR -CᵀR BᵀR  A │
     └               ┘
```

wobei R die Spaltenumkehr ist und A,B,C,D negazyklisch sind:

```txt
M[i,j] = a[(j-i) mod n] · sign,  sign = -1 wenn j < i
```

### Schritt 1: Block-Diagonalisierung

```txt
        ┌           ┐
        │S  0  0  0 │
HHᵀ  =  │0  S  0  0 │   mit  S = AAᵀ + BBᵀ + CCᵀ + DDᵀ
        │0  0  S  0 │
        │0  0  0  S │
        └           ┘
```

Die Kreuzterme heben sich wegen der GS4-Struktur exakt weg.
S ist eine nxn-Matrix.

### Schritt 2: Negazyklische Autokorrelation

Für eine negazyklische Matrix A aus Folge a gilt:

```txt
(AAᵀ)[i,j] = NAF_a((j-i) mod n)
```

die NAF (Negaperiodic Autocorrelation) von a:

```txt
NAF_a(t) = Σ_{j=0}^{n-t-1} a_j·a_{j+t}  -  Σ_{j=n-t}^{n-1} a_j·a_{j+t-n}
```

### Schritt 3: Residualvektor

Für die vier Folgen a,b,c,d definiert man:

```txt
r_t = NAF_a(t) + NAF_b(t) + NAF_c(t) + NAF_d(t),   t = 0,1,...,n-1
```

Wegen der Negaperiodizität gilt `r_{n-t} = -r_t`. Für gerades n ist `r_{n/2} = 0` automatisch.

### Schritt 4: Energieformel

```txt
E(H) = 4n · Σ_{t=1}^{n-1} r_t²
```

bzw. mit `m = ⌊(n-1)/2⌋` unabhängigen Werten:

```txt
E(H) = 8n · Σ_{t=1}^{m} r_t²
```

## Verifikation

Numerisch an n=167 mit 5 Zufallsstichproben gegen den `GramTracker` verifiziert:
alle Energien exakt identisch.

## Zustandsgröße

| Darstellung | Einträge bei n=167 |
| --- | --- |
| Gram-Matrix (4nx4n) | 446.224 |
| Residualvektor | 83 |

Faktor: **5.376x weniger Zustand**.

## Einzelflip-Delta

Beim Flip von aⱼ ändert sich für jeden Abstand t genau ein Korrelationsprodukt:

```txt
δ_t ∈ {-4, 0, +4}
```

Die Energieänderung:

```txt
ΔE = 8n · (2 · rᵀδ + |δ|²)
```

Das ist ein Skalarprodukt über m ≈ n/2 kleine Ganzzahlen — kein Matmul, kein float32.

## Laufzeitvergleich

| Operation | GramTracker | AutocorrTracker |
| --- | --- | --- |
| Single-Flip | O(n²) | O(n) |
| Voller Scan | O(n³) | O(n²) |
| Rescue (2-bit) | O(n²) via _combo_delta | O(n) via δ-Addition |
| Build (Start/Restart) | O(n²) Matmul | O(n²) NAF-Berechnung |

## Konsequenzen für KFlip

1. **Single-Scan ~5000x schneller** bei n=167 — statt Minuten pro Scan, Millisekunden
2. **Subset-Rescue trivial**: Gray-Code über top-k Bits, ein δ pro Flip, O(n) pro Kombination
3. **Kick/Tabu-Walk**: operieren direkt auf den Residuals, sehen *welche* Abstände falsch sind
4. **Suchzustand in δ speichern**: jeder Flip hat einen δ-Vektor (m Werte), kann gecached werden
5. **Geometrie sichtbar**: r zeigt welche Abstände systematisch falsch sind — plateau-Diagnose direkt

## Diagnostik via Residualvektor

Der NAF-Tracker macht die Suchlandschaft sichtbar. Statt nur "Energie = 2048"
sieht man jetzt *welche Abstände falsch sind*:

```txt
Abstand 1:   0
Abstand 2:  -4
Abstand 3:   0
Abstand 4:  12   ← hier liegt der Fehler
Abstand 5:  -8   ← und hier
```

### Plateau-Diagnose

Zwei Zustände können dieselbe Energie haben, aber völlig verschiedene Fehlerprofile:

```txt
Zustand A:  r = [12, 0, 0, 0]     → Q = 144
Zustand B:  r = [6, 6, 6, 6]      → Q = 144
```

Mit nur einer Gesamtzahl sieht der Solver keinen Unterschied. Der Residualvektor
ermöglicht zusätzliche Kriterien:

- **Anzahl nicht-null Residuen**: `nnz(r)` — weniger ist besser
- **Maximales Residuum**: `max(|r_t|)` — glättet Plateaus
- **L1-Summe**: `sum(|r_t|)` — weniger sensitiv für Ausreißer
- **Adaptive Gewichtung** (Breakout Local Search): `sum(w_t · r_t²)` wobei `w_t`
  für hartnäckige Abstände erhöht wird

Diese Hilfsziele haben alle den gleichen Nullpunkt (`r=0`), verändern aber die
Topologie der Suchlandschaft um den Solver aus lokalen Minima zu führen.

## Spektrale Perspektive

Das gleiche Problem lässt sich im Frequenzraum ausdrücken. Sei

```txt
ζ_k = exp(iπ(2k+1)/n)
```

eine Nullstelle von -1, und `A(ζ_k)` die entsprechende Fourier-Auswertung.
Die Hadamard-Bedingung lautet dann:

```txt
|A(ζ_k)|² + |B(ζ_k)|² + |C(ζ_k)|² + |D(ζ_k)|² = 4n   für alle k
```

Der Solver sucht vier binäre Signale, deren **summierte spektrale Leistung
vollkommen flach** ist. Das ist die Signalverarbeitungs-Interpretation des
Problems und relevant für GPU/FFT-basierte Ganzfolgen-Reparatur.

## Der Solver in verschiedenen Domänen

Der KFlip-Solver operiert gleichzeitig in mehreren mathematischen Gebieten:

| Domäne | Formulierung |
| --- | --- |
| **Kombinatorik** | GS-Differenzfamilien, Hadamard-Konstruktion |
| **Signalverarbeitung** | Vier Signale, summierte spektrale Leistung flach |
| **Optimierung** | HUBO (Polynom 4. Grades): `Q(x) = Σ_t (xᵀP_t x)²` |
| **Statistische Physik** | Ising-Modell mit Vierkörper-Wechselwirkungen |
| **Frame-Theorie** | Überschüssige Frame-Potential-Energie erzwingt Orthogonalbasis |

## Prioritätsverschiebung

Der NAF-Tracker macht die Geschwindigkeits-Aspekte der Testkaskade (Schritte 1-9)
obsolet. Statt die Suchdynamik des langsamen GramTrackers zu optimieren, können
wir den schnellen Tracker nutzen und die Suchstrategien direkt auf dem
Residualvektor entwickeln.

Der größte Gewinn ist nicht nur Geschwindigkeit. Wir suchen jetzt **in den
natürlichen Koordinaten des Problems** (Autokorrelations-Residuen) statt
ständig die vollständige Hadamard-Matrix als teuren Übersetzer dazwischenzuschalten.

## Referenz

- NAF-Definition: Djoković, "Hadamard matrices of order 668"
- GS4-Blockstruktur: Goethals-Seidel (1970)
- Energieäquivalenz: verifiziert 2026-08-06 an n=8 und n=167
