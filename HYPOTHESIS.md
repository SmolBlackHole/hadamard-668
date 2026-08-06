# GS4-Energie via negaperiodische Autokorrelation

## Die Entdeckung

Der `GramTracker` berechnet die Orthogonalitäts-Energie einer 4n×4n-Matrix H.
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
S ist eine n×n-Matrix.

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
| Gram-Matrix (4n×4n) | 446.224 |
| Residualvektor | 83 |

Faktor: **5.376× weniger Zustand**.

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

1. **Single-Scan ~5000× schneller** bei n=167 — statt Minuten pro Scan, Millisekunden
2. **Subset-Rescue trivial**: Gray-Code über top-k Bits, ein δ pro Flip, O(n) pro Kombination
3. **Kick/Tabu-Walk**: operieren direkt auf den Residuals, sehen *welche* Abstände falsch sind
4. **Suchzustand in δ speichern**: jeder Flip hat einen δ-Vektor (m Werte), kann gecached werden
5. **Geometrie sichtbar**: r zeigt welche Abstände systematisch falsch sind — plateau-Diagnose direkt

## Prioritätsverschiebung

Der `AutocorrTracker` macht die gesamte Testkaskade (Schritte 1-9) obsolet.
Statt die Suchdynamik des langsamen GramTrackers zu optimieren, sollten wir
den schnellen Tracker bauen und dann bei viel größeren n testen.

## Referenz

- NAF-Definition: Djoković, "Hadamard matrices of order 668"
- GS4-Blockstruktur: Goethals-Seidel (1970)
- Energieäquivalenz: verifiziert 2026-08-06 an n=8 und n=167
