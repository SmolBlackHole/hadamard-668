# State of the Art: Hadamard 668

## Offene Ordnungen (Stand August 2026)

668, 716, 892, 1132 sind die einzigen ungeloesten Ordnungen unter 1208.
668 = 4 x 167 ist die **kleinste** davon.

## Aktive Forschungsprojekte

### 1. jdbrande/hadamard-668-gs-tt-search (aktiv, Juli 2026)

**Ansatz:** Zwei Routen parallel:

- **Turyn-Typ TT(56):** 4 Sequenzen (56,56,56,55), nicht-periodische Autokorrelation, 223 freie Bits. DIESE Route hat Ordnung 428 (4x107) geknackt.
- **Goethals-Seidel GS(167):** 4 symmetrische Sequenzen a 167, periodische Autokorrelation, 336 freie Bits. Unser Raum.

**Methode:**

- Exakte Integer-PAF/NPAF-Vektoren (kein kontinuierlicher Proxy)
- Byte-genaues Hash-Matching (FNV-1a, open addressing)
- Sum-Patterns als harte Partition (Diophantische Constraints filtern 99% der Kombinationen)
- C++ Hot-Path, 120 Workers auf M3 Max / Server
- Monate Laufzeit, noch kein Fund

**Honest contract:**

```python
core.verify_hadamard(csv_path)  # exakt H*H^T == 668*I
# CSV wird NUR geschrieben wenn exakt verifiziert
```

**Relevanz fuer uns:**

- Die GS-Route ist identisch zu unserem Suchraum
- Die TT-Route haben wir NICHT implementiert — ist wahrscheinlich der vielversprechendere Raum (hat 428 geloest)
- Unser Gradienten-Ansatz ist fundamental anders als deren Hash-Matching — beides kann koexistieren

### 2. Ulam.ai / Frontier Research (2024-2026)

**Ansatz:** Legendre-Paare der Laenge 333.

- Kompressions-basierte Dekomposition
- mod-3 und mod-37 Obstruktionen
- Makro-Fall-Enumeration

**Status:** Kein Fund, aber strukturelle Hindernisse identifiziert.

### 3. 64-modular Hadamard 668 (2025)

Naechste Approximation: eine 64-modulare Hadamard-Matrix existiert.
Vorher: 32-modular (2001). Keine echte Hadamard-Matrix, aber naeher als je zuvor.

## Unser Energy-Metric vs. Literatur

| Aspekt | Literatur | Wir |
| --- | --- | --- |
| Metrik | 83-dim PAF-Vektor | Skalar `sum(autocorr[d]^2)` |
| Entscheidung | Exakter Match oder nicht | Gradientenabstieg minimiert Energie |
| Vergleichbarkeit | Determinisch, reproduzierbar | Inkompatibel mit Papers |
| Vorteil | Exakt, keine falsch-positiven | Differenzierbar, GPU-faehig |
| Nachteil | Kein Gradient, braucht Brute-Force-Matching | Kann nicht "fast richtig" von "richtig" unterscheiden |

**Empfehlung fuer unser Benchmarking:**

- Energie als Optimierungsmetrik behalten (funktioniert fuer Gradienten)
- ZUSAETZLICH reporten: max_corr, orth_pairs, Korrelations-Histogramm
- Das sind strukturelle Metriken die auch in Papers auftauchen
- Bei energy=0 sofort `check_orthogonality()` aufrufen — das ist der einzige Beweis

## Neue Konstruktionen die wir NICHT haben

### Turyn-Typ (Prioritaet)

- 4 Sequenzen mit nicht-periodischer Autokorrelation
- 223 Bits (vs. 336 bei symmetrischem GS)
- Hat Ordnung 428 (4x107) geknackt — DAS ist der vielversprechendste ungetestete Raum

### Legendre-Paare

- Laenge 333, aequivalent zu Hadamard 668
- Komplett anderer mathematischer Rahmen (Kombinatorik statt Signalverarbeitung)

## Schlussfolgerung

1. **Unsere Energie-Metrik ist legitim** als Optimierungs-Proxy, aber nicht als Benchmark zwischen Frameworks
2. **Niemand sonst nutzt FFT-Gradienten** fuer Hadamard-Suche — das ist unser Unique Selling Point
3. **Turyn-Typ ist die groesste Luecke** in unserem Strategie-Portfolio
4. **jdbrande's Framework ist unser naechster Konkurrent** — gleicher Raum (GS), anderer Algorithmus (Hash-Matching vs. Gradient)
