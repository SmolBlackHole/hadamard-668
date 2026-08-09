# TODO

Stand: 2026-08-09

Nur offene oder unmittelbar geplante Arbeit steht hier. Abgeschlossene und
verworfene Experimente sind in `SEARCH_FINDINGS.md` zusammengefasst.

## Direkte Konstruktionen

- [ ] **Paley-Primzahlpotenzen:** `paley-ng` von Primzahlen auf echte
  Primzahlpotenzen `q=2n-1` erweitern.
- [ ] **Paley Type II:** Für ungerades `n` und `q=2n-1≡1 mod 4` einen direkten
  Matrixpfad der Ordnung `4n` implementieren. Das ist nicht automatisch eine
  GS4-Sequenzparametrisierung.
- [ ] **Dispatcher-Abdeckung:** direkte Matrix- und Sequenzkonstruktionen vor
  jeder heuristischen Suche eindeutig ausweisen.

## Suchdynamik

- [ ] **15n-Symbolnachbarschaft:** alle Ersatzsymbole einer H4-transformierten
  Spalte vektorisieren und gegen Singles/Tabu gepaart testen.
- [ ] **Zwei-Ebenen-Ablation:** basis-erhaltende und basiswechselnde Moves
  getrennt testen; `beta` nicht voreilig als neue Zielfunktion behandeln.
- [ ] **Achsenreiner n=52-Test:** die reduzierte negaperiodische Symbolfamilie
  gegen freien GS4-Start vergleichen; klassische T-Bedingungen separat halten.
- [ ] **Tabu-Faserablation:** Best-Rollback gegen terminale Fortsetzung und
  kleines Archiv gleicher/naher Q-Zustände testen.
- [ ] **Scan-Symmetrie:** Greedy-Reihenfolge beziehungsweise unabhängige
  Negashifts an Plateaus randomisieren und auf reinen Indexbias prüfen.
- [ ] **Targeted-Portfolio:** `legacy625` und `support_lag625` mit getrennten,
  kandidatenstabilen RNG-Streams gepaart testen.
- [ ] **Low-Q-Eintritt messen:** vergleichen, welche Starts die ternäre
  `Q<=8`-Zone früh und häufig erreichen; erst dort Escape auslösen.
- [ ] **Strukturdiverses BS-Portfolio:** viele Fast-BS-Zustände kurz in GS4
  quenchen und nach Downstream-Reaktion statt minimalem BS-Q auswählen.
- [ ] **Halb-Längen-Solver:** Symbol-/Phasenmoves im exakten chirp-modulierten
  QPSK-Raum prototypisieren.

## H668-Konstruktionsräume

- [ ] **TT->BS->GS4 Relaxation:** constraint-erhaltende TT-/BS-Makromoves
  entwickeln; gewöhnliche Singles verlassen notwendige Mannigfaltigkeiten.
- [ ] **TT(56)-Hybrid:** erst nach besserer aperiodischer Suchdynamik erneut
  gegen direkte GS4(167)-Starts testen.
- [ ] **Base-Sequences:** Zerlegungen `BS(m,167-m)` nur mit neuer
  Move-Geometrie weiterverfolgen; Single-/Tabu ist bereits negativ pilotiert.

## Mathematik und Neuheit

- [ ] Die n=52-Diskrepanz aus erster Hand erklären: direkte dünne
  Konstruktionsfamilie gegen lokale Erreichbarkeit im freien GS4-Raum.
- [ ] Tight-Frame-/Integrabilitätscharakterisierung gegen Literatur zu
  diskreten Jacobians, Frames und complementary sequences prüfen.
- [ ] Erzeugbarkeitsbedingungen des Delta-Operators aus Kantenwörtern
  symbolisch herleiten.
- [ ] Wissenschaftliche Neuheit erst nach Literaturprüfung und vollständigem
  Beweis behaupten.

## Technik

- [ ] GPU-Screening erst mit einer belegten Ranking-Regel erneut einführen.
- [ ] Evaluationsbudgets pro Phase konsistent ausweisen.
- [ ] Benchmarks mit festen Seeds, Solve-Rate, CPU-Zeit pro Lösung und exaktem
  Rebuild verifizieren.
