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

- [x] **10n-Symbolnachbarschaft:** heutige Singles plus sechs
  basis-erhaltende Zwei-Bit-Moves je Spalte exakt vektorisiert. Als rein
  greedy Phase bei n=44/52 kein Solve-Gewinn; standardmäßig ausgeschaltet.
- [x] **Basis-erhaltender Walk:** `gs4-symbol` bewertet alle 10n Symbolmoves
  gemeinsam und erlaubt in einem kleinen separaten Tabu-Walk Verschlechterungen.
- [x] **Zwei-Kanal-Scorer:** `gs4-symbol-channel` verfolgt alle drei Paarungen
  exakt und normiert Q durch die stärkste Kanalenergie. Positives Signal bei
  n=28/30, neutral bei n=32; kein Default.
- [ ] **Zwei-Kanal-Holdout:** getrennte Gewichte oder größere n erst testen,
  wenn ein vorab festgelegter Holdout und Wall-Time-Budget verwendet werden.
- [ ] **Zwei-Ebenen-Ablation:** basis-erhaltende und basiswechselnde Moves
  getrennt testen; `beta` nicht voreilig als neue Zielfunktion behandeln.
- [ ] **Achsenreiner n=52-Test:** die reduzierte negaperiodische Symbolfamilie
  gegen freien GS4-Start vergleichen; klassische T-Bedingungen separat halten.
- [ ] **Tabu-Faserablation:** Best-Rollback gegen terminale Fortsetzung und
  kleines Archiv gleicher/naher Q-Zustände testen.
- [ ] **Scan-Symmetrie:** Greedy-Reihenfolge beziehungsweise unabhängige
  Negashifts an Plateaus randomisieren und auf reinen Indexbias prüfen.
- [ ] **Paar-Residual-Codebook:** Nach dem negativen exakten Archivabgleich die
  Komplementdichte `p_n(-w)` beziehungsweise eine billige Schätzung davon
  bestimmen. Minimales `||w||` ist als allgemeines Ziel widerlegt.
- [x] **Zwei-Kanal-Konstruktion:** kleine gemischte Lösungen der stärkeren
  Bedingungen `A=0` und `C=0` enumeriert und unabhängig auditiert. Exakt
  äquivalent zu zwei durch das Tripleprodukt-Dual gekoppelten GS4-Aufgaben.
- [ ] **Strong-Split-Warmstarts:** fast-starke und freie Zustände bei gleichem
  GS4-Start-Q downstream vergleichen; `Q_split` nicht als unabhängigen
  Basin-Indikator behandeln.
- [ ] **Targeted-Budget:** Support-Lag-Portfolio gegen deaktivierten Target-Escape
  unter gleichem Candidate-Evaluationsbudget testen.
- [ ] **Low-Q-Eintritt messen:** vergleichen, welche Starts die ternäre
  `Q<=8`-Zone früh und häufig erreichen; erst dort Escape auslösen.
- [ ] **Strukturdiverses BS-Portfolio:** viele Fast-BS-Zustände kurz in GS4
  quenchen und nach Downstream-Reaktion statt minimalem BS-Q auswählen.
- [ ] **Halb-Längen-Solver:** Symbol-/Phasenmoves im exakten chirp-modulierten
  QPSK-Raum prototypisieren.
- [x] **Q1-Barriere kartieren:** vollständige n=52-Komponenten bis Q=9 sowie
  beliebige Endpunkte bis Hamming-Radius drei geprüft; Passhöhe mindestens 10.
- [x] **Prospektiver Zwei-Schritt-Grad:** als exakten lokalen Topologiesensor
  verifiziert und unter gleichem Evaluationsbudget getestet; noch kein
  Lösungsrichtungs-Signal.
- [ ] **Minimax-Flooding:** für ausgewählte n=52-Q1-Zustände die tatsächliche
  minimale Passhöhe schrittweise bestimmen, statt eine feste Shell zu raten.

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
