# TODO

Stand: 2026-08-07

## Jetzt

- [ ] **Toten Mittelpunkt-Lag bei geradem n physisch entfernen**
  - Residuum, Delta-Cache und Tabu-Kernel auf `floor((n-1)/2)` begrenzen.
  - Energie, Deltas, feste Seeds und Solve-Rate gegen den jetzigen Tracker
    prüfen.
  - Erwarteter reiner Lag-Gewinn: etwa 5,3 % bei `n=38` und 3,8 % bei `n=52`.

- [ ] **Solvermetriken auf Entscheidungen ausrichten**
  - klären, welche Singles-/Tabu-/Kick-Zahlen eine konkrete Hypothese prüfen;
  - nicht verwendete Q-Buckets oder Zeitfelder entfernen;
  - keine allgemeine Trace-Infrastruktur erneut einführen.

## Nächster Suchversuch

- [ ] **Strukturierte Perturbation mit Single-Quench prototypisieren**
  - zunächst bei `n=38`, danach mit frischen pair-freien `n=52`-Zuständen;
  - aktueller Vier-Bit-Kick als gepaarte Baseline;
  - finalen Tabu-Zustand auch ohne unmittelbare Verbesserung übernehmen und
    mit Singles quenchen;
  - kontrolliert schlechtere Gray-Proposals ebenso testen;
  - Lösung, neues lokales Minimum, Barriere und CPU-Zeit messen.

- [ ] **Ein-Schritt-Wörterbuchwechsel messen**
  - für jeden zulässigen ersten Flip die beste Single-Richtung im neuen
    Wörterbuch bestimmen;
  - `min_j Q(x xor i xor j)` gegen unmittelbares `Q(x xor i)` vergleichen;
  - nur bei positivem Trace-freien Prototyp einen kleinen Beam erwägen.

## Mathematische Forschung

- [ ] Breitere Neuheitsprüfung über Literaturdatenbanken und Zitationsketten.
- [ ] Kantenwort-Balance als SAT-/CP-SAT- oder Faktorgraphmodell formulieren.
- [ ] Strukturtreue Proposals aus dem antiperiodischen `2n`-Lift testen.
- [ ] Gerade/ungerade Transformationen nur dann produktiv nutzen, wenn sie
  stärkere Propagation oder neue Move-Familien liefern.

## Erledigt oder verworfen

- [x] Pair-Rescue gepaart ablariert und vollständig entfernt.
- [x] K-Tuning gestrichen; es gehörte nur zum Pair-Pfad.
- [x] Statischen und Online-Gray-Code bis `n=52` geprüft; monotone Varianten
  verworfen.
- [x] `combo_energy()` unabhängig gegen direkte Neuberechnung geprüft.
- [x] Alle einzelnen Intervallflips auf `n=52`-Low-Q-Zuständen geprüft.
- [x] Kantenbalancierten `r_1=0`-Start geprüft und verworfen.
- [x] Blockintegrabilität des Flip-Operators charakterisiert und verifiziert.
- [x] Trace-Infrastruktur und abgeschlossene Wegwerfprototypen entfernt.

Details und Zahlen stehen in [`research/2026-08-07-search-ablations.md`](research/2026-08-07-search-ablations.md).
