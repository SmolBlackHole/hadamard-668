# TODO

Stand: 2026-08-07

Diese Datei enthält ausführbare Aufgaben. Begründungen und offene Aussagen
stehen in [`HYPOTHESES.md`](HYPOTHESES.md).

## Jetzt

- [ ] **Pair-Rescue ablatieren**
  - mindestens 1000 identische Seeds bei `n=38` und `n=40`,
  - `pairs=True` gegen `pairs=False`,
  - McNemar-Test, Wilson-Intervalle, Zeit pro Lösung und Phasencounter,
  - danach nur bei Bedarf `Single -> Tabu -> Pair -> Kick` testen.

- [ ] **Toten Mittelpunkt-Lag für gerade n entfernen**
  - Residuum, Delta-Cache und Tabu-Kernel auf `floor((n-1)/2)` umstellen,
  - Gleichheit von Energie, Deltas und Solve-Ergebnis gegen den bisherigen
    Tracker testen,
  - Performance bei `n=38` und `n=52` messen.

## Danach

- [ ] **Low-Q-Escape-Datensatz festlegen**
  - gespeicherte `Q=1/2`-Zustände aus `n=38` und `n=52` verwenden,
  - exakte Erfolgsmetrik: Lösung pro CPU-Sekunde,
  - Baselines: bestehender Tabu-Walk und kein Spezialmove.

- [ ] **Gezielten Gray-Code-/Subset-Rescue prototypisieren**
  - nur bei `Q<=2`, nicht im globalen Solverpfad,
  - 3 bis 8 ausgewählte Bits beziehungsweise Richtungen,
  - exakte Same-Sequence-Korrekturen verwenden,
  - gegen kleinen Beam und Tabu ablatieren.

- [ ] **Pair-K nur bei bestätigtem Pair-Nutzen tunen**
  - falls Pair-Rescue nach der Ablation aktiv bleibt: `3√B`, `4√B`, `5√B`,
  - andernfalls K-Tuning streichen, weil der Parameter den Hauptpfad nicht mehr
    beeinflusst.

## Parallele mathematische Forschung

- [ ] **Strang A – Neuheit prüfen**
  - relevante Literatur und Terminologie systematisch erfassen,
  - elementweise Gram-Identität und beide Normidentitäten vergleichen,
  - Ergebnis mit genauer Abgrenzung dokumentieren.

- [ ] **Strang B – Konstruktion über D untersuchen**
  - notwendige Integrabilitätsbedingungen herleiten,
  - kleine `n` vollständig enumerieren und Kandidatenbedingungen testen,
  - nur exakt gegen den bestehenden Tracker validierte Reduktionen weiterführen.

## Später

- [ ] Strukturtreue Proposal-Moves aus dem antiperiodischen `2n`-Lift testen.
- [ ] Hierarchische Reparatur über kleine zyklische Quotienten untersuchen.
- [ ] Spektrale Ganzfolgenreparatur als isolierten Prototyp evaluieren.
- [ ] Budget-Scaling für größere `n` nach Verbesserung des Low-Q-Escapes
  wiederholen.
