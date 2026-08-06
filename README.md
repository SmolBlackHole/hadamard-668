# Hadamard-668

Suche nach einer reellen Hadamard-Matrix der Ordnung \(668=4\cdot167\) über
die Goethals-Seidel-Konstruktion mit vier negazyklischen Folgen.

## Ausführen

```sh
python run.py --strategy gs4 --order 668
```

## Kernstruktur

```txt
src/
  builder.py     # GS4-Blockmatrix aus vier Folgen
  tracker.py     # reduzierte NAF-Energie und inkrementelle Deltas
  solver.py      # Single-Scan, Rescue, Kick und Restart
  generator.py   # Suchpipeline und finale Orthogonalitätsprüfung
  metrics.py     # Gram-Metriken
  output.py      # Ergebnis-Persistenz
scripts/
  ablation.py    # Komponentenablelationen
tests/
```

`Tracker` optimiert exakt dieselbe ungeordnete Gram-Energie, die `metrics.py`
für die finale Matrix berechnet. Die mathematische Reduktion steht in
[HYPOTHESIS.md](HYPOTHESIS.md), die experimentellen Grenzen in
[RESEARCH.md](RESEARCH.md).
