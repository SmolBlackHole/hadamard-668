# Dokumentation

Die Dokumentation ist nach Aufgabe getrennt:

- [`SOLVER_MODEL.md`](SOLVER_MODEL.md) erklärt kompakt, was der aktuelle Solver
  berechnet und wie seine Suchphasen zusammenspielen.
- [`BENCHMARK.md`](BENCHMARK.md) enthält den reproduzierbaren Leistungsstand.
- [`HYPOTHESES.md`](HYPOTHESES.md) enthält ausschließlich offene, testbare
  Vermutungen.
- [`TODO.md`](TODO.md) ist die priorisierte, ausführbare Roadmap.
- [`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md) ist die
  kanonische, ausführliche Darstellung des neuen mathematischen Hauptbefunds.

Die Dateien unter [`research/`](research/) sind datierte Laborprotokolle. Sie
bewahren Herleitungen, Messaufbauten, negative Ergebnisse und Rohbefunde, sind
aber keine aktuellen Roadmaps:

- [`research/2026-08-07-algebraic-frame-analysis.md`](research/2026-08-07-algebraic-frame-analysis.md)
- [`research/2026-08-07-solver-trace-analysis.md`](research/2026-08-07-solver-trace-analysis.md)

## Kurz gesagt, ohne Mathematik

Der Solver prüft nicht blind fertige riesige Matrizen. Er baut vier kurze
Bitfolgen und misst nur die wenigen Fehler, die durch die gewählte Bauweise
noch übrig bleiben. Kleine Änderungen beheben viele dieser Fehler sehr schnell.

Nahe am Ziel reichen direkte Verbesserungen aber oft nicht mehr. Dann verändert
der Tabu-Walk den Kandidaten vorübergehend in eine schlechtere Richtung, damit
er eine andere Nachbarschaft erreicht. Genau dieser Mechanismus hat die frühere
Leistungsgrenze deutlich verschoben.

Die neue Charakterisierung erklärt, warum der Weg bis kurz vor die Lösung so
gut funktioniert: Je besser ein Kandidat wird, desto gleichmäßiger decken seine
möglichen Änderungen alle verbleibenden Fehler ab. Der letzte Schritt bleibt
dennoch ein diskretes Kombinationsproblem. Der aktuelle Engpass ist deshalb
nicht das Finden guter Kandidaten, sondern der gezielte Sprung aus sehr guten,
aber lokal abgeschlossenen Zuständen.
