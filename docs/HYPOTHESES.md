# Offene Hypothesen

Stand: 2026-08-07

Diese Datei enthält ausschließlich unbewiesene oder empirisch noch nicht
ausreichend abgesicherte Aussagen. Bestätigte Solvermechanik steht in
[`SOLVER_MODEL.md`](SOLVER_MODEL.md), das bewiesene Hauptresultat in
[`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md) und
konkrete Arbeitsschritte in [`TODO.md`](TODO.md).

## H1: Pair-Rescue kann Tabu verschlechtern

**Vermutung:** Monotone Pair-Schritte verbessern zwar sofort die Energie,
können den Solver aber in eine schwer lösbare Ein-Fehler-Falle führen. Tabu
hätte aus dem vorherigen Zwei- oder Drei-Fehler-Zustand möglicherweise eine
bessere Chance auf einen direkten Lösungsweg.

**Bisherige Evidenz:** Kleine gepaarte Stichproben bei `n=38` und `n=40` waren
ohne Pair-Rescue schneller und lösten häufiger. In den detaillierten
`n=38`-Traces beendete Pair-Rescue keine Lösung, während Tabu alle Lösungen
abschloss.

**Offen:** Die Stichproben reichen nicht für einen Defaultwechsel. Benötigt
wird eine große Ablation mit identischen Seeds und gepaartem McNemar-Test.

## H2: Der letzte Engpass braucht einen gezielten Mehrbit-Move

**Vermutung:** Die verbleibenden `Q=1/2`-Barrieren lassen sich effizienter mit
einem kleinen, zustandsabhängigen Mehrbit-Lookahead überwinden als mit mehr
globalem Greedy-Budget.

**Bisherige Evidenz:** Bei `n=52` erreichten alle 150 untersuchten Runs
`Q<=2`, aber nur zwei wurden gelöst. Reale Endzustände waren für sämtliche
Singles und Pairs isoliert. Erfolgreiche Runs sprangen innerhalb von Tabu
direkt von `Q=2` auf `Q=0`.

**Zu testen:** Gray-Code, kleiner Beam und 3- bis 8-Bit-Subsets ausschließlich
auf gespeicherten `Q<=2`-Zuständen. Eine globale Triple- oder Subset-Suche ist
nicht Teil dieser Hypothese.

## H3: Das Flip-Frame besitzt einfache Integrabilitätsbedingungen

**Vermutung:** Die Bedingung, dass ein ternäres Tight Frame tatsächlich aus
vier binären GS4-Folgen entsteht, lässt sich durch lokale Vorzeichen-,
Paritäts-, Verschiebungs- oder Zyklusbedingungen charakterisieren.

**Warum das wichtig wäre:** Eine einfache Parametrisierung könnte `D` statt
des Residualvektors als primäres Konstruktionsobjekt nutzbar machen. Dann wäre
die Tight-Frame-Charakterisierung nicht nur eine alternative Beschreibung,
sondern möglicherweise der Anfang einer neuen Konstruktion.

**Zu untersuchen:** Rekonstruktion der Produkte `x[c]x[c+t]`, Kompatibilität
verschiedener Lags, Zerlegung der Zeilen in vier Folgen, Negashift-Symmetrien
und die antipodale Darstellung im `2n`-Lift.

## H4: Strukturtreue zyklische Vorschläge können neue Escapes liefern

**Vermutung:** Der exakte antiperiodische `2n`-Lift kann Vorschläge erzeugen,
die im ursprünglichen Suchraum schwer sichtbar sind, sofern jeder Move die
Antipodalbedingung und das korrekte Zweiniveauspektrum erhält.

**Abgrenzung:** Eine freie gewöhnliche zyklische Flat-Spectrum-Optimierung ist
bereits als falsches Ziel erkannt. Interessant sind nur strukturtreue Moves,
kleine zyklische Quotienten oder Proposal-Generatoren, deren Ergebnis weiterhin
mit dem exakten Tracker bewertet wird.

## Zwei parallele Forschungsstränge

### Strang A: Neuheit

Systematische Literaturrecherche nach Kombinationen aus Hadamard/GS4,
negaperiodischer Autokorrelation, Tight Frames, Flip- und Ableitungsmatrizen,
diskreten Jacobians, Inzidenzmatrizen und verwandten Gram-Identitäten.

Zu prüfen sind insbesondere die elementweise Grammatrixformel, die
gerade/ungerade Frobenius-Normidentität und bekannte Charakterisierungen über
relative Difference Families.

### Strang B: Konstruktion

`D` als primäres Objekt behandeln und die Integrabilitätsbedingungen aus H3
herleiten. Beide Stränge kontrollieren einander: Eine konstruktive Struktur
kann passende Literaturbegriffe liefern, während bekannte Strukturen die
möglichen Integrabilitätsbedingungen einschränken.

Bis diese Arbeiten abgeschlossen sind, lautet der Status bewusst:

> Ernstzunehmendes neues mathematisches Resultat für das Projekt;
> wissenschaftliche Neuheit und konstruktiver Vorteil noch ungeklärt.

## Derzeit nicht gestützt

- Eine einfache Startmetrik trennt spätere Lösungen zuverlässig von
  Fehlläufen.
- Bestimmte Lags oder Frequenzen werden immer in derselben Reihenfolge
  repariert.
- Rang oder Konditionierung des Flip-Operators liefern nahe am Ziel ein
  zusätzliches Escape-Signal.
- Niedrige individuelle NAF-Rauheit erklärt den entscheidenden letzten Erfolg.
- Eine einzelne kleine spektrale Partition charakterisiert die Lösungen.
