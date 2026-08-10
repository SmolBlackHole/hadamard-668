# Offene Hypothesen

Stand: 2026-08-09

Hier stehen nur unbestätigte, testbare Aussagen. Bewiesene Mathematik steht in
[`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md),
Messresultate in [`SEARCH_FINDINGS.md`](SEARCH_FINDINGS.md)
und konkrete Arbeitsschritte in [`TODO.md`](TODO.md).

## H1: Reachability ist nicht durch Q allein bestimmt

`Q` misst den aktuellen GS4-Fehler exakt, aber ein niedrigeres `Q` führt nicht
zuverlässig in ein besser lösbares Einzugsgebiet. Evidenz dafür ist die
signifikant schlechtere Online-Gray-Solve-Rate trotz tausender exakt
verbessernder Moves.

**Vermutung:** Ein strukturierter Move darf zunächst `Q` erhöhen, wenn der
anschließende greedy Single-Abstieg ein besseres lokales Minimum erreicht.

**Test:** Auf identischen lokalen Minima zufälligen Kick, finalen
Tabu-Zustand und kontrolliert schlechtere Gray-Proposals jeweils vollständig
mit Singles quenchen. Erst `Q` und Solve-Erfolg des neuen Minimums bewerten.

## H2: Der gezielte Wörterbuchwechsel ist wichtiger als aktuelle Tightness

Die Grammatrix `D^T D` enthält keine unabhängige Information neben dem
aktuellen Residuum. Nach jedem Flip ändert sich jedoch der zustandsabhängige
Operator `D`.

**Vermutung:** Ein erster, möglicherweise schlechterer Flip lässt sich danach
auswählen, welche guten Richtungen im neuen Wörterbuch entstehen. Ein
einfacher Prüfausdruck ist

```text
R_1(i) = min_j Q(x xor i xor j).
```

Ein positiver Befund wäre ein billigerer und gezielterer Ersatz für den
zufälligen Kick.

## H3: Kantenkoordinaten erlauben stärkere Constraint-Propagation

Jeder Flip-Block wird exakt durch ein odd-parity Kantenwort parametrisiert. Die
Koordinate spart allein nur vier globale Vorzeichenbits.

**Vermutung:** Die Balance aller zyklischen Kantenintervalle lässt sich mit
SAT/CP-SAT, Faktorgraphen, Transfermatrizen, mehreren Intervallen oder kleinen
zyklischen Quotienten stärker propagieren als im direkten Folgenraum.

Ein einzelner Intervallflip und ein bloß auf `r_1=0` konditionierter Start sind
bereits negativ getestet. Ein neuer Ansatz muss mehrere Skalen gleichzeitig
nutzen.

## H4: Ein strukturtreuer Lift eröffnet andere Werkzeuge

Der exakte Lift `x -> (x,-x)` überführt NAF in periodische Autokorrelation auf
den ungeraden `2n`-Frequenzbins.

**Vermutung:** Ein Optimierer, der die Antipodalbedingung erhält oder nur
strukturtreue Proposals erzeugt, kann zyklische, spektrale oder kontinuierliche
Werkzeuge nutzen, ohne das GS4-Ziel zu verlassen.

Eine freie zyklische Flat-Spectrum-Optimierung ist ausdrücklich nicht das
richtige Problem.

## H5: Basiswahl und Symbolorientierung getrennt optimieren

Die Hadamard-Spaltenkoordinate zerlegt jedes Folgenquartett exakt in eine
Basiswahl, einen Achsenindex und ein Vorzeichen. Der aktuelle Single-Flip
ändert Basiswahl und Orientierung gekoppelt.

Das Basislabel allein ist kein hinreichender Indikator: Bis `n=6` tritt jedes
normalisierte Labelwort in einer exakten Lösung auf, während simples Switching
größerer Paley-Lösungen nur konstante Label liefert.

**Vermutung:** Eine kategorielle Suche über alle 15 Ersatzsymbole einer Spalte
oder ein alternierender Optimierer für Basislabel und Orientierung besitzt eine
bessere lokale Geometrie als vier unabhängige Bitflips.

**Test:** Auf gepaarten Seeds drei exakt gescorte Nachbarschaften vergleichen:

1. nur basis-erhaltende Zwei-/Vier-Bit-Symbolmoves;
2. nur basiswechselnde Ein-/Drei-Bit-Symbolmoves;
3. alle `15n` Ersatzsymbole.

Zusätzlich ist ein achsenreiner `n=52`-Lauf sinnvoll, weil dort eine direkte
Paley-Lösung existiert. Ein Erfolg würde einen besseren Suchraum belegen, aber
keine neue Konstruktion oder Hadamardklasse.

## H6: Tabu schneidet informative neutrale Fasern ab

Zustände mit gleichem `Q` und sogar gleichem `D^T D` können völlig
verschiedene unmittelbar lösende Move-Zeilen besitzen. Der aktuelle Tabu-Walk
übernimmt nach außen jedoch nur einen strikt besseren Bestzustand. Sein
terminaler Zustand und gleich gute, anders faktorisierte Zustände gehen
verloren.

**Vermutung:** Weiterlaufen vom terminalen Walk-Zustand oder ein kleines Archiv
strukturell verschiedener Zustände im gleichen Q-Fenster verbessert die
Erreichbarkeit, ohne eine neue Zielfunktion zu benötigen.

**Test:** Striktes Best-Rollback, terminale Fortsetzung und ein kleines
Q-Faserarchiv mit identischen Kandidatbudgets gepaart vergleichen. Als
Symmetriekontrolle an Plateaus unabhängige Negashifts/Reversals randomisieren;
ein reiner Gewinn daraus würde den festen First-Improvement-Scan als
algorithmischen Bias identifizieren.

## H7: GS4 als Kollision zweier Paar-Residualräume

Schreibe für jede Folge ihren NAF-Vektor als `rho_s` und setze

```text
w = rho_1 + rho_2.
```

Dann ist die GS4-Bedingung exakt

```text
rho_3 + rho_4 = -w.
```

Paley-/Golay-Konstruktionen verwenden den Spezialfall `w=0`; freie Lösungen
besitzen nach den bisherigen Paarungen typischerweise `w!=0`.

Ein simples festes Template ist bereits ausgeschlossen: Unter allen drei
Paarungen der 351 gespeicherten freien GS4-Lösungen wiederholte sich kein
`w`-Vektor exakt, auch nicht bis auf globales Vorzeichen.

**Vermutung:** Ein Paar-Codebook oder alternierender Zwei-Paar-Solver findet
Residualkollisionen zuverlässiger als vier unabhängige Folgenbits. Die
temporäre Variable `w` macht dabei die Kopplung explizit.

**Test:** Zwei unabhängig erzeugte Paarbänke auf exakte beziehungsweise nahe
Gegenvektoren hashen und dimensionsweise Symmetrien statt fester Templates
prüfen. Erst die beobachtete Kollisionsrate entscheidet, ob CPU-, GPU- oder
Meet-in-the-Middle-Suche sinnvoll ist.

## H8: Der starke Dual-Schnitt ist ein brauchbarer Seedraum

Die Existenz und exakte Algebra des starken Splits sind inzwischen bestätigt:

```text
F(a,b,c,d)=(bcd,acd,abd,abc),
F²=id,
A=C=0 <=> x und F(x) sind beide GS4,
Q_split=2(Q(x)+Q(F(x))).
```

Der Raum enthält viele kleine gemischte Lösungen und Paley, aber keine der
archivierten freien großen Lösungen. Ein einfacher Split-Tabu-Pilot stößt ab
`n=20..24` erneut an eine diskrete Restwand.

**Vermutung:** Trotz der stärkeren Bedingungen erzeugt ein Portfolio
strukturell verschiedener fast-starker Zustände bessere Starts für den freien
GS4-Solver als zufällige Zustände mit vergleichbarem Q.

**Test:** Bei gleichem GS4-Start-Q und gleichem Candidate-Budget starke,
fast-starke und freie Zustände downstream quenchen. `Q_split` selbst ist dabei
kein unabhängiger Basin-Indikator.

## H9: Aktionsbezogene Features approximieren die Cancellation-Koordinate

Für die Cancellation-Grenze `C1={x: es existiert d=-u}` gilt exakt

```text
Phi(x,Lösung)=Phi(x,C1).
```

Damit ist `Gamma(x)=(H_C,L_C)` aus minimaler Höhe und kürzester lateraler
Strecke bis `C1` eine exakte lösungsgerichtete Koordinate. Vollständige
Enumeration bei `n=5` zeigt, dass weder `Q`, `u` noch `D^T D` sie bestimmen.
Die zunächst bei kleinen n beobachtete Konstanz innerhalb der vier getrennten
Einzel-NAF-Fasern war ein Symmetrieartefakt. Bei `n=12` existieren Zustände mit
identischen vier `rho_s`, gleichem `u` und gleichem `u(Fx)`, aber verschiedener
Cancellation-Koordinate. Auch auf 896 Multi-n-Roots verbesserten statische
Kanal- und Dualfeatures die gemessenen Basinlabels nicht robust.

**Vermutung:** Die relevante Zusatzinformation steckt in der Wirkung einer
konkreten Aktion auf das zustandsabhängige Wörterbuch. Ein kleiner
Horizon-Lookahead oder eine Schätzung von `Delta Gamma(x,a)` ist informativer
als statische Kanalnormen.

**Test:** Kandidaten auf Reverse-Pfaden und unabhängigen Production-Minima mit
dem reinen Zwei-Schritt-Lookahead sowie einer aktionsbezogenen
Cancellation-Hazard vergleichen. Nach Ursprungslösung beziehungsweise Root
falten und Candidate-Kosten vollständig zählen. Statische Kanalfeatures gelten
für diese Rolle vorläufig als negatives Ergebnis.

## Wissenschaftlicher Status

Die Tight-Frame- und Integrabilitätsformulierung ist ein ernstzunehmendes
Projektresultat. Wissenschaftliche Neuheit, eine neue Existenzklasse und ein
konstruktiver Vorteil sind weiterhin ungeklärt.
