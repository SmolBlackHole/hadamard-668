# Offene Hypothesen

Stand: 2026-08-07

Hier stehen nur unbestätigte, testbare Aussagen. Bewiesene Mathematik steht in
[`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md),
Messresultate in [`research/2026-08-07-search-ablations.md`](research/2026-08-07-search-ablations.md)
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

## Wissenschaftlicher Status

Die Tight-Frame- und Integrabilitätsformulierung ist ein ernstzunehmendes
Projektresultat. Wissenschaftliche Neuheit, eine neue Existenzklasse und ein
konstruktiver Vorteil sind weiterhin ungeklärt.
