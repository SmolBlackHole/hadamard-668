# Integrabilität und Symbolmoves als Navigationsoperatoren

Stand: 2026-08-09

## Ergebnis in einem Satz

Die GF(2)-Kantenrekonstruktion löst eine **eingefrorene inverse
Dictionary-Zeilenaufgabe** exakt, aber nicht die selbstkonsistente GS4-Aufgabe;
selbst nach Durchsuchen tausender affiner Lösungen war sie ab `n=24` kein
brauchbarer Solve-Aktor. `10n`/`15n` vergrößern dagegen den Move-Graphen real
und verkürzen passende, spaltenweise geclusterte Defekte, liefern unter
gleichem Auswertungsbudget auf normalen Greedy-Minima aber noch keinen robusten
Vorteil gegenüber `4n`.

Reproduzierbarer Prototyp:

```text
python -m experiments.learning.integrability_navigation_audit \
  --ns 20 24 28 32 --roots 32 --replicas 2 --moves 16 --sample-size 8
```

Die wesentlichen Rohdaten liegen in:

- `data/integrability-navigation-audit.json`
- `data/integrability-navigation-optimized.json`
- `data/integrability-navigation-reverse.json`

Jeder berichtete Endzustand wurde mit einem frischen `Tracker.build()` geprüft.

## 1. Was `frame_integrability.py` wirklich macht

Das Skript führt keine alternierende Projektion und keine Optimierung auf dem
Gram-Raum durch. Es beweist beziehungsweise enumeriert blockweise folgende
Parametrisierung. Für

```text
p_j=(-1)^e_j,                  product_j p_j=-1
P_t(c)=product_{r=0}^{t-1} p_(c+r)
```

gilt für eine Folge exakt

```text
B[c,t]=-(P_t(c)+P_t(c-t))/2.
```

Ein odd-parity Kantenwort erzeugt eine Folge bis auf globales Vorzeichen und
damit einen integrablen Delta-Block. Umgekehrt bestimmt die erste Delta-Spalte
einen integrablen Block bis auf die dokumentierte Kollision bei `n=2 mod 4`.
Das ist eine exakte Parametrisierung der **zulässigen Faktorisierungen** von
`D`, keine Projektion auf Tightness.

Da `D^T D=2nI` im Projekt bereits äquivalent zu `u=0` ist, wäre eine naive
alternierende Projektion

```text
beliebiges D -> tightes Gram -> integrabler ternärer Block
```

ohne eine neue, wohldefinierte Integrabilitätsprojektion nur `Q` in einer
anderen Darstellung. Die Tight-Frame-Projektion verlässt im Allgemeinen das
ternäre Alphabet; die Rückprojektion auf ein Kantenwort ist global,
nichtkonvex und nicht durch `frame_integrability.py` implementiert.

## 2. Exakte GF(2)-Form des inversen Zeilenproblems

Fixiere einen Mittelpunkt `c` und das alte Gesamtresiduum `u`. Gesucht ist ein
Kantenwort, dessen Zeile `d_c=-u` erfüllt. Schreibe die beiden Fensterparitäten
als

```text
alpha_t = sum_{r=0}^{t-1} e_(c+r)       mod 2
beta_t  = sum_{r=0}^{t-1} e_(c-t+r)     mod 2.
```

Aus

```text
d[c,t]=-( (-1)^alpha_t + (-1)^beta_t )/2 = -u_t
```

folgen exakt die linearen Bedingungen:

| altes `u_t` | GF(2)-Bedingung |
| ---: | --- |
| `+1` | `alpha_t=0`, `beta_t=0` |
| `-1` | `alpha_t=1`, `beta_t=1` |
| `0` | `alpha_t XOR beta_t=1` |
| `abs(u_t)>1` | unmöglich für eine einzelne Delta-Zeile |

Dazu kommt `XOR_j e_j=1`. Damit ist das eingefrorene inverse Problem reine
Gauß-Elimination über GF(2). Die Implementierung ist für gerade und ungerade
`n` getestet; bei geradem `n` wird der tote Mittelpunkt-Lag nicht künstlich
hinzugefügt.

## 3. Das Selbstkonsistenz-Hindernis

Der entscheidende Punkt ist, dass die rekonstruierte Folge selbst das
Gesamtresiduum verändert. Sei `rho_s` der alte Einzelkanal der ersetzten Folge
und `rho_y` der Einzelkanal der rekonstruierten Folge vor dem Mittelpunktflip.
Dann ist nach dem Ersetzen

```text
u' = u + (rho_y-rho_s)/4.
```

Die konstruierte Zeile erfüllt nur `d_y,c=-u`, nicht `d_y,c=-u'`. Nach dem
zusätzlichen, tatsächlich vorgesehenen Mittelpunktflip bleibt daher exakt

```text
u'' = u' + d_y,c = (rho_y-rho_s)/4

Q'' = ||rho_y-rho_s||^2 / 16.
```

Diese Identität wurde gegen vollständige Tracker-Rebuilds getestet. Sie zeigt
präzise, was der inverse Vorschlag noch lösen müsste:

```text
d_y,c=-u            (lineares GF(2)-Problem)
rho_y ungefähr rho_s (nichtlineares NAF-/Phase-Retrieval-Problem).
```

Ohne die zweite Bedingung ist die Konstruktion zirkulär: Sie erzeugt lokal die
gewünschte Zeile, verschiebt dabei aber das Ziel, gegen das diese Zeile
gerichtet war.

## 4. Experiment: affine Integrabilitätsfasern

Nach einem reinen Single-Greedy-Quench waren die inversen Systeme fast immer
lösbar. Ihre Nullität lag typischerweise bei:

| n | beobachtete Nullität |
| ---: | --- |
| 20 | 7–9 |
| 24 | 8–11 |
| 28 | 9–12 |
| 32 | 10–14 |
| 52, kleiner Zusatzcheck | 11–17 |

Es existieren also viele integrable Folgen mit der gewünschten eingefrorenen
Zeile. Der stärkere Test sampelte beziehungsweise enumerierte pro Root im
Mittel `6.447`, `18.828`, `27.053` und `32.768` solcher Folgen und minimierte
innerhalb dieser affinen Faser exakt `||rho_y-rho_s||^2`.

Vergleich der besten direkten `Q`-Werte mit gleich vielen zufälligen
Folgenproposals:

| n | Roots | GF(2)-gezielt | Zufall |
| ---: | ---: | ---: | ---: |
| 20 | 27 | 3,07 | 1,22 |
| 24 | 31 | 4,00 | 1,81 |
| 28 | 31 | 5,55 | 3,00 |
| 32 | 32 | 7,22 | 5,22 |

Kein gezielter Proposal löste direkt. Nach Quench der jeweils vier besten
Proposals ergaben sich gelöste Roots:

| n | GF(2)-gezielt | Zufall |
| ---: | ---: | ---: |
| 20 | 6/27 | 4/27 |
| 24 | 0/31 | 1/31 |
| 28 | 0/31 | 2/31 |
| 32 | 0/32 | 0/32 |

Der kleine `n=20`-Basin-Effekt ist nicht monoton und kehrt sich danach um.
Der aktuelle Befund ist daher negativ: Die lineare Integrabilitätsfaser ist
kein skalierender inverser Dictionary-Aktor. Ein neuer Versuch wäre nur dann
begründet, wenn die zweite Bedingung `rho_y≈rho_s` strukturell statt durch
Enumeration gelöst wird.

## 5. `4n`, `10n` und `15n` als verschiedene Move-Graphen

`src/symbol_solver.py` nutzt exakt folgende spaltenlokale Nachbarschaften:

```text
4n  = vier Singles
10n = 4n + sechs Hamming-2-Moves derselben Spalte
15n = alle nichtleeren Teilmengen der vier Bits einer Spalte
```

Damit sind `10n` und `15n` keine neuen Scores, sondern echte Makro-Kanten im
Hypercube. Der Audit setzte denselben gesampelten Horizon-2-Score ein und
verglich sowohl gleiche 16 Entscheidungen als auch das Auswertungsbudget der
`4n`-Policy.

Über 244 gepaarte Runs für `n=20,24,28,32`:

| Policy | tiefere Basins | Solves | mittleres End-Q | mittlere Evals |
| --- | ---: | ---: | ---: | ---: |
| `4n`, 16 Moves | 145 | 8 | 2,15 | 14.605 |
| `10n`, 16 Moves | 169 | 19 | 1,90 | 37.329 |
| `15n`, 16 Moves | 152 | 26 | 2,00 | 55.965 |
| `4n`, gleiches Budget | 157 | 15 | 2,07 | 14.598 |
| `10n`, gleiches Budget | 157 | 17 | 2,07 | 14.099 |
| `15n`, gleiches Budget | 155 | 14 | 2,12 | 14.112 |

Bei gleicher Walk-Länge kaufen die Makromoves zusätzliche Hazard mit 2,6- bis
3,8-mal so vielen Auswertungen. Bei gleichem Budget verschwindet der Vorteil
praktisch vollständig. Das ist kein Produktionsargument für einen permanenten
`15n`-Scan, lässt aber einen **bedingten Makro-Aktor** offen.

## 6. Reverse-Audit: wann Makromoves tatsächlich neue Hazard liefern

Auf zufälligen Depth-3/4-Rückwärtszuständen bekannter Lösungen senkten `10n`
und `15n` häufig das beste Zwei-Schritt-Q, erzeugten aber fast nie einen
direkten Zwei-Move-Solve. Bei `n=28` waren es `0/16`; bei `n=32` zufällig
`1/16` für beide größeren Familien.

Auf bewusst spaltengeclusterten Defekten ist die Lage exakt anders:

| Defektmuster, je n | `4n` Solve sichtbar | `10n` | `15n` |
| --- | ---: | ---: | ---: |
| Pair in einer Spalte + Single | 0 | ja | ja |
| Triple in einer Spalte + Single | 0 | nein | ja |

In je 16 gemischten Depth-3/4-Zuständen pro `n` sah `10n` in `8/16` und
`15n` in `16/16` einen exakten Zwei-Schritt-Solve; `4n` sah keinen. Ein
uniformes K8-Turnier wählte einen lösenden Plan aber nur in ungefähr 2–4 % der
Fälle, weil das größere Wörterbuch den strukturellen Move stark verdünnt.

Die richtige Folgerung ist deshalb nicht „immer 15n“, sondern:

> Spaltenlokale Defektkonzentration kann als Trigger dienen, gezielt den
> passenden 10n-/15n-Makroblock zu öffnen.

## 7. Einordnung des bisherigen Reverse-Curriculum-Befunds

Dass beim vorhandenen Depth-3-Audit die globale `lookahead_q`-Bestmenge in
`126/136` Fällen einen bekannten Reverse-First enthält, ist teilweise durch
das Curriculum begünstigt: Nach zwei korrekten Singles liegt der Zustand in
`C1`, was ein Horizon-2-Score direkt sehen kann. Nicht tautologisch ist, dass
dieser bekannte Schritt gegenüber allen anderen Schritten global
konkurrenzfähig bleibt.

Noch operativer ist die K8-Anreicherung: `12,82 %` gegenüber `1,89 %`
Random-First ist ein etwa 6,8-facher Gewinn trotz unvollständiger Sicht. Sie
belegt einen brauchbaren lokalen C1-Proxy, aber noch keinen allgemeinen
Committor für unbekannte Q1-Minima.

Der Audit von `1.327` Lösungen verschärft diese Trennung: Für freie Lösungen
bei `n=44..52` gab es unter `64.056` Single-Nachbarn keinen Q1-Nachbarn und
keinen stillen Flip. Die beobachtete Q1-Wand ist damit nicht einfach die
unmittelbare Rückwärtsgrenze dieser Lösungen. `C1` existiert, liegt dort aber
auf höheren Q-Schalen.

## 8. Konsequenz

Belastbar bleiben zwei unterschiedliche Werkzeuge:

1. **Horizon-2 auf Singles** ist der derzeit effizienteste allgemeine
   Navigationssensor.
2. **Symbol-Makromoves** sind ein echter Zusatzaktor, wenn ein billiger Sensor
   spaltenweise geclusterte Defekte erkennt.

Die GF(2)-Inverse ist dagegen vorerst Diagnose, keine Navigation. Ihr Wert ist
die exakte Zerlegung des fehlenden Problems:

```text
lokale Zeilen-Cancellation + Einzelkanal-Erhaltung.
```

Ein zukünftiges Lemma müsste genau diese beiden Bedingungen koppeln. Nur dann
würde aus Integrabilität eine direkte Projektion statt einer Umformulierung des
NAF-Phase-Retrievals.
