# Spiegelpunkt-Lemma und der allgemeine Satz d = -u

Stand: 2026-08-08 (erweitert um den allgemeinen Satz, algebraisch geprueft).
Fundort: `experiments/learning/` (Analyse der n=52- und n=44-Runs, eigene
Replays; Solver in `src/` unverändert).

Dieses Dokument beschreibt die exakte Charakterisierung, wie ein Zustand
durch EINEN Einzel-Flip zu einer Lösung (Q=0) wird. Der Q=1-Spezialfall
(Spiegelpunkt mit Exception) ist ein Korollar des allgemeinen Satzes `d = -u`.
Empirisch validiert auf 53+ echten Lösungen (n=44, `preflip_n44.jsonl`):
Der letzte lösende Übergang JEDER Lösung ist ein `d=-u`-Flip mit
Exception-Menge `supp(u_pre)` (siehe unten und `d_eq_minus_u_experiment.md`).

## Setup

Vier Folgen `x^(s) in {+1,-1}^n`, antiperiodische Fortsetzung
`x_bar[c+n] = -x_bar[c]`. Reduzierte NAF-Residualien

```text
u_t = r_t / 4,  r_t = sum_s NAF_{x^(s)}(t),   t = 1..m,  m = floor((n-1)/2),
Q = ||u||_2^2.
```

Ein Einzel-Flip (Bitwechsel an Position `c` der Folge `s`) hat den exakten
Delta-Vektor (verifiziert identisch zur Tracker-Delta-Cache, `int8`):

```text
d[s,c](t) = -(1/2) x_bar[c] (x_bar[c+t] + x_bar[c-t])
          = -(1/2) x_c (x_{c+t} + x_{c-t})     (Indizes mod n, Rand wie oben)
d[s,c](t) in {-1, 0, +1}.
```

Die Q-Änderung eines Flips ist exakt `Q' = Q + 2 <u,d> + ||d||^2`.

## Allgemeiner Satz: Einzel-Flip loest genau dann, wenn d = -u

**Satz.** Für einen beliebigen Zustand `x` (nicht nur Q=1) gilt:

```text
Ein Einzel-Flip an Position (s, c) loest (Q' = 0)
<=>  d[s,c] = -u .
```

**Beweis.** `u' = u + d` (Tracker-Update ist exakt), also
`Q' = ||u'||^2 = 0 <=> u + d = 0 <=> d = -u`. QED. Die Beweisidee des alten
Q=1-Lemmas ("jeder nichttriviale Flip hat ||d||^2 >= 1") wird nicht gebraucht;
Null-Deltas sind möglich, das Argument ist rein algebraisch.

**Konsequenzen** (da `d_t in {-1,0,1}`):

1. Ein direkt lösbarer Zustand muss ternäre Residuen haben:
   `u_t in {-1, 0, +1}` fuer alle t.
2. Komponentenweise uebersetzt die Deltaformel `d[s,c](t) = -(1/2) x_c
   (x_{c+t} + x_{c-t})` die Bedingung `d = -u` in lokale Spiegelstruktur:

   ```text
   u_t =  0  =>  x_{c-t} = -x_{c+t}        (Antisymmetrie-Paar)
   u_t = +1  =>  x_{c-t} = x_{c+t} =  x_c  (positive Exception)
   u_t = -1  =>  x_{c-t} = x_{c+t} = -x_c  (negative Exception)
   ```

   Die Exception-Menge ist **`S = supp(u)`** — nicht nur ein einzelner Lag.
   Wir nennen den Zustand einen **Spiegelpunkt mit Exception-Menge S**
   (auch: Exact-Cancellation-Zustand, Zustand auf der Cancellation Manifold
   `C = {x : existiert (s,c), d_{s,c}(x) = -u(x)}`).

## Lemma (Q=1-Spezialfall: Spiegelpunkt mit Exception k)

Sei `Q = 1`, also `u = sigma e_k`. Dann gilt (Korollar des allgemeinen Satzes):

```text
Ein Einzel-Flip an Position c der Folge s loest (Q' = 0)
<=>  x_{c-t} = -x_{c+t}  fuer alle t != k        (Antisymmetrie)
     und
     x_{c-k} = x_{c+k} = sigma * x_c              (Exception bei Lag k).
```

**Korrektur (gegenueber der ersten Fassung):** fuer `sigma = -1` sind die
beiden Spiegelpunkte entgegengesetzt zu `x_c` (nicht gleich).  Aus
`d_k = -sigma` und `d_k = -(1/2) x_c (x_{c+k} + x_{c-k})` folgt
`x_{c+k} + x_{c-k} = 2 sigma x_c`, also beide gleich `sigma x_c`. QED.

## Korollar 1 (Seltenheit / Motiv-Charakter)

Fuer einen zufaelligen Zustand und festen Exception-Lag `k`: `m-1`
Antisymmetriebedingungen (je Wahrscheinlichkeit 1/2) plus zwei festgelegte
Nachbarn (Wahrscheinlichkeit 1/4), also

```text
P(Spiegelpunkt mit Exception k) = 2^{-(m+1)}
E[Anzahl] = 4n * 2^{-(m+1)}.
```

Fuer `n = 44` (`m = 21`): `176 * 2^{-22} ~= 4.2e-5` (Korrektur: die erste
Fassung gab `2^{-(n+2)/2} = 2^{-23}`, Faktor 2 daneben). Die Bedingung ist
ein **seltenes, hochstrukturiertes Motiv** — kein statistisches Artefakt,
sondern ein diskretes Objekt im negazyklischen Raum.

Empirisch (eigene Sammlung, n=44, v1-Pipeline): alle 10 untersuchten Q=1-
Sackgassen-Zustände (ungelöste Runs) haben **0 Spiegelpunkte** bei ihrem
Exception-Lag `k`. Gelöste Runs erreichen Zustände mit Spiegelpunkt (die
Gier-Suche nimmt den lösenden Flip dann sofort).

## Korollar 2 (Frequenzbereichs-Form)

Mit der Skew-DFT `A(zeta) = sum_j x_j zeta^j`, `zeta^n = -1`, gilt

```text
d[s,c] = -u   <=>   Delta(zeta_k) = -U(zeta_k)  fuer alle Bins k
```

wobei `Delta` die Skew-DFT des Delta-Vektors und `U` die des Residuals ist.
Bei `Q = 1` ist `U(zeta_k) = sigma zeta_k^{-k}` eine reine Phase: Ein
lösender Flip ist ein Flip, dessen Delta-Spektrum eine reine Phase mit
entgegengesetztem Vorzeichen ist — er "kancelt alle Frequenzen gleichmäßig".
Das verbindet das diskrete Motiv mit der Tight-Frame-Sicht: Der Frame-Defekt
`|R(zeta_k)| = 4 |U(zeta_k)| = 4` ist bei Q=1 in allen Bins konstant, und
genau ein Flip mit konstantem Gegen-Spektrum schließt das Frame.

## Korollar 3 (Screening / C-Mitgliedschaft)

Die Zählfunktion

```text
in_C(x) = #{(s, c) : d[s, c] = -u}
```

ist in `O(4 n m)` berechenbar (µs) und entscheidet die Mitgliedschaft auf der
Cancellation Manifold `C`. Empirisch (n=44: 350 Escape-Zustände, n=52: 435):
**0 Zustände in C** — kein Escape-Zustand hat eine Einzelflip-Kanzelation.
Sogar `dist1 = 0` ueberall: auch EIN Flip von einem Escape-Zustand erreicht C
nie (C ist mindestens 2 Flips entfernt; `q10_results.json`).

## Beobachtungen aus den Runs (ergänzt um die Pre-Flip-Messung)

1. Bei n=52 erreichen die Quenches von **keinem** der 494 ungelösten Runs je
   Q<=1; die 6 gelösten Runs erreichten Q=0 (Sprung Q<=8->0 im Tabu-Walk).
2. Bei n=44 erreichen **alle** Runs Q=1 (u = +-e_k), aber nur ~24-58%
   (je nach Pipeline) schließen zu 0. Die Q=1-Zustände ohne Spiegelpunkt sind
   Sackgassen für die Gier-Suche; der Tabu-Walk muss zu einem
   C-Zustand navigieren.
3. **Das Motiv ist transient.** Alle statisch beobachtbaren Zustände sind
   motivfrei: 0/350 Escape-Zustände (n=44), 0/435 (n=52) haben einen
   Kanzelations-Flip; auch die Iterationsgrenzen-Zustände des SIEGER-Quenchs
   (30/30) sind motivfrei. Der lösende Flip existiert nur in den internen
   Zuständen des Tabu-Walks.
4. **Pre-Flip-Messung (53+ echte Lösungen, n=44, `preflip_n44.jsonl`):**
   Der Zustand UNMITTELBAR vor dem lösenden Flip erfüllt in 53/53 (jetzt
   >80) Fällen exakt `d = -u` mit ternärem `u_pre`. Die Exception-Mengen
   haben Größe `Q_pre in {4,...,9}` — **kein** Q_pre in {1,2,3} beobachtet.
   Die lösenden Übergänge sind also Q=4..9 → 0, nie Q=1 → 0. Der letzte
   Schritt ist ein Multi-Exception-C-Flip, kein Q=1-Spiegelpunkt.
5. Alle 53 lösenden Flips kamen aus dem Tabu-Walk (0 Greedy, 0 Kick). Der
   Walk startet bei Q=2..13 und erreicht C bei Q_pre=4..9 oft ÜBER seinem
   bisherigen Q-Minimum (Aufwärts-Exkursion in Q nötig, um C zu treffen).
6. Variante G (10x längerer Tabu-Walk bei Q<=1 im Quench) ist NICHT besser
   als die Baseline (4/12 vs 7/12 auf denselben Seeds, n=44): Das Erreichen
   des C-Zustands skaliert nicht mit mehr Tabu-Schritten (die Trajektorie
   ist RNG-empfindlich, nicht budget-limitiert).
7. Die Sieger-Kombos des v1-Escapes (Ränge 103-606/625, Q=44-83) sind in
   keinem lokalen Screening-Merkmal erkennbar; der C-Befund erklärt, warum:
   Der Erfolg hängt an der Erreichbarkeit von C in der Quench-Kette, nicht
   an lokalen Q-Merkmalen. Auch die C-Distanz des Escape-Zustands
   (min_1flip_q) korreliert NICHT mit dem Erfolg (corr = +0.067).

## Screening-Implikation (verfeinert, Messung durchgeführt)

Da C statisch unsichtbar ist, ist die nutzbare Screening-Größe die
**C-Distanz**: die minimale Anzahl Flips von einem beobachtbaren Zustand zu
einem C-Zustand. Messung (`q10_results.json`): dist0 = dist1 = 0 ueberall;
min_1flip_q (bester Einzel-Flip) liegt bei n=44 bei 5.7 (gelöst) vs 5.6
(ungelöst) — **ohne Vorhersagekraft**. Die lokale C-Nähe des Escape-Zustands
sagt nichts über den Erfolg voraus; die Outcome-Zufälligkeit ist
Trajektorien-Dominanz (erreicht der Walk C oder nicht), konsistent mit dem
ML-Befund (AUC ~0.6).

## Offene Fragen / nächste Schritte

- Kann man die C-Erreichbarkeit eines Escape-Zustands vorhersagen
  (die eigentliche Screening-Frage)?
- **Konstruktion:** Welche Flip-Folgen erzeugen aus einem Escape-Zustand
  einen C-Zustand? (Die gespeicherten Pre-Zustände + Q-Trajektorien in
  `preflip_n44.jsonl` sind die Datenbasis; eine direkte Transformation
  würde den Tabu-Walk für den letzten Abschnitt ersetzen.)
- Wie selten ist C bei gegebenem Q? (C-Dichte in der 1-Flip-Umgebung von
  Zufalls-Zuständen mit Q=4..9.)
- Gibt es eine konstruktive Erzeugung von C-Zuständen (Muster-Generierung
  im Sinne von PZinn)?
