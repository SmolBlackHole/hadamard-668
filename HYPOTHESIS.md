# GS4-Energie via negaperiodischer Autokorrelation

## Was der Solver löst

Gesucht sind vier Folgen \(a,b,c,d\in\{\pm1\}^n\), deren negazyklische
Matrizen in der Goethals-Seidel-Konstruktion eine Hadamard-Matrix der Ordnung
\(4n\) bilden. Das ist äquivalent zu einem **negaperiodischen komplementären
Sequenzsatz**:

\[
  \operatorname{NAF}_a(t)+\operatorname{NAF}_b(t)+
  \operatorname{NAF}_c(t)+\operatorname{NAF}_d(t)=0,
  \qquad t=1,\ldots,n-1.
\]

Mit \(R\) als Spaltenumkehr und negazyklischen Blöcken \(A,B,C,D\) gilt für
die verwendete GS4-Struktur

\[
HH^\mathsf T=\operatorname{diag}(S,S,S,S),\qquad
S=AA^\mathsf T+BB^\mathsf T+CC^\mathsf T+DD^\mathsf T.
\]

Die Off-Diagonale von \(S\) besteht aus den Residuen
\(r_t=\sum_x\operatorname{NAF}_x(t)\). In der hier verwendeten Konvention
ist \(S_{i,j}=r_{|i-j|}\), mit \(r_{n-t}=-r_t\).

## Zielfunktion und Normierung

Die Repository-Metrik zählt ungeordnete Off-Diagonalpaare der vollständigen
Gram-Matrix:

\[
 E_{\mathrm{repo}}
 =\tfrac12\lVert HH^\mathsf T-4nI\rVert_F^2
 =2n\sum_{t=1}^{n-1}r_t^2
 =4n\sum_{t=1}^{\lfloor n/2\rfloor}r_t^2.
\]

Die volle Frobenius-Norm ist genau \(2E_{\mathrm{repo}}\). Beide Konventionen
sind zulässig, dürfen aber nicht vermischt werden.

Ein Bitflip verändert jedes Residuum um \(-4,0\) oder \(+4\). Deshalb sind
alle \(r_t\) durch vier teilbar. Der Tracker speichert

\[
 u_t=r_t/4,\quad Q=\sum_tu_t^2,\quad E_{\mathrm{repo}}=64nQ.
\]

Das halbiert die unabhängige Lag-Dimension und erlaubt einen `int8`-Delta-Cache.

## Inkrementelle Bewertung

Sei \(d\) das reduzierte Delta eines Flips. Dann gilt exakt

\[
 Q' = Q+2u^\mathsf Td+d^\mathsf Td.
\]

Der Tracker stellt sowohl einzelne Scores als auch einen vektorisierten
Vollscan bereit. Die lokale Suche nutzt bewusst den einzelnen Score mit
Early-Exit: Bei der ersten Verbesserung wird der Scan abgebrochen. Nach einem
akzeptierten Flip wird die betroffene Cache-Scheibe in \(O(n)\) mittels der
paarweisen quadratischen Korrektur aktualisiert; ein vollständiger
\(O(n^2)\)-Neuaufbau ist nicht erforderlich.

Für eine Menge von Flips genügt die Summe ihrer Single-Deltas plus für jedes
Paar derselben Folge genau eine Korrektur. Es gibt keine zusätzlichen
Triple-Terme. Das ist wichtig für ungerade \(n\): Der Lag \(\lfloor n/2\rfloor\)
ist dort kein Sonderfall und darf nicht übersprungen werden.

## Spektrale Formulierung

Für \(\zeta_k=\exp(i\pi(2k+1)/n)\), also die Nullstellen von \(z^n+1\),
ist die exakte Bedingung

\[
 \sum_{x\in\{a,b,c,d\}}|X(\zeta_k)|^2=4n\quad\text{für alle }k.
\]

Setzt man \(p_k=\sum_x|X(\zeta_k)|^2-4n\), dann gilt

\[
 E_{\mathrm{repo}}=2\sum_{k=0}^{n-1}p_k^2.
\]

Die Spektraldarstellung reduziert die Anzahl unabhängiger Freiheitsgrade nicht;
sie ist jedoch die richtige Basis für spätere ganze-Folgen-Moves oder
Phasen-Reparatur. Diese Strategien sind nicht Teil der aktuellen Suche.

## Einordnung

Der Kern gehört zugleich zu diesen Domänen:

- kombinatorische Konstruktion von Hadamard-Matrizen und komplementären Folgen;
- Signalverarbeitung/Radar: flache Leistung auf den negaperiodischen Frequenzen;
- quartische \(\{\pm1\}\)-Optimierung bzw. 4-Spin-/HUBO-Problem;
- lokale Suche auf einem diskreten Vektor-Balancing-Problem.

Die aktuelle Implementierung ist ein exakter lokaler Optimierer für diese
Zielfunktion, nicht ein allgemeiner Hadamard-Löser für beliebige Matrizen.
