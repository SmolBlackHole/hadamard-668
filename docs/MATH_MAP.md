# Mathematische Landkarte des Projekts

Stand: 2026-08-09

Diese Datei verbindet die mathematischen Darstellungen des Projekts. Sie ist
kein weiterer Ergebnisbericht. Ihr Zweck ist, sichtbar zu machen, welche
Größen exakt dieselbe Information tragen, welche zusätzliche Zustandsstruktur
enthalten und welche nur hinreichende Unterräume beschreiben.

Die daraus abgeleitete Suchsteuerung steht in
[`BASIN_NAVIGATION.md`](BASIN_NAVIGATION.md).

## Legende

| Zeichen | Bedeutung |
| --- | --- |
| `<=>` | exakte Äquivalenz im angegebenen Gültigkeitsbereich |
| `=` | exakte Identität beziehungsweise invertierbare Umrechnung |
| `=>` | nur hinreichend; die Rückrichtung gilt nicht allgemein |
| `subset` | echter oder empirisch belegter Unterraum |
| `~>` | Suchheuristik oder empirische Beziehung, kein Satz |

Skalierungen sind wichtig. `rho` und `r` bezeichnen rohe NAF-Residualgrößen;
der Produktions-Tracker speichert `u=r/4` und `Q=||u||²`.

## 1. Die Gesamtkarte

```text
vier Folgen x=(a,b,c,d) in {+-1}^{4n}
│
├─ vier Einzelresiduen R=(rho_a,rho_b,rho_c,rho_d)
│  │
│  ├─ Gesamtresiduum r=sum rho_s, u=r/4, Q=||u||²
│  │      GS4 <=> u=0 <=> Q=0
│  │
│  ├─ Paarseparator w=rho_a+rho_b, v=rho_c+rho_d
│  │      GS4 <=> v=-w
│  │
│  └─ H4-Kanäle (U,S1,S2,S3)=H4 R
│         U=r; zusammen exakt äquivalent zu allen vier rho_s
│
├─ H4-Spaltensymbole v_j=H4 x_j/4
│  │      Zeichen eta + Basis beta + Achse j
│  │
│  ├─ u=A+C
│  └─ A=0 und C=0 => GS4                [starker Split]
│
├─ Kantenwörter p_c=x_bar[c]x_bar[c+1]
│  │      exakt integrierbare Parametrisierung bis auf globales Vorzeichen
│  └─ Fensterprodukte P_t(c) erzeugen NAF und Flip-Blöcke
│
├─ zustandsabhängiges Flip-Wörterbuch D(x)
│  │      Zeile d_i = Änderung von u durch Flip i
│  ├─ Q_i'=Q+2<u,d_i>+||d_i||²
│  ├─ direkter Solve <=> d_i=-u
│  └─ GS4 <=> D^T D=2n I               [Tight Frame]
│
└─ spektrale Darstellungen
   ├─ anti-periodischer 2n-Lift: nur ungerade DFT-Bins
   └─ gerades n=2h: chirp-modulierte QPSK-Faltung der Länge h
```

Die zentrale dynamische Schleife ist:

```text
x  ──bestimmt──>  (u,D)
│                    │
│ Flip i             │ Score aus u und Zeile d_i
▼                    │
x' ──bestimmt neu──> (u',D')
```

Der Solver verändert mit jedem Move gleichzeitig den Fehler `u` und das
zukünftige Wörterbuch `D`. Genau deshalb ist niedriges `Q` keine vollständige
Beschreibung der Erreichbarkeit.

## 2. Primäres Objekt: NAF und Residuen

Für eine antiperiodisch fortgesetzte Folge gilt

```text
rho_x(t) = NAF_x(t) = sum_c x_bar[c] x_bar[c+t].
```

Für vier Folgen:

```text
r = rho_a+rho_b+rho_c+rho_d,
u = r/4,
Q = ||u||².
```

Damit sind folgende Aussagen exakt äquivalent:

```text
GS4-Hadamard
<=> r=0
<=> u=0
<=> Q=0.
```

`Q` zählt nicht allgemein falsche Bits. Nur bei ternärem `u` ist `Q` gleich
der Zahl seiner nichtverschwindenden Koordinaten.

## 3. Welche Größen nur Q anders darstellen

Diese Größen enthalten keine unabhängige Basin-Information:

- die volle summierte NAF-Energie;
- das aggregierte Spektrum des Gesamtresiduums;
- bei geradem `n` der Frobenius-Framefehler
  `||D^T D-2nI||_F²=4(n-4)Q`;
- Rang, Eigenwerte und Konditionierung von `D^T D`, weil die komplette Matrix
  durch `u` bestimmt ist;
- skalare Paar-Kanalenergien bei festem `Q`, soweit sie nur Normen der
  H4-transformierten Residualkanäle wiedergeben;
- der zustandsweite Mittelwert der Zeilennormen `||d_i||²`, der bei festem
  `n` exakt konstant ist.

Diese Größen können numerisch praktisch oder interpretierbar sein. Sie
liefern aber keinen neuen Zustandskompass.

## 4. Welche Größen zusätzliche Information tragen

### 4.1 Die konkrete Zeilenfaktorisierung von D

`D^T D` kennt die Summe aller äußeren Produkte der Zeilen, nicht deren
konkrete Anordnung. Zustände mit identischem `u` und identischem `D^T D`
können unterschiedliche Zahlen unmittelbar lösender Zeilen besitzen.

Zusätzliche Information steckt daher beispielsweise in:

- den einzelnen Zeilen `d_i`;
- ihrer Richtung relativ zu `u`;
- den nach einem Probe-Move entstehenden Zeilen `D(x flip i)`;
- sparsamen höheren Wechselwirkungen mehrerer konkreter Zeilen.

### 4.2 Volle Paar- und H4-Kanäle

Für eine Paarung:

```text
w = rho_a+rho_b,
v = rho_c+rho_d,
r = w+v.
```

Die Lösung verlangt `v=-w`. Der volle signierte Vektor `w` enthält mehr
Information als `r` allein. Seine Norm ist jedoch kein allgemeiner
Qualitätsindikator. Der relevante kombinatorische Wert wäre die Zahl
erzeugbarer Gegenvektoren:

```text
p_n(-w) = Anzahl Folgenpaare mit Residuum -w.
```

Über alle drei Paarungen entsprechen die vier vollen H4-Kanäle
`(U,S1,S2,S3)` exakt den vier Einzelresiduen. Die drei skalaren Kanalenergien
werfen dagegen Lag-, Richtungs- und Vorzeicheninformation weg.

### 4.3 Kantenwörter

Das Kantenwort

```text
p_c=x_bar[c]x_bar[c+1],  product_c p_c=-1
```

parametrisiert eine Folge bis auf ihr globales Vorzeichen. Seine
Fensterprodukte erzeugen sowohl die NAF als auch den kompletten Flip-Block:

```text
P_t(c)=product_{j=0}^{t-1} p_{c+j},
B(x)[c,t]=-(P_t(c)+P_t(c-t))/2.
```

Ein Bitflip ändert zwei benachbarte Kanten. Ein beliebig langer Intervallflip
ändert ebenfalls nur seine beiden Randkanten. Das verändert die natürliche
Metrik der Move-Suche, reduziert den Raum aber asymptotisch nicht.

## 5. Move-Landkarte

Eine Spalte besitzt vier Bits und damit 16 gültige Symbole. Vom aktuellen
Symbol existieren 15 Alternativen:

| Hamming-Distanz | Moves je Spalte | H4-Bedeutung |
| ---: | ---: | --- |
| 1 | 4 | Wechsel zwischen den beiden erlaubten Basen |
| 2 | 6 | Bewegung innerhalb der aktuellen Basis |
| 3 | 4 | Basiswechsel plus andere Orientierung |
| 4 | 1 | globales Spaltenvorzeichen |

Daraus folgen die untersuchten Nachbarschaften:

```text
4n  = nur Singles
6n  = nur basis-erhaltende Hamming-2-Moves
10n = 4n+6n
15n = alle anderen Spaltensymbole
```

Die drei Paarungen `01|23`, `02|13`, `03|12` sind keine zusätzlichen Achsen.
Sie sind drei lineare Zerlegungen derselben vier Einzelresiduen.

Für zwei Single-Flips liefert die diskrete Hessian-Struktur alle Endenergien
ohne Neubau der Kandidaten:

```text
Q_ij = Q + deltaQ_i + deltaQ_j + 2 <d_i,d_j>
```

für verschiedene Folgen. Bei derselben Folge kommt exakt eine sparsame
Ein-Lag-Korrektur hinzu. Diese paarweisen Korrekturen genügen, um das
**Residual** nach beliebig vielen Flips exakt zusammenzusetzen: Es ist in den
Flipindikatoren höchstens quadratisch. Die Energie

```text
Q(z)=||u(z)||²
```

ist dadurch jedoch im Allgemeinen quartisch. Konkrete n=7-Zustände besitzen
nichtverschwindende dritte und vierte Möbius-Differenzen von `Q`. Der
Multi-Flip-Code bleibt exakt, weil er zuerst das quadratische Residuum bildet
und erst anschließend dessen Norm quadriert; nur die frühere Interpretation
als rein paarweise Energie war falsch.

## 6. Erreichbarkeitskarte

Definiere die Cancellation-Menge

```text
C = {x : es existiert i mit d_i(x)=-u(x)}.
```

Ein Zustand ist genau dann einen Single-Flip von einer Lösung entfernt, wenn
er in `C` liegt. Für `Q=1`, also `u=sigma e_k`, gilt exakt:

- Verbesserung ist nur als direkter Solve `d=-u` möglich;
- ein neutraler Move ist `d=0` oder transportiert den Defekt mit
  `||d||²=2` auf genau einen anderen Lag;
- fehlen beide Movearten, ist der Q1-Zustand im Q1-Graph isoliert.

Auf den 109 untersuchten n=52-Q1-Zuständen gilt:

```text
keine Q0/Q1-Endpunkte in Hamming-Radius 1, 2 oder 3,
keine Lösung in den vollständig erschöpften Single-Sublevelräumen Q<=8/9.
```

Dies ist eine datensatzspezifische Passhöhenaussage, kein allgemeines
Kongruenztheorem. Kleine `n` besitzen neutrale Q1-Transportpfade.

Der prospektive Grad

```text
b_H(i)=#{j != i : Q(x flip i,j)<=H}
```

misst exakt die lokale Verzweigung nach dem ersten Move. Er ist ein echter
Topologiesensor, aber bisher kein Lösungsrichtungs-Sensor: adaptive K4/K8-
Scans erschlossen mehr enge H12-Komponenten, lieferten aber keinen Solve.

## 7. Konstruktionsräume und ihre Beziehungen

```text
freier GS4-Raum: u=0
│
├─ starker Split: A=0 und C=0
│  ├─ enthält direkte Paley/Ito-Lösungen
│  ├─ enthält viele kleine basisgemischte Lösungen
│  └─ ist echt kleiner als freier GS4
│
├─ Paarfaktorisierung: w und -w separat erzeugt
│  └─ w=0 ist der NG-/Golay-Spezialfall, bei ungeradem n unmöglich
│
├─ Base-/T-/TT-Unterräume
│  └─ TT -> Base Sequences -> GS4 ist exakt
│
└─ freie Solverlösungen
   └─ große archivierte Lösungen liegen in keiner bislang implementierten
      Paley-/Golay-/BS-/TT- oder Strong-Split-Schublade
```

### 7.1 Starker Split

In der Zwei-Basen-Koordinate gilt exakt

```text
u=A+C.
```

Daher ist `A=C=0` hinreichend, aber nicht notwendig. Der laufende unabhängige
Audit liefert eine zweite Beschreibung. Definiere

```text
beta_j=a_j b_j c_j d_j,
F(a,b,c,d)=(bcd, acd, abd, abc).
```

Dann gelten `F²=id` und `u(F(x))=A-C`. Folglich:

```text
A=C=0
<=> x ist GS4 und F(x) ist GS4.
```

Damit ist der starke Split der Schnitt zweier GS4-Bedingungen, verbunden durch
ein involutives punktweises Tripleprodukt-Dual. Zusätzlich gilt exakt

```text
Q_split = 2 (Q(x)+Q(F(x))).
```

`Q_split` ist damit keine dritte unabhängige Energie, sondern die gekoppelte
Energie zweier GS4-Aufgaben. Der unabhängige Audit ordnet GS4 in bekannter
Sprache als nega-Williamson ein. Die starke Familie ist darin ein unter `F`
dual-geschlossener Schnitt. Sie enthält Base/T und duplizierte NG-Paare, ist
aber nach den geprüften Gegenbedingungen nicht allgemein quaternion-QT,
Williamson-type oder eine bekannte Relative-Difference-Set-Familie. Die
wissenschaftliche Neuheit der konkreten `F`-Dualbedingung bleibt offen.

### 7.2 Paarseparator und Defect Gluing

Für `w=rho_a+rho_b` gilt koordinatenweise eine Paritätsbedingung:

```text
w_t = 2n mod 4.
```

Bei ungeradem `n` ist `w=0` unmöglich; bei geradem `n` sind alle Koordinaten
durch vier teilbar. Die exakte Zählzerlegung lautet

```text
#GS4 = sum_w p_n(w) p_n(-w).
```

Dies ist ein echter Separator für Meet-in-the-Middle oder Variable
Elimination. Die bisherige Minimalnorm-Idee ist negativ: Schon bei `n=16`
liegt die dominante Kollisionsmasse nicht bei minimalem `||w||`, und keine der
19 freien n=52-Lösungen liegt in der Minimaldefektschale.

### 7.3 Bekannte strukturtreue Lifts

- Paley/Ito erzeugt für passende `q=2n-1` direkt ein NG-Paar und damit GS4.
- Golay-Paare liefern eine direkte komplementäre Unterfamilie.
- Turyn-Multiplikation liftet geeignete Paare oder GS4-Tupel
  multiplikativ in größere Längen.
- TT und Base Sequences bilden strengere aperiodische Unterräume, die exakt in
  GS4 eingebettet werden können.
- Der zyklische `2n`-Lift und die QPSK-Halblängenfaltung sind exakte
  Darstellungen, aber allein keine Suchraumreduktionen.

## 8. Was aktuell als neuheitsverdächtig gilt

Noch nicht wissenschaftlich als neu belegt, aber projektspezifisch und nicht
als bloße Umbenennung widerlegt sind:

1. die GS4-spezifische Flip-Grammatrixidentität und Tight-Frame-Äquivalenz;
2. die exakte Integrabilität des Flip-Blocks aus einem odd-parity Kantenwort;
3. Cancellation-Menge, Q-Passhöhe und Wörterbuchnavigation als gemeinsames
   algorithmisches Framework;
4. die H4-Zweibasenform der gültigen binären Spaltensymbole;
5. der starke Split als dual-geschlossener Schnitt von `x` und seinem
   involutiven Tripleprodukt-Dual `F(x)`.

Bekannte oder klar verwandte Grundideen sind NAF, Paley/Ito, Golay, Turyn,
Base/TT, relative Difference Families, Pair-Hashing und allgemeine
Pseudo-Boolean-Zwei-Flip-Auswertung.

## 9. Konsequenz für weitere Experimente

Eine neue Metrik ist nur dann interessant, wenn sie Information enthält, die
nicht bereits Funktion von `u` oder `D^T D` ist. Vor einem Solverexperiment
sind deshalb drei Fragen zu beantworten:

1. **Informationsprüfung:** Ist die Größe aus `u` rekonstruierbar? Dann ist sie
   kein neuer Basin-Sensor.
2. **Aktionsprüfung:** Kann sie einen konkret anderen zulässigen Move wählen?
3. **Budgetprüfung:** Bleibt der Gewinn bei gleichem Candidate- oder
   Wall-Time-Budget bestehen?

Die derzeit saubersten noch offenen Richtungen sind:

- Komplementdichte `p_n(-w)` billig approximieren, statt `||w||` zu
  minimieren;
- Wörterbuchänderung nach wenigen Probe-Moves beurteilen, ohne alle Paare
  permanent zu scannen;
- die minimale Passhöhe einzelner n=52-Q1-Zustände mit adaptivem
  Minimax-Flooding bestimmen;
- den starken Split nach abgeschlossenem Algebra-Audit nur bei belegter
  Suchskalierung als Seedraum verwenden; seine Energie ist bereits als
  `2(Q(x)+Q(F(x)))` vollständig verstanden;
- terminale beziehungsweise gleichhohe Tabu-Zustände behalten, weil dieselbe
  Q-Faser unterschiedliche konkrete `D`-Faktorisierungen enthält.

## 10. Kanonische Quellen im Repository

- [`SOLVER_MODEL.md`](SOLVER_MODEL.md): Produktionsformeln und Kontrollfluss.
- [`TIGHT_FRAME_CHARACTERIZATION.md`](TIGHT_FRAME_CHARACTERIZATION.md):
  bewiesene Frame-, Kanten- und Spektralidentitäten.
- [`CONSTRUCTION_SPACE.md`](CONSTRUCTION_SPACE.md): Paley, Golay, Turyn,
  Base/TT und H4-Spaltenkoordinate.
- [`SEARCH_FINDINGS.md`](SEARCH_FINDINGS.md): reproduzierte Suchbefunde und
  negative Ablationen.
- [`HYPOTHESES.md`](HYPOTHESES.md): ausschließlich offene Vermutungen.
- [`TODO.md`](TODO.md): offene konkrete Arbeit.

Experimentberichte unter `experiments/` sind Evidenzquellen, aber keine
parallele kanonische Wahrheit. Belastbare Aussagen werden in die obigen
Dokumente übernommen.
