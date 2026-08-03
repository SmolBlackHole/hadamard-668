# Signal Processing Perspective on Hadamard-668

## Die zentrale Erkenntnis

Unser Problem IST ein Filterdesign-Problem. Die Signalverarbeitungs-Community
hat exakt diese Problemklasse seit den 1980ern geloest -- nur nicht fuer
binaere (+-1) Sequenzen.

## Was wir eigentlich machen

Wir suchen 4 Sequenzen a,b,c,d der Laenge 167, deren AUTOKORRELATIONS-SUMME
an allen lags (ausser 0) gleich null ist:

    autocorr(a) + autocorr(b) + autocorr(c) + autocorr(d) = 4k * delta[0]

Nach Wiener-Khinchin ist Autokorrelation = IFFT(|FFT|^2). Im Frequenzraum
wird das zu:

    |FFT(a)[f]|^2 + |FFT(b)[f]|^2 + |FFT(c)[f]|^2 + |FFT(d)[f]|^2 = 4k

fuer JEDE der 167 Frequenzen f. Das sind 167 unabhaengige Constraints.

## Das ist Power Complementarity

In der Sprache der Signalverarbeitung: a,b,c,d bilden eine
**4-Kanal power-complementary filter bank**. An jeder Frequenz muss
die Gesamtleistung exakt 4k betragen.

Das ist fundamental bekannt aus:

- **QMF (Quadrature Mirror Filters)** -- 2-Kanal power-complementary
- **Paraunitaren Filterbanken** -- mehrkanalig
- **Wavelet-Theorie** -- orthogonale Wavelets erfuellen Power-Complementarity

## Was Signalverarbeiter anders machen wuerden

### 1. Spektrale Faktorisierung statt lokaler Suche

Statt Eintraege zu flippen: entwerfe das Power-Spektrum, dann faktorisiere.

Schritt 1: Finde 4 Spektren S_a[f], S_b[f], S_c[f], S_d[f] >= 0
           mit S_a+S_b+S_c+S_d = 4k an JEDER Frequenz.
Schritt 2: Faktorisiere jedes Spektrum in ein Minimum-Phase-Filter:
           H(z) so dass |H(e^{jw})|^2 = S(w).
           Das ist SPEKTRALE FAKTORISIERUNG -- ein Standardverfahren.
Schritt 3: Das Minimum-Phase-Filter hat KOEFFIZIENTEN. Die muessen
           NICHT +-1 sein. ABER: wir koennen das Residuum minimieren.

Das ist komplett anders als unser Ansatz -- nicht "finde die Sequenz
durch Flippen", sondern "designe das Spektrum und approximiere mit
binaeren Koeffizienten".

### 2. POCS (Projection onto Convex Sets)

Alternierende Projektionen:

- Projiziere auf +-1 Constraint: sign(x)
- Projiziere auf Power-Complementarity via FFT/IFFT

Iteriere bis Konvergenz. Das ist Douglas-Rachford aber auf den
SEQUENZEN (O(n log n)), nicht auf der vollen Matrix (O(n^3)).

### 3. Nyquist-Filter-Perspektive

Ein symmetrisches Filter dessen Autokorrelation exakt delta[0] ist
(ausser am lag 0), ist ein **Nyquist-Filter** (zero-ISI).

Unsere 4 Sequenzen muessen 4 Nyquist-Filter sein, deren Summe der
Power-Spektren konstant = 4k ist. Das heisst: die 4 Filter bilden
zusammen ein perfektes Rekonstruktionssystem.

Mit BINAEREN Koeffizienten. Das existiert fuer n=2,4,8 (Sylvester),
aber nicht fuer n=167. Die Frage ist: existiert es ueberhaupt?

### 4. FFT-basierte Gradientenberechnung

Statt Energie per Autokorrelation zu berechnen (O(n^2)) und dann
per Finite-Differenzen den Gradienten zu schaetzen (O(n) calls =
O(n^3)), koennen wir den EXAKTEN Gradienten per FFT berechnen:

dEnergy/d(seq[k]) = 2 *sum_f violation[f]* 2 *cos(2*pi*f*k/n)

Alle 336 Gradienten auf einmal: O(4 *n* log n) ≈ 4 *167* 8 = 5.3k ops.
vs. aktuell: 336 *O(n^2) = 336* 27889 = 9.4M ops fuer Finite-Differenzen.

Das ist ein 1000x Speedup fuer Gradienten-basierte Optimierung.
Statt 270 Schritte/s → 270.000 Gradienten-Schritte/s.

Der Code dafuer:

```python
def fft_gradient(seqs):
    """Gradient von sum |sum_i |FFT(s_i)|^2 - 4k|^2.
    Returns: list of 4 gradient vectors, each length k.
    """
    k = len(seqs[0])
    target = 4 * k
    # Compute power spectrum sum
    ps_sum = np.zeros(k)
    fft_seqs = []
    for s in seqs:
        f = np.fft.rfft(s.astype(np.float64))
        fft_seqs.append(f)
        ps_sum += np.abs(f) ** 2

    violation = ps_sum - target  # per-frequency violation
    grads = []
    for f in fft_seqs:
        # d/ds of sum (ps_sum - target)^2
        # = 2 * (ps_sum - target) * d(ps_sum)/ds
        # d(|FFT(s)|^2)/ds = IFFT of gradient in freq domain
        grad_freq = 2 * violation * f
        grad = np.fft.irfft(grad_freq, n=k)
        grads.append(np.round(grad).astype(np.float64))
    return grads
```

## Was das fuer Ising bedeutet

Der Ising-Fehler "tanh ignoriert Zeilen-Korrelationen" ist aus
Signalverarbeitungssicht ein **fehlender Whitening-Schritt**.

Bevor der tanh angewendet wird, muesste der Gradient ZEILENWEISE
dekorreliert werden. Das geht per:

```python
# Cholesky der Praezisionsmatrix
L = np.linalg.cholesky(gram / n + 1e-6 * np.eye(n))
grad_white = np.linalg.solve(L, grad)
state = tanh(beta * (state - grad_white))
```

Das wuerde die Korrelationen zwischen den Gradienten-Zeilen entfernen.
DANN funktioniert der elementweise tanh korrekt.

## Naechste Schritte

1. **FFT-Gradient** in CirculantSearch einbauen → 1000x schneller
2. **POCS-Sequenz-Projektion** als neue Strategie
3. **Spektrale Faktorisierung + Binaer-Approximation** testen
4. **Ising + Cholesky-Whitening** fuer den Durchbruch bei 16x16
