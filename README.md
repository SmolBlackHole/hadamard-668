# Hadamard-668

Finde eine reelle Hadamard-Matrix der Ordnung 668.

## Hintergrund

Ordnung 668 = 4 × 167 ist die **kleinste ungelöste Ordnung** der Hadamard-Vermutung (seit 1867). Eine Hadamard-Matrix H in {-1,+1}^(668×668) mit H·H^T = 668·I zu finden, wäre ein bedeutendes mathematisches Resultat.

## Projektstruktur

```
├── src/
│   └── search.py             # Such-Engine (direkt + Williamson)
├── verifier/                 # Prüf-Toolchain
│   ├── verify.py             # Matrix-Prüfung
│   ├── review.py             # Bundle-Review
│   ├── score.py              # Fortschritt-Metriken
│   ├── validate.py           # JSON-Manifest-Validator
│   └── seed.py               # Deterministischer Seed-Generator
├── schema/
│   ├── run.schema.json       # JSON Schema fuer Suchlauf-Dokumentation
│   └── template.json         # Vorlage fuer run.json
├── tests/
│   ├── test_search.py        # Engine-Smoke-Tests
│   └── fixtures/
│       └── h4/               # 4x4 Kontroll-Fixture
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Tests

```bash
PYTHONPATH="src;verifier" python tests/test_search.py
```

Baut eine bekannte 4x4 Hadamard-Matrix per Williamson, prüft Orthogonalität,
und verifiziert dass die Suche die Lösung in <100 Schritten findet.

## Verifier (4x4 Kontrolle)

```bash
python verifier/verify.py tests/fixtures/h4/candidate.csv --order 4 --audit --require-filename candidate.csv
python verifier/review.py tests/fixtures/h4 --order 4
```

## Suche starten

`src/search.py` direkt ausführen oder importieren:

```python
from search import run, direct_search, williamson_search

# Kurzer Testlauf
run(strategy="both", steps=5000, seed=42)

# Oder: Konstanten oben in search.py anpassen und direkt ausführen
```

```bash
python src/search.py
```

Ergebnis einer exakten Lösung landet in `output/candidate.csv`.

Parameter (ganz oben in `src/search.py` änderbar):

- `STEPS` — Anzahl Suchschritte
- `SEED` — Zufallsseed
- `STRATEGY` — `"direct"`, `"williamson"` oder `"both"`

## Strategien

### Direct Local Search

Arbeitet direkt auf der 668×668-Matrix. Flippt symmetrische Eintragspaare und akzeptiert Schritte, die die Energie nicht erhöhen. 222.778 Zeilenpaare müssen orthogonal werden.

### Williamson-Konstruktion

Nutzt 668 = 4 × 167. Baut die Matrix aus vier symmetrischen zirkulanten 167×167-Blöcken. Reduziert die Variablen von 446.224 auf 4×84 = 336, mit 83 quadratischen Nebenbedingungen.

## Lizenz

MIT
