# Graphen mit Knoten und Kanten

Implementierung der Algorithmen aus der Arbeit "Graphen mit Knoten und Kanten" von Stephan Epp.

## Überblick

Dieses Projekt implementiert effiziente Algorithmen zur Berechnung von Graphprofilen unter Verwendung der Signatur-Methode aus der Boolean Matrixmultiplikation.

### Hauptmerkmale

- **Boolean Matrixmultiplikation in O(n²)** statt O(n³)
- **Graphprofil-Berechnung in O(n³)** für:
  - Kürzeste Wege (Distanzmatrix D)
  - Längste Wege (Matrix L)
  - Kantenmaß κ = |V| / |E|
- Vollständige Testsuite mit pytest
- Experimente mit SVG-Visualisierungen

## Projektstruktur

```
graphs/
├── src/
│   └── graph_profile.py          # Graphprofil-Berechnung
├── tests/
│   ├── test_graph_profile.py
│   └── test_integration.py
├── doc/
│   └── coverage/                 # Test-Coverage Reports
├── pyproject.toml
├── pytest.ini
└── README.md
```

## Installation

### Installation

```bash
# Repository-Verzeichnis
cd graphs

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren
pip install -e .

# Test-Abhängigkeiten installieren
pip install -e ".[test]"
```

## Verwendung

### Boolean Matrixmultiplikation

```python
from boolean_matrix_multiplier import BooleanMatrixMultiplier
import numpy as np

multiplier = BooleanMatrixMultiplier()

A = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 1, 0]])

B = np.array([[0, 1],
              [1, 0],
              [1, 1]])

# O(n²) Multiplikation mit Signaturen
C = multiplier.multiply_optimized(A, B)
```

### Graphprofil-Berechnung

```python
from src.graph_profile import GraphProfileCalculator
import numpy as np

calculator = GraphProfileCalculator()

# Adjazenzmatrix eines Graphen
adj = np.array([[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]])

# Berechne vollständiges Profil
D, L, kappa = calculator.compute_profile(adj)

print(f"Kantenmaß κ = {kappa}")
print(f"Distanzmatrix:\n{D}")
print(f"Längste Wege:\n{L}")

# Statistiken abrufen
stats = calculator.get_profile_statistics(D, L, kappa)
print(f"Durchmesser: {stats['diameter']}")
```

## Tests ausführen

```bash
# Alle Tests
pytest

# Mit Coverage-Report
pytest --cov=src --cov-report=html

# Nur spezifische Tests
pytest tests/test_bool_matrix_mult.py

# Verbose-Modus
pytest -v

# Coverage-Report öffnen
open doc/coverage/index.html  # Mac/Linux
```

## Experimente

```bash
# Experimente ausführen
python experiments/run_experiments.py
```

Dies führt folgende Experimente durch:

1. **Boolean Matrixmultiplikation**: Vergleich naive O(n³) vs. Signatur O(n²)
2. **Graphprofile**: Analyse verschiedener Graphtypen (vollständig, Pfad, Zufall)

Die Ergebnisse werden als SVG-Dateien in `experiments/results/` gespeichert.

## Algorithmen

### Algorithmus 1: Boolean Matrixmultiplikation (O(n²))

```
Phase 1: Signatur-Berechnung
- Berechne Zeilen-Signaturen von A: O(n²)
- Berechne Spalten-Signaturen von B: O(n²)

Phase 2: Multiplikation
- Für alle i,j: bitweise AND der Signaturen: O(n²)
- Gesamt: O(n²)
```

### Algorithmus 2: Kürzeste Wege (O(n³))

```
Für k = 1 bis n-1:
  Für alle i,j:
    Falls Current[i,j] = 1 und D[i,j] = ∞:
      D[i,j] = k
  Current = Current · A  (Boolean Multiplikation in O(n²))
```

### Algorithmus 3: Vollständige Profilberechnung (O(n³))

Berechnet gleichzeitig kürzeste Wege, längste Wege und Kantenmaß in O(n³) Gesamtzeit.

## Theoretische Grundlagen

Die Implementierung basiert auf folgenden Sätzen:

**Satz (Wege und Matrixpotenzen):** 
Für die k-te Potenz A^k der Adjazenzmatrix gilt: (A^k)[i,j] = 1 genau dann, wenn ein Weg der Länge k von i nach j existiert.

**Satz (Laufzeit):**
- Boolean Matrixmultiplikation mit Signaturen: O(n²)
- Graphprofil-Berechnung: O(n³)
- Speicherbedarf: O(n²)