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

### Voraussetzungen

- Python 3.8 oder höher
- NumPy

### Installation

```bash
# Repository-Verzeichnis
cd graphs

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac

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
# Beispiel: Pfadgraph 0 -> 1 -> 2
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
print(f"Maximaler längster Weg: {stats['max_longest_path']}")
```

## Tests ausführen

```bash
# Alle Tests
pytest

# Mit Coverage-Report
pytest --cov=src --cov-report=html

# Nur spezifische Tests
pytest tests/test_graph_profile.py

# Verbose-Modus
pytest -v

# Coverage-Report öffnen
open doc/coverage/index.html  # Mac/Linux
start doc/coverage/index.html # Windows
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

**Kernidee**: Jede Zeile von A und jede Spalte von B wird als Bitkette kodiert. Die Multiplikation C[i,j] reduziert sich dann auf eine bitweise AND-Operation zwischen den Signaturen.

### Algorithmus 2: Kürzeste Wege (O(n³))

```
Für k = 1 bis n-1:
  Für alle i,j:
    Falls Current[i,j] = 1 und D[i,j] = ∞:
      D[i,j] = k
  Current = Current · A  (Boolean Multiplikation in O(n²))
```

**Idee**: Current = A^k enthält genau die Wege der Länge k. Beim ersten Auftreten eines Weges wird die kürzeste Distanz gesetzt.

### Algorithmus 3: Vollständige Profilberechnung (O(n³))

Berechnet gleichzeitig kürzeste Wege, längste Wege und Kantenmaß in O(n³) Gesamtzeit.

```
Für k = 1 bis n-1:
  Für alle i,j:
    Falls Current[i,j] = 1:
      Falls D[i,j] = ∞: D[i,j] = k     (kürzester Weg)
      L[i,j] = k                       (längster Weg, überschreiben)
  Current = Current · A
```

## Theoretische Grundlagen

Die Implementierung basiert auf folgenden Sätzen:

**Satz (Wege und Matrixpotenzen):** 
Für die k-te Potenz A^k der Adjazenzmatrix gilt: (A^k)[i,j] = 1 genau dann, wenn ein Weg der Länge k von i nach j existiert.

**Beweis**: Durch Induktion über k. Für k=1 ist A¹ = A und enthält direkte Kanten. Für k+1 gilt: (A^(k+1))[i,j] = ⋁ₗ (A^k[i,l] ∧ A[l,j]). Dies ist genau dann 1, wenn es ein l gibt mit einem Weg der Länge k von i nach l und einer Kante von l nach j.

**Satz (Laufzeit):**
- Boolean Matrixmultiplikation mit Signaturen: O(n²)
- Graphprofil-Berechnung: O(n³)
- Speicherbedarf: O(n²)

**Beweis**: Die n-1 Iterationen führen jeweils eine Boolean-Multiplikation in O(n²) durch und werten n² Einträge aus. Gesamt: O(n) · O(n²) = O(n³).

## Beispiele

### Pfadgraph

```python
# Pfadgraph: 0 -> 1 -> 2 -> 3
adj = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

D, L, kappa = calculator.compute_profile(adj)
# D[0,3] = 3 (kürzester Weg: 0->1->2->3)
# L[0,3] = 3 (längster Weg: gleich, da azyklisch)
# kappa = 4/3 (4 Knoten, 3 Kanten)
```

### Vollständiger Graph

```python
# K₃: Jeder Knoten verbunden mit jedem
adj = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

D, L, kappa = calculator.compute_profile(adj)
# Alle D[i,j] = 1 für i≠j (direkte Kanten)
# kappa = 3/6 = 0.5 (3 Knoten, 6 Kanten)
```

### Zyklischer Graph

```python
# Zyklus: 0 -> 1 -> 2 -> 3 -> 0
adj = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
])

D, L, kappa = calculator.compute_profile(adj)
# D[0,2] = 2 (kürzester Weg: 0->1->2)
# L[0,2] = 3 (längster Weg: 0->3->2->1, bei gerichtetem Graph)
```

## Anwendungen

Die effiziente Profilberechnung findet Anwendung in:

- **Netzwerkanalyse**: Charakterisierung der Kommunikationsstruktur
- **Graphklassifikation**: Einordnung von Graphen anhand struktureller Eigenschaften
- **Algorithmenauswahl**: Wahl des optimalen Algorithmus basierend auf Grapheigenschaften
- **Social Network Analysis**: Ermittlung von Distanzen und zentralen Knoten
- **Routing-Algorithmen**: Berechnung kürzester und längster Pfade in Netzwerken

## Ausblick

Zukünftige Arbeiten können die Signatur-Technik auf weitere Graphprobleme übertragen:

- **Transitive Hülle** in O(n³) statt O(n⁴)
- **Zykelerkennung** durch Analyse der Diagonale von A^k
- **Zusammenhangskomponenten** durch iterative Erreichbarkeitsanalyse
- **Parallelisierung** der Signatur-Berechnung für Multicore-Systeme
- **Sparse Graphen**: Optimierung für dünnbesetzte Adjazenzmatrizen
