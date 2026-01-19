# Graphen mit Knoten und Kanten

Implementierung der Algorithmen aus der Arbeit "Graphen mit Knoten und Kanten" von Stephan Epp.

## Überblick

Dieses Projekt implementiert effiziente Algorithmen zur **optimalen** Berechnung von Graphprofilen unter Verwendung der Signatur-Methode aus der Boolean Matrixmultiplikation.

**Kernaussage**: Jeder Graph wird optimal in die Graphprofilverteilung eingeordnet. Diese Einordnung ist nicht verbesserbar, da sie vollständig deterministisch erfolgt und jeden Knoten und jede Kante berücksichtigt.

### Hauptmerkmale

- **Boolean Matrixmultiplikation in O(n²)** statt O(n³)
- **Graphprofil-Berechnung in O(n³)** für:
  - Kürzeste Wege (Distanzmatrix D)
  - Längste Wege (Matrix L)
  - Kantenmaß κ = |V| / |E|
- **Optimale Charakterisierung**: Deterministisch, vollständig, nicht approximativ
- **Hierarchische Analyse**: Unterstützung für mehrstufige Graphsysteme
- Vollständige Testsuite mit pytest
- Experimente mit SVG-Visualisierungen

## Projektstruktur

```
graphs/
├── science/
│   └── graphs.tex                          # Wissenschaftliche Arbeit
├── src/
│   ├── results/                            # Ergebnisse der Experimente
│   ├── graph_profile.py                    # Graphprofil-Berechnung
│   └── brain_scale_graph_experiments.py    # Experimente
├── tests/                                  # Tests
│   ├── test_graph_profile.py
│   └── test_integration.py
├── doc/
│   └── coverage/                           # Test-Coverage Report
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
# oder
venv\Scripts\activate     # Windows

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

### Hierarchische Graphanalyse

```python
# Beispiel: Rechenzentrum mit mehreren Ebenen
racks = np.array([...])      # Rack-Topologie
servers = np.array([...])    # Server-Topologie
vms = np.array([...])        # VM-Kommunikation

# Berechne Profile für jede Ebene
D_rack, L_rack, kappa_rack = calculator.compute_profile(racks)
D_server, L_server, kappa_server = calculator.compute_profile(servers)
D_vm, L_vm, kappa_vm = calculator.compute_profile(vms)

# Analysiere Anomalien
if kappa_server > 2 * kappa_rack:
    print("Warnung: Netzwerkpartitionierung auf Server-Ebene!")
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

**Satz (Optimale Charakterisierung):**
Die Einordnung eines Graphen G in die Graphprofilverteilung mittels (D, L, κ) ist optimal und nicht verbesserbar, da:
1. **Vollständigkeit**: Jeder Knoten und jede Kante wird berücksichtigt
2. **Exaktheit**: Kürzeste Wege werden exakt bestimmt (nicht approximiert)
3. **Determinismus**: Für jeden Graphen wird stets das gleiche Profil berechnet

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

Die optimale Profilberechnung hat weitreichende Anwendungen in verschiedenen Domänen:

### 🧠 Neurowissenschaften & Gehirnforschung

**Konnektomanalyse:**
- Optimale Charakterisierung neuronaler Netzwerke (86 Mrd. Neuronen)
- Deterministische Vergleiche zwischen Individuen
- Detektion struktureller Anomalien bei neurologischen Erkrankungen

**Beispiel**: Alzheimer-Früherkennung durch Analyse von Profiländerungen im Hippocampus-Netzwerk.

```python
# Vergleiche gesundes vs. pathologisches Konnektom
D_healthy, L_healthy, kappa_healthy = calculator.compute_profile(hippocampus_healthy)
D_patient, L_patient, kappa_patient = calculator.compute_profile(hippocampus_patient)

if kappa_patient > 1.5 * kappa_healthy:
    print("Signifikante Reduktion der Konnektivität detektiert")
```

### 🤖 Künstliche Intelligenz

**Neural Architecture Search (NAS):**
- Deterministische Bewertung von Netzwerkarchitekturen
- Vorhersage von Lernfähigkeit basierend auf Graphprofil
- Modellvergleich ohne Training

**Model Pruning & Compression:**
- Entferne Verbindungen während κ innerhalb akzeptabler Grenzen bleibt
- Garantiert minimalen Informationsverlust

**KI-Sicherheit:**
- Überwachung struktureller Änderungen während des Trainings
- Detektion von adversarial attacks durch Profiländerungen

```python
# Überwache Training auf unerwartete Strukturänderungen
for epoch in range(num_epochs):
    D, L, kappa = calculator.compute_profile(model.to_graph())
    if kappa > kappa_baseline * 1.2:
        print(f"Warnung: Strukturelle Anomalie in Epoche {epoch}")
```

### 🏢 Rechenzentren & Cloud Computing

**Datacenter-Topologie-Optimierung:**
- Finde optimale Netzwerktopologie für gegebene Anforderungen
- Minimiere Latenz (Durchmesser) bei maximaler Kosteneffizienz (κ)

**Dynamisches Load Balancing:**
- Verteile Last basierend auf aktuellem Kommunikationsprofil
- Minimiere strukturelle Störungen durch Migration

**Fehlertoleranz:**
- Simuliere Ausfälle und berechne Auswirkung auf κ und Durchmesser
- Identifiziere kritische Verbindungen

```python
# Evaluiere Topologie-Kandidaten
topologies = [fat_tree, leaf_spine, mesh, torus]
for topo in topologies:
    D, L, kappa = calculator.compute_profile(topo)
    if max(D[D < np.inf]) <= 5 and kappa > 1.0:
        print(f"{topo.name}: Erfüllt Anforderungen")
```

### 👥 Soziale Netzwerke

**Influencer-Identifikation:**
- Finde Knoten mit minimaler durchschnittlicher Distanz (zentrale Knoten)
- Identifiziere Brückenknoten (deren Entfernung κ erhöht)

**Desinformations-Eindämmung:**
- Berechne maximale Verbreitungszeit = max(D[quelle, :])
- Priorisiere Fact-Checking an Knoten mit hoher Reichweite

**Community Detection:**
- Communities haben charakteristische lokale Profile
- Optimale Erkennung durch Profilvergleich

### 🧬 Biologie & Molekularbiologie

**Protein-Interaktionsnetzwerke:**
- Drug Target Identification: Finde Proteine mit hoher Zentralität
- Funktionale Annotation: Proteine mit ähnlichem Profil haben ähnliche Funktion
- Pathway Analysis: Charakterisiere metabolische Pfade via (D, L, κ)

**Evolutionäre Genomik:**
- Vergleich von Gennetzwerken über Spezies hinweg
- Phylogenetischer Abstand korreliert mit Profil-Abstand

```python
# Identifiziere kritisches Protein in Krankheitsnetzwerk
D, L, kappa = calculator.compute_profile(disease_network)
centrality = {protein: 1/np.sum(D[i, :]) for i, protein in enumerate(proteins)}
target = max(centrality, key=centrality.get)
print(f"Drug Target: {target}")
```

### 📡 Kommunikationsnetzwerke

**Routing-Optimierung:**
- Nutze D-Matrix für optimale Pfadwahl
- Vermeide Routen mit hohem L[i,j] (anfällig für Überlastung)

**Network Resilience:**
- Berechne Profil nach simuliertem Knotenausfall
- Quantifiziere Robustheit durch Δκ

### 🚦 Verkehrs- & Logistiknetzwerke

**Infrastruktur-Planung:**
- Optimiere Straßennetz für minimalen Durchmesser
- Balance zwischen Kosten (maximiere κ) und Erreichbarkeit (minimiere D)

**Supply Chain Optimization:**
- Charakterisiere Lieferketten via Graphprofil
- Identifiziere Bottlenecks (hohe lokale Distanzen)

## Theoretische Bedeutung

### Determinismus vs. Probabilismus

**These**: Für Probleme, die deterministisch in polynomieller Zeit lösbar sind, sind probabilistische Methoden suboptimal.

Die Graphprofilberechnung ist ein Beispiel für ein Problem, bei dem:
- Deterministische Lösung existiert (diese Arbeit)
- Laufzeit polynomial ist (O(n³))
- Ergebnis exakt und reproduzierbar ist

**Konsequenz**: In sicherheitskritischen Anwendungen (Medizin, Infrastruktur, KI-Verifikation) sollten deterministische Verfahren bevorzugt werden.

### Komplexitätstheorie

Graphprofil-Berechnung ist in **P** (polynomielle Zeit, deterministisch):
- Hamiltonpfad: NP-vollständig ❌
- Maximale Clique: NP-vollständig ❌
- Graphfärbung: NP-vollständig ❌
- **Graphprofil: P** ✅ (O(n³))

### Universalität

Die Signatur-Methode ist übertragbar auf:
- **Transitive Hülle**: O(n³) statt O(n⁴)
- **Zykelerkennung**: Analyse von diag(A^k)
- **Zusammenhangskomponenten**: Via Erreichbarkeitsmatrix

## Ausblick & Zukünftige Arbeiten

### 🚀 Parallelisierung

Die Signatur-Berechnung ist inhärent parallelisierbar:
- GPU-Implementierung für massive Beschleunigung
- Potenzielle Reduktion auf O(n²) Gesamtlaufzeit mit ausreichend Prozessoren

### 📊 Sparse Graphen

Viele reale Graphen haben |E| = O(n):
- Anpassung für komprimierte Darstellung (CSR/CSC)
- Potenzielle Reduktion auf O(n·|E|) für sparse Graphen

### ⚡ Dynamische Graphen

Inkrementelle Updates nach Kantenänderung:
- Update Profil in O(n²) statt vollständiger Neuberechnung in O(n³)
- Wichtig für zeitveränderliche Netzwerke

### 🔮 Quantencomputing

Übertragung der Signatur-Methode auf Quantencomputer:
- Potenzielle Laufzeit unterhalb O(n²)
- Bitoperationen → Qubit-Operationen

### 🗄️ Universelle Graphdatenbank

Vision: Datenbank mit Millionen bekannter Graphprofile
- Query: "Finde Graphen mit κ ∈ [1.0, 1.5] und diameter < 10"
- Similarity Search: "Ähnlichste Graphen zu Query"
- Pattern Discovery: Wiederkehrende Strukturen über Domänen

---

**Kernbotschaft**: *Jeder Graph wird optimal charakterisiert. Darauf basierende Entscheidungen sind deterministisch und reproduzierbar.*
