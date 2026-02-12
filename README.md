# Graphen mit Knoten und Kanten

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/hjstephan/graphs/releases)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](doc/coverage/)

Implementierung der Algorithmen aus der Arbeit "Graphen mit Knoten und Kanten" von Stephan Epp.

## 🎯 Überblick

Dieses Projekt implementiert effiziente Algorithmen zur **optimalen** Berechnung von Graphprofilen unter Verwendung der Signatur-Methode aus der Boolean Matrixmultiplikation.

**Kernaussage**: Jeder Graph wird optimal in die Graphprofilverteilung eingeordnet. Diese Einordnung ist nicht verbesserbar, da sie vollständig deterministisch erfolgt und jeden Knoten und jede Kante berücksichtigt.

### 🌟 Hauptmerkmale

- **Boolean Matrixmultiplikation in O(n²)** statt O(n³)
- **Graphprofil-Berechnung in O(n³)** für:
  - Kürzeste Wege (Distanzmatrix D)
  - Längste Wege (Matrix L)
  - Kantenmaß κ = |V| / |E|
- **Gehirn-Informationsverarbeitung mit Rotationsrichtung**:
  - Rechtsherum (Uhrzeigersinn) - negative Winkelrichtung
  - Ad-hoc Pfadwahl basierend auf Synapsenverknüpfungen
  - Rotationsfluss-Analyse für neuronale Netzwerke
- **Optimale Charakterisierung**: Deterministisch, vollständig, nicht approximativ
- **Hierarchische Analyse**: Unterstützung für mehrstufige Graphsysteme
- **Vollständige Testsuite**: 51 Tests mit 98% Code Coverage
- Experimente mit SVG-Visualisierungen

### 📊 Qualitätsmetriken

- ✅ **51 Tests** - Alle bestanden
- ✅ **98% Code Coverage** - Vollständige Testabdeckung
- ✅ **Type Hints** - Vollständige Typisierung
- ✅ **Dokumentation** - Umfassende Docstrings
- ✅ **Wissenschaftliche Arbeit** - 46 Seiten LaTeX-Dokumentation

## 📦 Release v1.1.0

Diese Version enthält die vollständige Implementierung der Algorithmen aus der wissenschaftlichen Arbeit mit folgenden Highlights:

### ✨ Neue Features
- 🧠 Gehirn-Informationsverarbeitung mit Rotationsrichtung
- 📊 Bidirektionale Graphprofil-Analyse (Forward/Backward)
- 🔄 Rotationsfluss-Analyse für zirkuläre Netzwerke
- 📈 Erweiterte Statistiken und Metriken

### 🔧 Verbesserungen
- 📝 Aktualisierte LaTeX-Dokumentation mit microtype-Paket
- 🧪 Erweiterte Testsuite mit 51 Tests
- 📊 98% Code Coverage
- 🗂️ Bessere Code-Struktur (alle Module in src/)

### 📄 Dokumentation
- 📖 46-seitige wissenschaftliche Arbeit (graphs.pdf)
- 📚 Vollständige API-Dokumentation
- 🎓 Tutorials und Beispiele
- 🔬 Experimentelle Validierung

## 📁 Projektstruktur

```
graphs/
├── science/
│   ├── graphs.tex                          # Wissenschaftliche Arbeit (LaTeX)
│   └── graphs.pdf                          # Kompilierte PDF (46 Seiten)
├── src/
│   ├── __init__.py
│   ├── graph_profile.py                    # Graphprofil-Berechnung (Kernmodul)
│   ├── brain_information_processing.py     # Gehirn-Rotationsanalyse
│   ├── brain_rotation_experiments.py       # Rotationsexperimente
│   ├── brain_scale_graph_experiments.py    # Skalierungsexperimente
│   ├── boolean_matrix_multiplier.py        # Boolean Matrix Ops
│   ├── demo_brain_direction.py             # Demo: Drehrichtung
│   ├── demo_brain_rotation.py              # Demo: Rotation
│   └── results/                            # Experimentelle Ergebnisse
├── tests/                                  # Testsuite (51 Tests)
│   ├── test_graph_profile.py               # Graphprofil-Tests
│   ├── test_brain_information_processing.py
│   ├── test_brain_direction.py
│   └── test_integration.py                 # Integrationstests
├── doc/
│   └── coverage/                           # HTML Coverage Report (98%)
├── pyproject.toml                          # Projekt-Konfiguration
└── README.md                               # Diese Datei
```

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- NumPy >= 1.20.0
- Git

### Schnellstart

```bash
# Repository klonen
git clone https://github.com/hjstephan/graphs.git
cd graphs

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install -e .

# Test-Abhängigkeiten installieren (optional)
pip install -e ".[test]"
```

## 📖 Verwendung

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

### Gehirn-Informationsverarbeitung mit Drehrichtung (NEU)

```python
from src.graph_profile import GraphProfileCalculator
import numpy as np

calculator = GraphProfileCalculator()

# Kortikale Säulenstruktur (vereinfacht)
cortical_column = np.array([...])  # 6 Schichten

# Analysiere Informationsfluss
analysis = calculator.analyze_brain_information_flow(cortical_column, 'forward')

print(f"Primäre Richtung: {analysis['primary_direction']}")
print(f"Adaptivitäts-Score: {analysis['adaptivity_score']}")
print(f"Interpretation: {analysis['interpretation']}")

# Bidirektionale Analyse (Forward + Feedback)
bidirectional = calculator.compute_bidirectional_profile(cortical_column)

print(f"Forward Effizienz: {bidirectional['combined']['avg_path_forward']}")
print(f"Backward Effizienz: {bidirectional['combined']['avg_path_backward']}")
print(f"Direktionalitäts-Ratio: {bidirectional['combined']['directionality_ratio']}")

# Spezifische Richtung berechnen
D_fwd, L_fwd, kappa_fwd = calculator.compute_profile_with_direction(
    cortical_column, direction='forward'
)
D_bwd, L_bwd, kappa_bwd = calculator.compute_profile_with_direction(
    cortical_column, direction='backward'
)
```

**Konzept**:
Das Gehirn hat von oben betrachtet eine Drehrichtung zur allgemeinen Verarbeitung von Informationen. Die Drehrichtung ist bei Menschen rechtsherum (negative Winkelrichtung). Bei der ad-hoc Informationsselektion wird nach Bedarf der entsprechende Weg durch das Gehirn gewählt, wie es zur aktuellen Synapsenverknüpfung passt.

**Anwendungen**:
- Analyse kortikaler Schichten-Strukturen (Layer 1-6)
- Feed-forward vs. Feedback-Verarbeitung
- Charakterisierung von Netzwerk-Adaptivität
- Ad-hoc Pfadselektions-Flexibilität

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

## Demonstration ausführen

```bash
# Gehirn-Informationsverarbeitung Demo
python demo_brain_direction.py
```

Dies demonstriert:
1. Grundlegende Drehrichtung (clockwise/counter-clockwise)
2. Kortikale Schichtenstruktur (6 Layer)
3. Ad-hoc Informationsselektion
4. Vergleich verschiedener Verarbeitungstypen

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

**Gehirn-Informationsverarbeitung mit Rotationsrichtung:**

Das Gehirn hat von oben betrachtet eine Drehrichtung zur allgemeinen Verarbeitung von Informationen. Die Drehrichtung ist bei Menschen **rechtsherum** (im Uhrzeigersinn), also von oben in negativer Winkelrichtung. Bei der Ad-hoc Informationsselektion wird nach Bedarf der entsprechende Weg durch das Gehirn gewählt, wie es zur aktuellen Synapsenverknüpfung passt.

```python
from src.brain_information_processing import BrainInformationProcessor

processor = BrainInformationProcessor()

# Generiere zirkuläres Layout für Gehirnregionen
n = 20
positions = processor.generate_circular_layout(n)

# Erstelle neuronales Netzwerk
adj = create_neural_network(n)

# Analysiere Rotationseigenschaften
props = processor.analyze_rotational_properties(adj, positions)
print(f"Rotationseffizienz: {props['rotation_efficiency']:.3f}")
print(f"Durchmesser: {props['diameter']}")

# Ad-hoc Pfadwahl basierend auf Synapsenverknüpfungen
synaptic_weights = get_synaptic_strengths(adj)
path = processor.select_path_by_synaptic_strength(
    adj, start=0, end=10, 
    synaptic_weights=synaptic_weights,
    rotation_preference=True  # Bevorzuge rechtsdrehende Pfade
)
print(f"Gewählter Informationspfad: {path}")
```
**Gerichtete Informationsverarbeitung:**
- **NEU**: Analyse der Gehirn-Informationsfluss-Richtung (clockwise/counter-clockwise)
- Das Gehirn hat von oben betrachtet eine Drehrichtung zur Informationsverarbeitung
- Bei Menschen ist diese rechtsherum (negative Winkelrichtung = clockwise)
- Ad-hoc Informationsselektion wählt Pfade basierend auf aktueller Synapsenverknüpfung

**Beispiel**: Alzheimer-Früherkennung durch Analyse von Profiländerungen im Hippocampus-Netzwerk.

```python
# Vergleiche gesundes vs. pathologisches Konnektom
D_healthy, L_healthy, kappa_healthy = calculator.compute_profile(hippocampus_healthy)
D_patient, L_patient, kappa_patient = calculator.compute_profile(hippocampus_patient)

if kappa_patient > 1.5 * kappa_healthy:
    print("Signifikante Reduktion der Konnektivität detektiert")

# NEU: Analysiere Informationsfluss-Richtung
analysis = calculator.analyze_brain_information_flow(hippocampus_patient, 'forward')
print(f"Adaptivitäts-Score: {analysis['adaptivity_score']}")
print(f"Interpretation: {analysis['interpretation']}")
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

## 🤝 Beitragen

Beiträge sind willkommen! Bitte beachten Sie folgende Richtlinien:

### Entwicklungs-Setup

```bash
# Repository forken und klonen
git clone https://github.com/<your-username>/graphs.git
cd graphs

# Entwicklungsumgebung einrichten
python -m venv venv
source venv/bin/activate
pip install -e ".[test]"

# Tests ausführen
pytest -v

# Coverage-Report generieren
pytest --cov=src --cov-report=html
```

### Pull Request Prozess

1. Erstellen Sie einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
2. Committen Sie Ihre Änderungen (`git commit -m 'Add some AmazingFeature'`)
3. Stellen Sie sicher, dass alle Tests bestehen (`pytest`)
4. Pushen Sie den Branch (`git push origin feature/AmazingFeature`)
5. Öffnen Sie einen Pull Request

### Code-Qualitätsstandards

- ✅ Alle Tests müssen bestehen (pytest)
- ✅ Code Coverage sollte mindestens 95% sein
- ✅ Type Hints für alle öffentlichen Funktionen
- ✅ Docstrings im Google-Stil
- ✅ PEP 8 Konformität

## 📜 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

## 👤 Autor

**Stephan Epp**
- Email: hjstephan86@gmail.com
- GitHub: [@hjstephan](https://github.com/hjstephan)

## 📚 Zitierung

Wenn Sie diese Arbeit in Ihrer Forschung verwenden, zitieren Sie bitte:

```bibtex
@misc{epp2024graphs,
  author = {Epp, Stephan},
  title = {Graphen mit Knoten und Kanten: Optimale Einordnung in die Graphprofilverteilung},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hjstephan/graphs}},
  version = {1.1.0}
}
```

## 🙏 Danksagungen

- NumPy-Community für die exzellente wissenschaftliche Computing-Bibliothek
- pytest-Entwickler für das hervorragende Testing-Framework
- LaTeX-Community für das professionelle Typesetting-System

## 📞 Support

Bei Fragen oder Problemen:
- 🐛 [Issues](https://github.com/hjstephan/graphs/issues) - Fehlerberichte und Feature-Requests
- 💬 [Discussions](https://github.com/hjstephan/graphs/discussions) - Allgemeine Fragen und Diskussionen
- 📧 Email: hjstephan86@gmail.com

---

**Kernbotschaft**: *Jeder Graph wird optimal mit Einordnung in die Graphprofilverteilung charakterisiert. Darauf basierende Entscheidungen sind deterministisch und reproduzierbar.*
