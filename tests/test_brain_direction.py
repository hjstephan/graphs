"""
Tests für Gehirn-Informationsverarbeitungsrichtung.
"""

import pytest
import numpy as np
from src.graph_profile import GraphProfileCalculator


class TestBrainDirectionProcessing:
    """Tests für gerichtete Informationsverarbeitung im Gehirn."""

    @pytest.fixture
    def calculator(self):
        """Fixture für GraphProfileCalculator Instanz."""
        return GraphProfileCalculator()

    def test_forward_direction_basic(self, calculator):
        """Test der Vorwärts-Verarbeitung (clockwise/rechtsherum)."""
        # Einfacher Feed-forward Graph: 0 -> 1 -> 2
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        D, L, kappa = calculator.compute_profile_with_direction(adj, 'forward')
        
        # Überprüfe Distanzen
        assert D[0, 1] == 1
        assert D[0, 2] == 2
        assert D[1, 2] == 1
        
        # Kantenmaß
        assert abs(kappa - 3/2) < 1e-10

    def test_backward_direction_basic(self, calculator):
        """Test der Rückwärts-Verarbeitung (counter-clockwise/linksherum)."""
        # Graph mit Feedback-Struktur
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]  # Feedback von 2 zu 0
        ])
        
        D_bwd, L_bwd, kappa_bwd = calculator.compute_profile_with_direction(adj, 'backward')
        
        # Rückwärts nutzt transponierte Matrix
        # Überprüfe, dass Berechnung durchläuft
        assert D_bwd.shape == (3, 3)
        assert L_bwd.shape == (3, 3)
        assert abs(kappa_bwd - 1.0) < 1e-10  # 3 Knoten, 3 Kanten

    def test_bidirectional_profile(self, calculator):
        """Test der bidirektionalen Profil-Berechnung."""
        # Symmetrischer Graph (Feedback-Verbindungen)
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        
        result = calculator.compute_bidirectional_profile(adj)
        
        # Prüfe Struktur
        assert 'forward' in result
        assert 'backward' in result
        assert 'combined' in result
        
        # Prüfe Forward-Profil
        D_fwd, L_fwd, kappa_fwd = result['forward']
        assert D_fwd.shape == (4, 4)
        
        # Prüfe Backward-Profil
        D_bwd, L_bwd, kappa_bwd = result['backward']
        assert D_bwd.shape == (4, 4)
        
        # Prüfe kombinierte Statistiken
        combined = result['combined']
        assert 'avg_path_forward' in combined
        assert 'avg_path_backward' in combined
        assert 'directionality_ratio' in combined

    def test_brain_information_flow_analysis(self, calculator):
        """Test der vollständigen Gehirn-Informationsfluss-Analyse."""
        # Hierarchisches Feed-forward Netzwerk (wie sensorischer Kortex)
        adj = np.array([
            [0, 1, 1, 0, 0],  # Input-Layer
            [0, 0, 0, 1, 1],  # Hidden-Layer
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],  # Output-Layer
            [0, 0, 0, 0, 0]
        ])
        
        analysis = calculator.analyze_brain_information_flow(adj, 'forward')
        
        # Prüfe Struktur
        assert 'primary_direction' in analysis
        assert 'primary_profile' in analysis
        assert 'secondary_profile' in analysis
        assert 'information_flow_efficiency' in analysis
        assert 'adaptivity_score' in analysis
        assert 'interpretation' in analysis
        
        # Primary direction sollte clockwise sein
        assert 'clockwise' in analysis['primary_direction']
        
        # Effizienz-Werte sollten positiv sein
        assert analysis['information_flow_efficiency']['primary'] > 0
        
        # Adaptivity score sollte > 1 sein (Feed-forward dominant)
        assert analysis['adaptivity_score'] >= 1.0

    def test_cortical_column_structure(self, calculator):
        """Test mit kortikaler Säulenstruktur (6 Schichten)."""
        # Vereinfachte 6-Schicht kortikale Struktur
        # Layer 1-6 mit typischen Verbindungen
        n_per_layer = 3
        n_total = n_per_layer * 6
        adj = np.zeros((n_total, n_total), dtype=int)
        
        # Verbindungen zwischen Schichten (vereinfacht)
        for layer in range(5):
            start = layer * n_per_layer
            end = start + n_per_layer
            next_start = end
            next_end = next_start + n_per_layer
            
            # Jeder Knoten in Schicht i verbindet zu einem in Schicht i+1
            for i in range(start, end):
                if next_start < n_total:
                    adj[i, next_start + (i % n_per_layer)] = 1
        
        # Feedback von Layer 6 zu Layer 4 (typisch)
        adj[15, 9] = 1  # Layer 6 -> Layer 4
        
        analysis = calculator.analyze_brain_information_flow(adj, 'forward')
        
        # Sollte moderat gerichtete Verarbeitung zeigen
        assert analysis['adaptivity_score'] > 1.0
        assert 'Feed-forward' in analysis['interpretation'] or 'gerichtete' in analysis['interpretation']

    def test_direction_parameter_validation(self, calculator):
        """Test der Parametervalidierung für Richtung."""
        adj = np.array([[0, 1], [0, 0]])
        
        # Valide Richtungen
        calculator.compute_profile_with_direction(adj, 'forward')
        calculator.compute_profile_with_direction(adj, 'backward')
        
        # Invalide Richtung sollte Fehler werfen
        with pytest.raises(ValueError):
            calculator.compute_profile_with_direction(adj, 'invalid')

    def test_symmetric_graph_directionality(self, calculator):
        """Test mit symmetrischem Graph (keine Vorzugsrichtung)."""
        # Vollständig symmetrischer Graph
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        result = calculator.compute_bidirectional_profile(adj)
        
        # Bei symmetrischem Graph sollten beide Richtungen ähnlich sein
        combined = result['combined']
        
        # Directionality ratio sollte nahe 1 sein
        assert 0.8 < combined['directionality_ratio'] < 1.2

    def test_acyclic_vs_cyclic_directionality(self, calculator):
        """Test: azyklischer vs. zyklischer Graph."""
        # Azyklischer Graph (DAG)
        adj_acyclic = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        # Zyklischer Graph
        adj_cyclic = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]  # Zyklus
        ])
        
        analysis_acyclic = calculator.analyze_brain_information_flow(adj_acyclic, 'forward')
        analysis_cyclic = calculator.analyze_brain_information_flow(adj_cyclic, 'forward')
        
        # Azyklischer Graph sollte höhere oder gleiche Adaptivity (mehr/gleich gerichtet) haben
        # Zyklischer Graph ist oft flexibler (niedrigere Adaptivity)
        assert analysis_acyclic['adaptivity_score'] >= analysis_cyclic['adaptivity_score']

    def test_empty_graph_direction(self, calculator):
        """Test mit leerem Graph (keine Kanten)."""
        adj = np.zeros((4, 4), dtype=int)
        
        result = calculator.compute_bidirectional_profile(adj)
        
        # Sollte keine Abstürze verursachen
        assert result['combined']['avg_path_forward'] == 0
        assert result['combined']['avg_path_backward'] == 0

    def test_single_node_direction(self, calculator):
        """Test mit einzelnem Knoten."""
        adj = np.array([[0]])
        
        analysis = calculator.analyze_brain_information_flow(adj, 'forward')
        
        # Sollte mit trivialen Werten durchlaufen
        assert analysis['information_flow_efficiency']['primary'] == 0
        assert analysis['information_flow_efficiency']['secondary'] == 0

    @pytest.mark.parametrize("primary_dir", ['forward', 'backward'])
    def test_both_primary_directions(self, calculator, primary_dir):
        """Parametrisierter Test für beide Primärrichtungen."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        analysis = calculator.analyze_brain_information_flow(adj, primary_dir)
        
        # Prüfe, dass Analysis durchläuft
        assert 'primary_direction' in analysis
        assert 'adaptivity_score' in analysis
        assert analysis['adaptivity_score'] > 0

    def test_interpretation_categories(self, calculator):
        """Test der verschiedenen Interpretations-Kategorien."""
        # Stark gerichteter Graph (hohe Adaptivity)
        adj_directed = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        analysis_directed = calculator.analyze_brain_information_flow(adj_directed, 'forward')
        
        # Sollte "gerichtete Verarbeitung" erwähnen
        assert ('gerichtete' in analysis_directed['interpretation'] or 
                'Feed-forward' in analysis_directed['interpretation'])
        
        # Symmetrischer Graph (niedrige Adaptivity)
        adj_balanced = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        analysis_balanced = calculator.analyze_brain_information_flow(adj_balanced, 'forward')
        
        # Adaptivity sollte nahe 1 sein
        assert 0.5 < analysis_balanced['adaptivity_score'] < 1.5
