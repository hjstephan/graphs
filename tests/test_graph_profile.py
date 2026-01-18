"""
Tests für Graphprofil-Berechnung.
"""

import pytest
import numpy as np
import sys
import os

from src.graph_profile import GraphProfileCalculator


class TestGraphProfileCalculator:
    """Tests für die Graphprofil-Berechnung."""

    @pytest.fixture
    def calculator(self):
        """Fixture für GraphProfileCalculator Instanz."""
        return GraphProfileCalculator()

    def test_path_graph_profile(self, calculator):
        """Test mit Pfadgraph P_4."""
        # Pfadgraph: 0 -> 1 -> 2 -> 3
        adj = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Kantenmaß: κ = |V| / |E| = 4 / 3
        assert abs(kappa - 4/3) < 1e-10
        
        # Kürzeste Wege
        assert D[0, 1] == 1
        assert D[0, 2] == 2
        assert D[0, 3] == 3
        assert D[1, 2] == 1
        assert D[1, 3] == 2
        assert D[2, 3] == 1
        
        # Längste Wege (azyklisch)
        assert L[0, 3] == 3
        assert L[0, 2] == 2
        assert L[0, 1] == 1

    def test_complete_graph_profile(self, calculator):
        """Test mit vollständigem Graph K_3."""
        # Vollständiger Graph mit 3 Knoten
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Kantenmaß: κ = |V| / |E| = 3 / 6 = 0.5
        assert abs(kappa - 0.5) < 1e-10
        
        # Alle kürzesten Wege sind 1 (direkte Kanten)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert D[i, j] == 1

    def test_disconnected_graph(self, calculator):
        """Test mit nicht zusammenhängendem Graph."""
        # Zwei getrennte Komponenten
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Kantenmaß
        assert abs(kappa - 1.0) < 1e-10  # 4 / 4 = 1
        
        # Innerhalb Komponenten erreichbar
        assert D[0, 1] == 1
        assert D[2, 3] == 1
        
        # Zwischen Komponenten nicht erreichbar
        assert np.isinf(D[0, 2])
        assert np.isinf(D[0, 3])
        assert np.isinf(D[1, 2])
        assert np.isinf(D[1, 3])

    def test_single_node_graph(self, calculator):
        """Test mit Graph aus einem Knoten."""
        adj = np.array([[0]])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Kantenmaß: κ = 1 / 0 = inf
        assert np.isinf(kappa)
        
        # Distanz zu sich selbst ist 0
        assert D[0, 0] == 0

    def test_cycle_graph(self, calculator):
        """Test mit Zyklusgraph C_4."""
        # Zyklus: 0 -> 1 -> 2 -> 3 -> 0
        adj = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Kantenmaß: κ = 4 / 4 = 1
        assert abs(kappa - 1.0) < 1e-10
        
        # Kürzeste Wege im Zyklus
        assert D[0, 1] == 1
        assert D[0, 2] == 2
        assert D[0, 3] == 3
        assert D[1, 3] == 2

    def test_shortest_paths_only(self, calculator):
        """Test der separaten Berechnung kürzester Wege."""
        adj = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        D = calculator.compute_shortest_paths(adj)
        
        assert D[0, 0] == 0
        assert D[0, 1] == 1
        assert D[0, 2] == 1  # Direkter Weg ist kürzer
        assert D[1, 2] == 1

    def test_longest_paths_only(self, calculator):
        """Test der separaten Berechnung längster Wege."""
        adj = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        L = calculator.compute_longest_paths(adj)
        
        assert L[0, 1] == 1
        assert L[0, 2] == 2  # Längster Weg: 0 -> 1 -> 2
        assert L[1, 2] == 1

    def test_kappa_computation(self, calculator):
        """Test der Kantenmaß-Berechnung."""
        # Graph mit 5 Knoten und 7 Kanten
        adj = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ])
        
        kappa = calculator.compute_kappa(adj)
        
        # κ = 5 / 7
        assert abs(kappa - 5/7) < 1e-10

    def test_profile_statistics(self, calculator):
        """Test der Profil-Statistik-Berechnung."""
        adj = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        stats = calculator.get_profile_statistics(D, L, kappa)
        
        assert 'kappa' in stats
        assert 'avg_shortest_path' in stats
        assert 'max_shortest_path' in stats
        assert 'diameter' in stats
        assert 'max_longest_path' in stats
        
        assert stats['kappa'] == kappa
        assert stats['diameter'] > 0

    def test_star_graph(self, calculator):
        """Test mit Sterngraph."""
        # Zentrum ist Knoten 0, verbunden mit 1, 2, 3
        adj = np.array([
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Alle Blätter sind Distanz 1 vom Zentrum
        assert D[0, 1] == 1
        assert D[0, 2] == 1
        assert D[0, 3] == 1
        
        # Blätter sind nicht untereinander erreichbar
        assert np.isinf(D[1, 2])
        assert np.isinf(D[1, 3])
        assert np.isinf(D[2, 3])

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_various_graph_sizes(self, calculator, n):
        """Parametrisierter Test für verschiedene Graphgrößen."""
        # Erstelle Pfadgraph der Länge n
        adj = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            adj[i, i + 1] = 1
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Verifiziere Dimensionen
        assert D.shape == (n, n)
        assert L.shape == (n, n)
        
        # Verifiziere längsten Weg
        assert L[0, n-1] == n - 1

    def test_empty_graph_edges(self, calculator):
        """Test mit Graph ohne Kanten."""
        n = 4
        adj = np.zeros((n, n), dtype=int)
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Keine Kanten -> κ = inf
        assert np.isinf(kappa)
        
        # Nur Diagonale ist 0, Rest unendlich
        for i in range(n):
            assert D[i, i] == 0
            for j in range(n):
                if i != j:
                    assert np.isinf(D[i, j])