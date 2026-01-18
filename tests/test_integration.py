"""
Integrationstests für das Gesamtsystem.
"""

import pytest
import numpy as np
import sys
import os


from boolean_matrix_multiplier import BooleanMatrixMultiplier
from src.graph_profile import GraphProfileCalculator


class TestIntegration:
    """Integrationstests für Boolean Multiplikation und Graphprofile."""

    @pytest.fixture
    def multiplier(self):
        return BooleanMatrixMultiplier()

    @pytest.fixture
    def calculator(self):
        return GraphProfileCalculator()

    @pytest.mark.integration
    def test_complete_workflow_small_graph(self, calculator):
        """
        Test des kompletten Workflows für einen kleinen Graphen.
        """
        # Erstelle einen einfachen Graphen: 0 -> 1 -> 2
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Berechne Profil
        D, L, kappa = calculator.compute_profile(adj)
        
        # Verifiziere Ergebnisse
        assert kappa == 3/2  # 3 Knoten, 2 Kanten
        
        # Kürzeste Wege
        assert D[0, 0] == 0
        assert D[0, 1] == 1
        assert D[0, 2] == 2
        assert np.isinf(D[1, 0])
        
        # Längste Wege
        assert L[0, 1] == 1
        assert L[0, 2] == 2
        assert L[1, 2] == 1
        
        # Statistiken
        stats = calculator.get_profile_statistics(D, L, kappa)
        assert stats['diameter'] == 2
        assert stats['max_longest_path'] == 2

    @pytest.mark.integration
    def test_matrix_power_equals_paths(self, multiplier, calculator):
        """
        Test dass A^k korrekt Wege der Länge k repräsentiert.
        """
        # Dreiecksgraph
        adj = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Berechne A^2
        A2 = multiplier.multiply_optimized(adj, adj)
        
        # A^2[0,2] sollte 1 sein (Weg 0->1->2)
        assert A2[0, 2] == 1
        
        # Vergleiche mit Profil-Berechnung
        D, L, kappa = calculator.compute_profile(adj)
        
        # Wo A^2[i,j] = 1, sollte D[i,j] <= 2 sein
        for i in range(3):
            for j in range(3):
                if A2[i, j] == 1 and i != j:
                    assert D[i, j] <= 2

    @pytest.mark.integration
    def test_consistency_across_methods(self, calculator):
        """
        Test dass verschiedene Berechnungsmethoden konsistent sind.
        """
        # Zufälliger Graph
        np.random.seed(42)
        n = 10
        adj = np.random.randint(0, 2, size=(n, n))
        np.fill_diagonal(adj, 0)
        
        # Berechne mit verschiedenen Methoden
        D_combined, L_combined, kappa_combined = calculator.compute_profile(adj)
        D_separate = calculator.compute_shortest_paths(adj)
        L_separate = calculator.compute_longest_paths(adj)
        kappa_separate = calculator.compute_kappa(adj)
        
        # Vergleiche Ergebnisse
        np.testing.assert_array_equal(D_combined, D_separate)
        np.testing.assert_array_equal(L_combined, L_separate)
        assert kappa_combined == kappa_separate

    @pytest.mark.integration
    def test_profile_invariants(self, calculator):
        """
        Test wichtiger Invarianten des Graphprofils.
        """
        # Graph: 0 -> 1 -> 2 -> 3
        adj = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Invariante 1: Diagonale ist 0
        for i in range(4):
            assert D[i, i] == 0
        
        # Invariante 2: D ist symmetrisch bezüglich Erreichbarkeit
        # Wenn D[i,j] < inf, dann existiert Weg von i nach j
        assert D[0, 3] < np.inf
        assert D[0, 2] < np.inf
        
        # Invariante 3: Längster Weg >= kürzester Weg
        for i in range(4):
            for j in range(4):
                if D[i, j] < np.inf and L[i, j] > 0:
                    assert L[i, j] >= D[i, j]
        
        # Invariante 4: Transitivität
        # Wenn D[i,j] und D[j,k] endlich, dann auch D[i,k]
        if D[0, 1] < np.inf and D[1, 2] < np.inf:
            assert D[0, 2] < np.inf

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_graph_performance(self, calculator):
        """
        Performance-Test mit größerem Graphen.
        """
        import time
        
        n = 50
        np.random.seed(123)
        adj = np.random.randint(0, 2, size=(n, n))
        np.fill_diagonal(adj, 0)
        
        start = time.perf_counter()
        D, L, kappa = calculator.compute_profile(adj)
        end = time.perf_counter()
        
        time_taken = end - start
        
        # Sollte in vernünftiger Zeit laufen (< 5 Sekunden)
        assert time_taken < 5.0
        
        # Ergebnisse sollten korrekt dimensioniert sein
        assert D.shape == (n, n)
        assert L.shape == (n, n)
        assert isinstance(kappa, float)

    @pytest.mark.integration
    def test_graph_types_examples(self, calculator):
        """
        Test mit Beispiel-Graphen aus der Arbeit.
        """
        # Vollständiger Graph K_4
        n = 4
        K4 = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        
        D_k4, L_k4, kappa_k4 = calculator.compute_profile(K4)
        
        # Alle Paare haben Distanz 1
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert D_k4[i, j] == 1
        
        # Kantenmaß für K_4: 4 Knoten, 12 Kanten
        expected_kappa = 4 / 12
        assert abs(kappa_k4 - expected_kappa) < 1e-10
        
        # Pfadgraph P_4
        P4 = np.zeros((4, 4), dtype=int)
        for i in range(3):
            P4[i, i+1] = 1
        
        D_p4, L_p4, kappa_p4 = calculator.compute_profile(P4)
        
        # Längster Weg in P_4 ist 3
        assert L_p4[0, 3] == 3
        
        # Kantenmaß für P_4: 4 Knoten, 3 Kanten
        expected_kappa_p4 = 4 / 3
        assert abs(kappa_p4 - expected_kappa_p4) < 1e-10

    @pytest.mark.integration
    def test_error_handling(self, calculator):
        """
        Test der Fehlerbehandlung.
        """
        # Leere Matrix
        empty = np.array([])
        
        # Sollte nicht abstürzen, aber sinnvolle Ergebnisse liefern
        # oder einen klaren Fehler werfen
        try:
            D, L, kappa = calculator.compute_profile(empty.reshape(0, 0))
            # Wenn es funktioniert, sollten die Dimensionen stimmen
            assert D.shape == (0, 0)
            assert L.shape == (0, 0)
        except (ValueError, IndexError):
            # Akzeptabel wenn Fehler geworfen wird
            pass

    @pytest.mark.integration
    def test_disconnected_components_profile(self, calculator):
        """
        Test mit Graph mit mehreren Zusammenhangskomponenten.
        """
        # Zwei getrennte Dreiecke
        adj = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0]
        ])
        
        D, L, kappa = calculator.compute_profile(adj)
        
        # Innerhalb Komponenten erreichbar
        assert D[0, 1] == 1
        assert D[3, 4] == 1
        
        # Zwischen Komponenten nicht erreichbar
        assert np.isinf(D[0, 3])
        assert np.isinf(D[0, 4])
        assert np.isinf(D[1, 5])
        
        # Kantenmaß: 6 Knoten, 12 Kanten (6 pro Dreieck)
        assert abs(kappa - 0.5) < 1e-10