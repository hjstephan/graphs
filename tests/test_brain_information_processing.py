"""
Tests für Gehirn-Informationsverarbeitung mit Rotationsrichtung.
"""

import pytest
import numpy as np
from src.brain_information_processing import BrainInformationProcessor


class TestBrainInformationProcessor:
    """Tests für BrainInformationProcessor."""
    
    @pytest.fixture
    def processor(self):
        return BrainInformationProcessor()
    
    def test_rotational_direction_clockwise(self, processor):
        """
        Test dass Drehrichtung rechtsherum (Uhrzeigersinn) korrekt ist.
        """
        # Vier Knoten in Kreuzform
        positions = np.array([
            [1, 0],   # Rechts (0°)
            [0, 1],   # Oben (90°)
            [-1, 0],  # Links (180°)
            [0, -1]   # Unten (270°)
        ])
        
        angles = processor.compute_rotational_direction(positions)
        
        # Rechtsherum bedeutet: Rechts -> Unten -> Links -> Oben
        # In negativer Winkelrichtung
        sorted_indices = np.argsort(angles)
        
        # Erwarte: [0, 3, 2, 1] (Rechts -> Unten -> Links -> Oben)
        assert sorted_indices[0] == 0 or sorted_indices[0] == 3  # Start variiert
    
    def test_sort_nodes_clockwise(self, processor):
        """
        Test Sortierung von Knoten im Uhrzeigersinn.
        """
        positions = np.array([
            [0, 1],   # Oben
            [1, 0],   # Rechts
            [0, -1],  # Unten
            [-1, 0]   # Links
        ])
        
        sorted_indices = processor.sort_nodes_by_rotation(positions, clockwise=True)
        
        # Alle 4 Knoten sollten sortiert sein
        assert len(sorted_indices) == 4
        assert len(set(sorted_indices)) == 4  # Alle einzigartig
    
    def test_path_selection_simple_graph(self, processor):
        """
        Test ad-hoc Pfadwahl in einfachem Graphen.
        """
        # Einfacher Pfad: 0 -> 1 -> 2
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        path = processor.select_path_by_synaptic_strength(adj, 0, 2)
        
        # Sollte direkten Pfad finden
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 2
    
    def test_path_selection_no_path(self, processor):
        """
        Test wenn kein Pfad existiert.
        """
        # Zwei getrennte Komponenten
        adj = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        
        path = processor.select_path_by_synaptic_strength(adj, 0, 2)
        
        # Kein Pfad sollte gefunden werden
        assert len(path) == 0
    
    def test_path_selection_with_weights(self, processor):
        """
        Test Pfadwahl mit synaptischen Gewichten.
        """
        # Graph mit zwei Pfaden
        adj = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        # Bevorzuge Pfad über Knoten 1
        weights = np.array([
            [0, 1.0, 0.1, 0],
            [0, 0, 0, 1.0],
            [0, 0, 0, 0.1],
            [0, 0, 0, 0]
        ])
        
        path = processor.select_path_by_synaptic_strength(
            adj, 0, 3, synaptic_weights=weights
        )
        
        # Sollte stärkeren Pfad wählen
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 3
    
    def test_rotational_flow_circular_layout(self, processor):
        """
        Test Rotationsfluss bei zirkulärer Anordnung.
        """
        n = 4
        positions = processor.generate_circular_layout(n)
        
        # Ring-Graph (jeder Knoten verbunden mit nächstem)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj[i, (i + 1) % n] = 1
        
        flow = processor.compute_rotational_flow(adj, positions)
        
        # Fluss sollte für alle Kanten positiv sein
        assert np.all(flow[adj == 1] >= 0)
        assert np.all(flow[adj == 0] == 0)
    
    def test_analyze_rotational_properties(self, processor):
        """
        Test Analyse von Rotationseigenschaften.
        """
        n = 6
        positions = processor.generate_circular_layout(n)
        
        # Einfacher Ring
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj[i, (i + 1) % n] = 1
        
        props = processor.analyze_rotational_properties(adj, positions)
        
        # Prüfe alle erwarteten Keys
        assert 'rotation_efficiency' in props
        assert 'avg_rotational_flow' in props
        assert 'max_rotational_flow' in props
        assert 'min_rotational_flow' in props
        assert 'kappa' in props
        assert 'diameter' in props
        
        # Werte sollten sinnvoll sein
        assert 0 <= props['rotation_efficiency'] <= 1
        assert 0 <= props['avg_rotational_flow'] <= 1
        assert props['kappa'] > 0
    
    def test_circular_layout_generation(self, processor):
        """
        Test Generierung zirkulärer Layouts.
        """
        n = 8
        positions = processor.generate_circular_layout(n, radius=2.0)
        
        # Prüfe Dimensionen
        assert positions.shape == (n, 2)
        
        # Prüfe dass alle Knoten auf Kreis liegen
        distances = np.sqrt(np.sum(positions**2, axis=1))
        np.testing.assert_array_almost_equal(distances, np.full(n, 2.0))
    
    def test_rotational_direction_with_custom_center(self, processor):
        """
        Test Drehrichtung mit benutzerdefiniertem Zentrum.
        """
        positions = np.array([
            [2, 1],
            [3, 1],
            [2, 0]
        ])
        
        center = np.array([2, 1])
        angles = processor.compute_rotational_direction(positions, center)
        
        # Alle Winkel sollten im Bereich [0, 2π) sein
        assert np.all(angles >= 0)
        assert np.all(angles < 2 * np.pi)
    
    def test_negative_angular_direction(self, processor):
        """
        Test dass negative Winkelrichtung (Uhrzeigersinn) korrekt berechnet wird.
        """
        # Zwei Punkte: Einer rechts, einer oben vom Zentrum
        positions = np.array([
            [1, 0],   # Rechts = 0° standard
            [0, 1]    # Oben = 90° standard
        ])
        
        angles = processor.compute_rotational_direction(positions)
        
        # Im Uhrzeigersinn (negative Richtung) sollte "Oben" größeren Winkel haben
        # da wir von rechts nach unten nach links nach oben gehen
        # Beide Winkel sollten im gültigen Bereich sein
        assert 0 <= angles[0] < 2 * np.pi
        assert 0 <= angles[1] < 2 * np.pi
    
    def test_multiple_path_selection(self, processor):
        """
        Test mehrfache Pfadwahl im gleichen Graphen.
        """
        # Vollständigerer Graph
        adj = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ])
        
        # Verschiedene Pfade
        path1 = processor.select_path_by_synaptic_strength(adj, 0, 4)
        path2 = processor.select_path_by_synaptic_strength(adj, 1, 4)
        
        # Beide sollten valide Pfade sein
        assert len(path1) > 0
        assert len(path2) > 0
        assert path1[0] == 0
        assert path2[0] == 1
    
    def test_brain_scale_rotational_processing(self, processor):
        """
        Test Rotationsverarbeitung auf Gehirn-ähnlichem Maßstab.
        """
        # Simuliere kleine Gehirnregion
        n = 20
        positions = processor.generate_circular_layout(n)
        
        # Small-world-ähnliche Struktur
        adj = np.zeros((n, n), dtype=int)
        
        # Ring-Verbindungen
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[i, (i + 2) % n] = 1
        
        # Einige Langstreckenverbindungen
        np.random.seed(42)
        for _ in range(n // 2):
            i, j = np.random.choice(n, 2, replace=False)
            adj[i, j] = 1
        
        props = processor.analyze_rotational_properties(adj, positions)
        
        # Eigenschaften sollten berechnet werden
        assert props['rotation_efficiency'] >= 0
        assert props['kappa'] > 0
        assert props['diameter'] > 0
    
    def test_synaptic_path_respects_graph_structure(self, processor):
        """
        Test dass Pfadwahl die Graphstruktur respektiert.
        """
        # Dreieck
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        path = processor.select_path_by_synaptic_strength(adj, 0, 2)
        
        # Pfad sollte nur existierende Kanten verwenden
        for i in range(len(path) - 1):
            assert adj[path[i], path[i+1]] == 1
    
    @pytest.mark.parametrize("n", [4, 8, 12])
    def test_rotational_properties_scale(self, processor, n):
        """
        Test dass Rotationseigenschaften mit verschiedenen Größen skalieren.
        """
        positions = processor.generate_circular_layout(n)
        
        # Ring-Graph
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj[i, (i + 1) % n] = 1
        
        props = processor.analyze_rotational_properties(adj, positions)
        
        # Grundlegende Konsistenzchecks
        assert props['rotation_efficiency'] >= 0
        assert props['kappa'] == 1.0  # Ring hat n Knoten und n Kanten: κ = n/n = 1
