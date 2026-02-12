"""
Gehirn-Informationsverarbeitung mit Drehrichtung.

Das Gehirn hat von oben betrachtet eine Drehrichtung zur allgemeinen
Verarbeitung von Informationen. Die Drehrichtung ist bei Menschen rechtsherum.
Also von oben in negativer Winkelrichtung.

Bei der Ad-hoc Informationsselektion wird nach Bedarf der entsprechende Weg
durch das Gehirn gewählt, wie es zur aktuellen Synapsenverknüpfung passt.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.graph_profile import GraphProfileCalculator


class BrainInformationProcessor:
    """
    Verarbeitet Informationen in neuronalen Graphen unter Berücksichtigung
    der Drehrichtung des Gehirns.
    
    Die Drehrichtung ist rechtsherum (im Uhrzeigersinn), was von oben betrachtet
    einer negativen Winkelrichtung entspricht.
    """
    
    def __init__(self):
        self.calculator = GraphProfileCalculator()
        
    def compute_rotational_direction(self, node_positions: np.ndarray,
                                     center: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Berechnet die Drehrichtung für Knoten im 2D-Raum.
        
        Rechtsherum (Uhrzeigersinn) entspricht negativer Winkelrichtung.
        
        Args:
            node_positions: Array der Knotenpositionen (n x 2), wobei jede Zeile [x, y] ist
            center: Zentrum der Rotation (Standard: Schwerpunkt)
            
        Returns:
            Array der Winkel in negativer Richtung (radians)
        """
        if center is None:
            center = np.mean(node_positions, axis=0)
        
        # Berechne Vektoren vom Zentrum zu jedem Knoten
        vectors = node_positions - center
        
        # Berechne Winkel (atan2 gibt Winkel gegen Uhrzeigersinn)
        # Für Uhrzeigersinn (rechtsherum) negieren wir die Winkel
        angles = -np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # Normalisiere auf [0, 2π)
        angles = angles % (2 * np.pi)
        
        return angles
    
    def sort_nodes_by_rotation(self, node_positions: np.ndarray,
                               center: Optional[np.ndarray] = None,
                               clockwise: bool = True) -> np.ndarray:
        """
        Sortiert Knoten nach ihrer Winkelposition.
        
        Args:
            node_positions: Array der Knotenpositionen (n x 2)
            center: Zentrum der Rotation (Standard: Schwerpunkt)
            clockwise: True für Uhrzeigersinn (Standard), False für Gegenuhrzeigersinn
            
        Returns:
            Indizes der sortierten Knoten
        """
        angles = self.compute_rotational_direction(node_positions, center)
        
        if clockwise:
            # Rechtsherum: aufsteigende negative Winkel
            return np.argsort(angles)
        else:
            # Gegenuhrzeigersinn
            return np.argsort(-angles)
    
    def select_path_by_synaptic_strength(self, adj_matrix: np.ndarray,
                                         start_node: int,
                                         end_node: int,
                                         synaptic_weights: Optional[np.ndarray] = None,
                                         rotation_preference: bool = True) -> List[int]:
        """
        Wählt ad-hoc einen Weg durch das neuronale Netzwerk basierend auf
        Synapsenverknüpfungen und Rotationspräferenz.
        
        Args:
            adj_matrix: Adjazenzmatrix des Graphen
            start_node: Startknoten
            end_node: Zielknoten
            synaptic_weights: Optionale Gewichte der Synapsen (Standard: uniform)
            rotation_preference: Bevorzuge Pfade in Rotationsrichtung
            
        Returns:
            Liste der Knoten im gewählten Pfad
        """
        n = adj_matrix.shape[0]
        
        # Standard-Gewichte wenn nicht angegeben
        if synaptic_weights is None:
            synaptic_weights = adj_matrix.astype(float)
        
        # Berechne alle kürzesten Wege
        D, L, _ = self.calculator.compute_profile(adj_matrix)
        
        # Wenn kein Weg existiert
        if np.isinf(D[start_node, end_node]):
            return []
        
        # Ad-hoc Pfadwahl: Greedy-Suche mit Rotationspräferenz
        path = [start_node]
        current = start_node
        visited = {start_node}
        
        while current != end_node:
            # Finde alle Nachbarn
            neighbors = []
            for next_node in range(n):
                if adj_matrix[current, next_node] == 1 and next_node not in visited:
                    # Prüfe, ob dieser Nachbar näher am Ziel ist
                    if D[next_node, end_node] < D[current, end_node]:
                        neighbors.append(next_node)
            
            if not neighbors:
                # Kein Fortschritt möglich, Backtracking notwendig
                # Für Einfachheit: gib partiellen Pfad zurück
                break
            
            # Wähle besten Nachbarn basierend auf Synapsenstärke
            best_neighbor = max(neighbors,
                              key=lambda n: synaptic_weights[current, n])
            
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor
        
        return path if current == end_node else []
    
    def compute_rotational_flow(self, adj_matrix: np.ndarray,
                               node_positions: np.ndarray,
                               center: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Berechnet den Informationsfluss unter Berücksichtigung der Rotationsrichtung.
        
        Quantifiziert, wie stark jede Kante zur rechtsdrehenden
        Informationsverarbeitung beiträgt.
        
        Args:
            adj_matrix: Adjazenzmatrix
            node_positions: Knotenpositionen (n x 2)
            center: Rotationszentrum
            
        Returns:
            Matrix mit Rotationsfluss-Werten für jede Kante
        """
        n = adj_matrix.shape[0]
        
        # Berechne Winkel für alle Knoten
        angles = self.compute_rotational_direction(node_positions, center)
        
        # Initialisiere Fluss-Matrix
        flow = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] == 1:
                    # Berechne Winkeldifferenz (rechtsherum)
                    angle_diff = (angles[j] - angles[i]) % (2 * np.pi)
                    
                    # Fluss ist maximal bei ~π/2 (90° rechtsherum)
                    # und minimal bei ~3π/2 (270° oder -90°)
                    # Verwende Cosinus für glatte Gewichtung
                    flow[i, j] = (1 + np.cos(angle_diff - np.pi/2)) / 2
        
        return flow
    
    def analyze_rotational_properties(self, adj_matrix: np.ndarray,
                                     node_positions: np.ndarray) -> dict:
        """
        Analysiert Rotationseigenschaften eines neuronalen Graphen.
        
        Args:
            adj_matrix: Adjazenzmatrix
            node_positions: Knotenpositionen
            
        Returns:
            Dictionary mit Rotationsmetriken
        """
        # Berechne Rotationsfluss
        flow = self.compute_rotational_flow(adj_matrix, node_positions)
        
        # Berechne Standardprofil
        D, L, kappa = self.calculator.compute_profile(adj_matrix)
        
        # Rotationsstatistiken
        edges_with_flow = adj_matrix == 1
        avg_flow = np.mean(flow[edges_with_flow]) if np.any(edges_with_flow) else 0
        
        # Berechne Rotationseffizienz
        # (Wie gut unterstützt die Struktur rechtsdrehende Verarbeitung?)
        rotation_efficiency = avg_flow
        
        return {
            'rotation_efficiency': float(rotation_efficiency),
            'avg_rotational_flow': float(avg_flow),
            'max_rotational_flow': float(np.max(flow)),
            'min_rotational_flow': float(np.min(flow[edges_with_flow])) if np.any(edges_with_flow) else 0,
            'kappa': float(kappa),
            'diameter': float(np.max(D[np.isfinite(D) & (D > 0)])) if np.any(np.isfinite(D) & (D > 0)) else 0,
        }
    
    def generate_circular_layout(self, n: int, radius: float = 1.0) -> np.ndarray:
        """
        Generiert eine zirkuläre Anordnung von Knoten (typisch für Gehirnregionen).
        
        Args:
            n: Anzahl Knoten
            radius: Radius des Kreises
            
        Returns:
            Knotenpositionen (n x 2)
        """
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.zeros((n, 2))
        positions[:, 0] = radius * np.cos(angles)
        positions[:, 1] = radius * np.sin(angles)
        return positions
