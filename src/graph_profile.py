"""
Graphprofil-Berechnung mit Boolean Matrixmultiplikation.
Implementiert die Algorithmen aus der Arbeit graphs.tex
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import sys
import os

from src.boolean_matrix_multiplier import BooleanMatrixMultiplier

class GraphProfileCalculator:
    """
    Berechnet das vollständige Profil eines Graphen:
    - Kürzeste Wege (Distanzmatrix D)
    - Längste Wege (Matrix L)
    - Kantenmaß κ = |V| / |E|
    
    Unterstützt auch gerichtete Informationsverarbeitung für Gehirn-Modelle:
    - Vorwärts (rechtsherum/clockwise): Feed-forward Processing
    - Rückwärts (linksherum/counter-clockwise): Feedback/Rekurrenz
    """

    def __init__(self):
        self.multiplier = BooleanMatrixMultiplier()

    def compute_profile(self, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Berechnet das vollständige Profil eines Graphen.
        
        Implementiert Algorithmus 3 aus graphs.tex
        
        Args:
            adj_matrix: Adjazenzmatrix des Graphen (n x n)
            
        Returns:
            Tuple (D, L, κ) mit:
            - D: Distanzmatrix (kürzeste Wege)
            - L: Matrix der längsten Wege
            - κ: Kantenmaß |V| / |E|
        """
        n = adj_matrix.shape[0]
        m = int(np.sum(adj_matrix))
        
        # Maß für die Anzahl der Kanten
        kappa = n / m if m > 0 else float('inf')
        
        # Initialisierung
        D = np.full((n, n), np.inf)
        L = np.zeros((n, n), dtype=int)
        np.fill_diagonal(D, 0)  # Distanz zu sich selbst ist 0
        
        # WICHTIG: Starte mit k=1 und Current = A (nicht A^0)
        current = adj_matrix.copy()
        
        # Iterative Wegberechnung - O(n) Iterationen
        for k in range(1, n):
            for i in range(n):
                for j in range(n):
                    if current[i, j] == 1:
                        # Kürzester Weg: nur beim ersten Mal setzen
                        if D[i, j] == np.inf:
                            D[i, j] = k
                        
                        # Längster Weg: immer aktualisieren (überschreiben)
                        L[i, j] = k
            
            # Nächste Potenz via Boolean Multiplikation - O(n²)
            current = self.multiplier.multiply_optimized(current, adj_matrix)
        
        return D, L, kappa

    def compute_shortest_paths(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Berechnet nur die kürzesten Wege (Algorithmus 2 aus graphs.tex).
        
        Args:
            adj_matrix: Adjazenzmatrix des Graphen (n x n)
            
        Returns:
            Distanzmatrix D mit kürzesten Weglängen
        """
        n = adj_matrix.shape[0]
        
        # Initialisierung
        D = np.full((n, n), np.inf)
        np.fill_diagonal(D, 0)
        
        current = adj_matrix.copy()
        
        for k in range(1, n):
            for i in range(n):
                for j in range(n):
                    if current[i, j] == 1 and D[i, j] == np.inf:
                        D[i, j] = k
            
            # Boolean Multiplikation
            current = self.multiplier.multiply_optimized(current, adj_matrix)
        
        return D

    def compute_longest_paths(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Berechnet nur die längsten Wege in azyklischen Graphen
        (Algorithmus 3 aus graphs.tex).
        
        Args:
            adj_matrix: Adjazenzmatrix eines azyklischen Graphen (n x n)
            
        Returns:
            Matrix L mit längsten Weglängen
        """
        n = adj_matrix.shape[0]
        
        # Initialisierung
        L = np.zeros((n, n), dtype=int)
        current = adj_matrix.copy()
        
        for k in range(1, n):
            for i in range(n):
                for j in range(n):
                    if current[i, j] == 1:
                        L[i, j] = k
            
            # Boolean Multiplikation
            current = self.multiplier.multiply_optimized(current, adj_matrix)
        
        return L

    def compute_kappa(self, adj_matrix: np.ndarray) -> float:
        """
        Berechnet das Kantenmaß κ = |V| / |E|.
        
        Args:
            adj_matrix: Adjazenzmatrix des Graphen
            
        Returns:
            Kantenmaß κ
        """
        n = adj_matrix.shape[0]
        m = int(np.sum(adj_matrix))
        
        return n / m if m > 0 else float('inf')

    def get_profile_statistics(self, D: np.ndarray, L: np.ndarray, 
                               kappa: float) -> Dict[str, float]:
        """
        Berechnet Statistiken über das Profil.
        
        Args:
            D: Distanzmatrix
            L: Matrix der längsten Wege
            kappa: Kantenmaß
            
        Returns:
            Dictionary mit Statistiken
        """
        # Entferne Diagonale und unendliche Werte für Statistiken
        finite_distances = D[np.isfinite(D) & (D > 0)]
        
        stats = {
            'kappa': kappa,
            'avg_shortest_path': np.mean(finite_distances) if len(finite_distances) > 0 else 0,
            'max_shortest_path': np.max(finite_distances) if len(finite_distances) > 0 else 0,
            'min_shortest_path': np.min(finite_distances) if len(finite_distances) > 0 else 0,
            'avg_longest_path': np.mean(L[L > 0]) if np.any(L > 0) else 0,
            'max_longest_path': np.max(L),
            'diameter': np.max(finite_distances) if len(finite_distances) > 0 else 0,
        }
        
        return stats

    def compute_profile_with_direction(self, adj_matrix: np.ndarray, 
                                       direction: str = 'forward') -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Berechnet das Graphprofil mit spezifischer Verarbeitungsrichtung.
        
        Für Gehirn-Informationsverarbeitung:
        - 'forward' (rechtsherum/clockwise): Feed-forward Processing
        - 'backward' (linksherum/counter-clockwise): Feedback-Schleifen
        
        Args:
            adj_matrix: Adjazenzmatrix des Graphen (n x n)
            direction: 'forward' für Vorwärts, 'backward' für Rückwärts
            
        Returns:
            Tuple (D, L, κ) mit:
            - D: Distanzmatrix (kürzeste Wege)
            - L: Matrix der längsten Wege
            - κ: Kantenmaß |V| / |E|
        """
        if direction not in ['forward', 'backward']:
            raise ValueError(f"direction muss 'forward' oder 'backward' sein, nicht '{direction}'")
        
        n = adj_matrix.shape[0]
        m = int(np.sum(adj_matrix))
        
        # Kantenmaß
        kappa = n / m if m > 0 else float('inf')
        
        # Initialisierung
        D = np.full((n, n), np.inf)
        L = np.zeros((n, n), dtype=int)
        np.fill_diagonal(D, 0)
        
        # Wähle Matrix basierend auf Richtung
        if direction == 'forward':
            # Normal: A × A × A ... (Feed-forward)
            current = adj_matrix.copy()
            multiplication_matrix = adj_matrix
        else:  # backward
            # Rückwärts: A × A^T × A^T ... (Feedback)
            current = adj_matrix.copy()
            multiplication_matrix = adj_matrix.T
        
        # Iterative Wegberechnung
        for k in range(1, n):
            for i in range(n):
                for j in range(n):
                    if current[i, j] == 1:
                        if D[i, j] == np.inf:
                            D[i, j] = k
                        L[i, j] = k
            
            # Multiplikation in gewählter Richtung
            current = self.multiplier.multiply_optimized(current, multiplication_matrix)
        
        return D, L, kappa

    def compute_bidirectional_profile(self, adj_matrix: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Berechnet Graphprofile für beide Verarbeitungsrichtungen.
        
        Gehirn-Informationsverarbeitung hat von oben betrachtet eine Drehrichtung.
        Bei Menschen ist die Hauptrichtung rechtsherum (negative Winkelrichtung).
        Diese Methode analysiert beide Richtungen:
        
        - Forward (clockwise/rechtsherum): Feed-forward Processing
        - Backward (counter-clockwise/linksherum): Feedback/Rekurrenz
        
        Dies ermöglicht die Analyse von ad-hoc Informationsselektion, bei der
        der entsprechende Weg durch das Gehirn gewählt wird, wie es zur aktuellen
        Synapsenverknüpfung passt.
        
        Args:
            adj_matrix: Adjazenzmatrix des Gehirn-/Neuronalen Graphen (n x n)
            
        Returns:
            Dictionary mit:
            - 'forward': Tuple (D_forward, L_forward, κ_forward)
            - 'backward': Tuple (D_backward, L_backward, κ_backward)
            - 'combined': Statistik über beide Richtungen
        """
        # Berechne Profile in beiden Richtungen
        D_fwd, L_fwd, kappa_fwd = self.compute_profile_with_direction(adj_matrix, 'forward')
        D_bwd, L_bwd, kappa_bwd = self.compute_profile_with_direction(adj_matrix, 'backward')
        
        # Kombinierte Statistiken
        # Durchschnittliche Weglänge über beide Richtungen
        finite_fwd = D_fwd[np.isfinite(D_fwd) & (D_fwd > 0)]
        finite_bwd = D_bwd[np.isfinite(D_bwd) & (D_bwd > 0)]
        
        combined_stats = {
            'avg_path_forward': np.mean(finite_fwd) if len(finite_fwd) > 0 else 0,
            'avg_path_backward': np.mean(finite_bwd) if len(finite_bwd) > 0 else 0,
            'diameter_forward': np.max(finite_fwd) if len(finite_fwd) > 0 else 0,
            'diameter_backward': np.max(finite_bwd) if len(finite_bwd) > 0 else 0,
            'directionality_ratio': (np.mean(finite_fwd) / np.mean(finite_bwd) 
                                    if len(finite_bwd) > 0 and np.mean(finite_bwd) > 0 else 1.0),
            'kappa_forward': kappa_fwd,
            'kappa_backward': kappa_bwd,
        }
        
        return {
            'forward': (D_fwd, L_fwd, kappa_fwd),
            'backward': (D_bwd, L_bwd, kappa_bwd),
            'combined': combined_stats
        }

    def analyze_brain_information_flow(self, adj_matrix: np.ndarray, 
                                       primary_direction: str = 'forward') -> Dict[str, Any]:
        """
        Analysiert die Gehirn-Informationsverarbeitung mit Drehrichtung.
        
        Das Gehirn hat von oben betrachtet eine Drehrichtung zur allgemeinen
        Verarbeitung von Informationen. Bei Menschen ist diese rechtsherum
        (negative Winkelrichtung). Bei der ad-hoc Informationsselektion wird
        nach Bedarf der entsprechende Weg gewählt.
        
        Args:
            adj_matrix: Adjazenzmatrix des neuronalen Netzwerks
            primary_direction: 'forward' (default, rechtsherum) oder 'backward'
            
        Returns:
            Dictionary mit detaillierter Analyse:
            - primary_profile: Profil in Hauptrichtung
            - secondary_profile: Profil in Gegenrichtung
            - information_flow_efficiency: Effizienz der Informationsverarbeitung
            - adaptivity_score: Wie gut das Netzwerk alternative Pfade nutzen kann
        """
        # Berechne bidirektionales Profil
        bidirectional = self.compute_bidirectional_profile(adj_matrix)
        
        # Wähle primäre und sekundäre Richtung
        if primary_direction == 'forward':
            primary = bidirectional['forward']
            secondary = bidirectional['backward']
            direction_name = 'clockwise (rechtsherum)'
        else:
            primary = bidirectional['backward']
            secondary = bidirectional['forward']
            direction_name = 'counter-clockwise (linksherum)'
        
        D_primary, L_primary, kappa_primary = primary
        D_secondary, L_secondary, kappa_secondary = secondary
        
        # Berechne Effizienz und Adaptivität
        finite_primary = D_primary[np.isfinite(D_primary) & (D_primary > 0)]
        finite_secondary = D_secondary[np.isfinite(D_secondary) & (D_secondary > 0)]
        
        # Informationsfluss-Effizienz: Wie kurz sind die Wege?
        efficiency_primary = 1.0 / np.mean(finite_primary) if len(finite_primary) > 0 else 0
        efficiency_secondary = 1.0 / np.mean(finite_secondary) if len(finite_secondary) > 0 else 0
        
        # Adaptivitäts-Score: Verhältnis der Effizienzen
        # Hoher Wert: Primärrichtung dominant (spezialisiert)
        # Wert nahe 1: Beide Richtungen ähnlich effizient (flexibel)
        adaptivity = efficiency_primary / efficiency_secondary if efficiency_secondary > 0 else float('inf')
        
        analysis = {
            'primary_direction': direction_name,
            'primary_profile': {
                'D': D_primary,
                'L': L_primary,
                'kappa': kappa_primary,
                'avg_path_length': np.mean(finite_primary) if len(finite_primary) > 0 else 0,
                'diameter': np.max(finite_primary) if len(finite_primary) > 0 else 0,
            },
            'secondary_profile': {
                'D': D_secondary,
                'L': L_secondary,
                'kappa': kappa_secondary,
                'avg_path_length': np.mean(finite_secondary) if len(finite_secondary) > 0 else 0,
                'diameter': np.max(finite_secondary) if len(finite_secondary) > 0 else 0,
            },
            'information_flow_efficiency': {
                'primary': float(efficiency_primary),
                'secondary': float(efficiency_secondary),
            },
            'adaptivity_score': float(adaptivity),
            'interpretation': self._interpret_brain_flow(efficiency_primary, efficiency_secondary, adaptivity)
        }
        
        return analysis

    def _interpret_brain_flow(self, eff_primary: float, eff_secondary: float, 
                             adaptivity: float) -> str:
        """
        Interpretiert die Gehirn-Informationsfluss-Charakteristiken.
        
        Args:
            eff_primary: Effizienz der Primärrichtung
            eff_secondary: Effizienz der Sekundärrichtung
            adaptivity: Adaptivitäts-Score
            
        Returns:
            Interpretationstext
        """
        if adaptivity > 2.0:
            return ("Stark gerichtete Verarbeitung: Das Netzwerk ist auf "
                   "Feed-forward-Verarbeitung spezialisiert. Typisch für "
                   "sensorische Verarbeitung (z.B. früher visueller Kortex).")
        elif adaptivity > 1.3:
            return ("Mäßig gerichtete Verarbeitung: Primärrichtung dominant, "
                   "aber signifikante Feedback-Kapazität vorhanden. Typisch für "
                   "höhere kortikale Regionen mit Top-down-Modulation.")
        elif adaptivity > 0.7:
            return ("Balancierte bidirektionale Verarbeitung: Feed-forward und "
                   "Feedback ähnlich effizient. Typisch für assoziative Regionen "
                   "und rekurrente Netzwerke (z.B. präfrontaler Kortex).")
        else:
            return ("Feedback-dominierte Verarbeitung: Rückwärtsrichtung effizienter. "
                   "Ungewöhnlich in biologischen Netzwerken, könnte auf Fehler "
                   "oder spezielle rekurrente Strukturen hinweisen.")
