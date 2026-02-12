"""
Graphprofil-Berechnung mit Boolean Matrixmultiplikation.
Implementiert die Algorithmen aus der Arbeit graphs.tex
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os

from src.boolean_matrix_multiplier import BooleanMatrixMultiplier

class GraphProfileCalculator:
    """
    Berechnet das vollständige Profil eines Graphen:
    - Kürzeste Wege (Distanzmatrix D)
    - Längste Wege (Matrix L)
    - Kantenmaß κ = |V| / |E|
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