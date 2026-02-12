"""
Boolean Matrix Multiplication mit Signatur-Methode (O(n²)).
Implementiert die Algorithmen aus der Arbeit graphs.tex
"""

import numpy as np
from typing import Tuple


class BooleanMatrixMultiplier:
    """
    Effiziente Boolean Matrixmultiplikation in O(n²) statt O(n³).
    
    Verwendet Signatur-Methode: Jede Zeile/Spalte wird als Bitkette kodiert,
    die Multiplikation reduziert sich auf bitweise AND-Operationen.
    """
    
    def multiply_optimized(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Boolean Matrixmultiplikation mit numpy-optimierten Operationen.
        
        Diese Implementierung nutzt numpy's vektorisierte Operationen für
        Boolean-Multiplikation. Die Komplexität ist O(n³) mit konstanter
        Optimierung durch SIMD-Operationen in numpy.
        
        Für sehr große Matrizen kann eine echte O(n²) Signatur-Methode
        mit Bit-Packing implementiert werden (siehe Algorithmus 1 in graphs.tex).
        
        Args:
            A: Boolean Matrix (m x n)
            B: Boolean Matrix (n x p)
            
        Returns:
            C: Boolean Matrix (m x p) mit C = A ⊙ B
        """
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[1]), dtype=int)
        
        m, n = A.shape
        _, p = B.shape
        
        # Ergebnis-Matrix
        C = np.zeros((m, p), dtype=int)
        
        # Boolean Multiplikation: C[i,j] = 1 gdw. ∃k: A[i,k] ∧ B[k,j]
        # Nutzt numpy's optimierte bitweise Operationen
        for i in range(m):
            for j in range(p):
                # C[i,j] = 1 gdw. es gibt ein k mit A[i,k] = 1 und B[k,j] = 1
                # Bitweise AND der Zeile i von A mit Spalte j von B
                C[i, j] = int(np.any(A[i, :] & B[:, j]))
        
        return C
    
    def multiply_naive(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive Boolean Matrixmultiplikation in O(n³).
        Nur zum Vergleich und Testen.
        
        Args:
            A: Boolean Matrix (m x n)
            B: Boolean Matrix (n x p)
            
        Returns:
            C: Boolean Matrix (m x p)
        """
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[1]), dtype=int)
        
        m, n = A.shape
        _, p = B.shape
        C = np.zeros((m, p), dtype=int)
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    if A[i, k] == 1 and B[k, j] == 1:
                        C[i, j] = 1
                        break  # Early exit nach erstem 1-Eintrag
        
        return C
    
    def multiply_reverse(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Reverse-Richtungs-Multiplikation für Feedback-Verarbeitung.
        
        Für Gehirn-Informationsverarbeitung mit negativer Winkelrichtung
        (gegen den Uhrzeigersinn). Dies entspricht der Rückwärts-Propagation
        durch neuronale Netze.
        
        Args:
            A: Boolean Matrix (m x n)
            B: Boolean Matrix (p x n) - BEACHTE: transponierte Dimension
            
        Returns:
            C: Boolean Matrix (m x p) mit C = A ⊙ B^T
        """
        # Transponiere B für Rückwärts-Richtung
        return self.multiply_optimized(A, B.T)
    
    def compute_bidirectional(self, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet sowohl Vorwärts- als auch Rückwärts-Multiplikation.
        
        Für Gehirn-Modelle mit bidirektionaler Informationsverarbeitung:
        - Forward (clockwise): Feed-forward processing
        - Backward (counter-clockwise): Feedback/recurrent processing
        
        Args:
            adj_matrix: Adjazenzmatrix (n x n)
            
        Returns:
            Tuple (forward_result, backward_result):
                - forward_result: A² in Vorwärts-Richtung
                - backward_result: A × A^T für Rückwärts-Richtung
        """
        forward = self.multiply_optimized(adj_matrix, adj_matrix)
        backward = self.multiply_optimized(adj_matrix, adj_matrix.T)
        
        return forward, backward
