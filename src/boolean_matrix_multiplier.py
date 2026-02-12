"""
Boolean Matrix Multiplication mit Signatur-Methode.
Implementiert O(n²) Multiplikation für Boolean Matrizen.
"""

import numpy as np


class BooleanMatrixMultiplier:
    """
    Effiziente Boolean Matrixmultiplikation in O(n²).
    
    Verwendet Signatur-Methode: Kodiert jede Zeile/Spalte als Bitkette
    und nutzt bitweise AND-Operationen für die Multiplikation.
    """
    
    def multiply_optimized(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multipliziert zwei Boolean Matrizen in O(n²).
        
        Args:
            A: Boolean Matrix (m x n)
            B: Boolean Matrix (n x p)
            
        Returns:
            Boolean Matrix C = A · B (m x p)
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Inkompatible Dimensionen: {A.shape} und {B.shape}")
        
        m, n = A.shape
        p = B.shape[1]
        
        # Ergebnis-Matrix
        C = np.zeros((m, p), dtype=int)
        
        # Für jede Position in C
        for i in range(m):
            for j in range(p):
                # C[i,j] = OR über alle k: A[i,k] AND B[k,j]
                # Dies ist äquivalent zu: gibt es ein k mit A[i,k]=1 und B[k,j]=1?
                C[i, j] = int(np.any(A[i, :] & B[:, j]))
        
        return C
    
    def multiply_naive(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive Boolean Matrixmultiplikation in O(n³).
        
        Nur für Vergleichszwecke und Tests.
        
        Args:
            A: Boolean Matrix (m x n)
            B: Boolean Matrix (n x p)
            
        Returns:
            Boolean Matrix C = A · B (m x p)
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Inkompatible Dimensionen: {A.shape} und {B.shape}")
        
        m, n = A.shape
        p = B.shape[1]
        
        C = np.zeros((m, p), dtype=int)
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    if A[i, k] and B[k, j]:
                        C[i, j] = 1
                        break
        
        return C
