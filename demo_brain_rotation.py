#!/usr/bin/env python3
"""
Demonstration der Gehirn-Informationsverarbeitung mit Rotationsrichtung.

Zeigt wie das Gehirn Informationen rechtsherum (im Uhrzeigersinn, negative
Winkelrichtung) verarbeitet und Pfade ad-hoc basierend auf Synapsenverknüpfungen wählt.
"""

import numpy as np
from src.brain_information_processing import BrainInformationProcessor
from src.graph_profile import GraphProfileCalculator


def main():
    print("=" * 80)
    print("GEHIRN-INFORMATIONSVERARBEITUNG MIT ROTATIONSRICHTUNG")
    print("=" * 80)
    print()
    
    processor = BrainInformationProcessor()
    calculator = GraphProfileCalculator()
    
    # 1. Demonstriere Rotationsrichtung
    print("1. ROTATIONSRICHTUNG (Rechtsherum / Uhrzeigersinn)")
    print("-" * 80)
    
    n = 8
    positions = processor.generate_circular_layout(n, radius=1.0)
    
    print(f"Generiere {n} Knoten in zirkulärer Anordnung...")
    
    # Zeige Positionen und ihre Winkel
    angles = processor.compute_rotational_direction(positions)
    sorted_indices = processor.sort_nodes_by_rotation(positions, clockwise=True)
    
    print("\nKnoten sortiert nach Rotationsrichtung (rechtsherum):")
    for idx in sorted_indices[:5]:  # Zeige erste 5
        print(f"  Knoten {idx}: Position ({positions[idx, 0]:.2f}, {positions[idx, 1]:.2f}), "
              f"Winkel: {np.degrees(angles[idx]):.1f}°")
    
    # 2. Neuronales Netzwerk mit Rotationsanalyse
    print("\n2. NEURONALES NETZWERK MIT ROTATIONSANALYSE")
    print("-" * 80)
    
    # Erstelle Ring-Netzwerk (simuliert kortikale Regionen)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i, (i + 1) % n] = 1  # Ring
        adj[i, (i + 2) % n] = 1  # Überspringende Verbindungen
    
    m = int(np.sum(adj))
    print(f"Netzwerk: {n} Knoten, {m} Synapsen")
    
    # Analysiere Rotationseigenschaften
    props = processor.analyze_rotational_properties(adj, positions)
    
    print("\nRotationseigenschaften:")
    print(f"  Rotationseffizienz: {props['rotation_efficiency']:.3f}")
    print(f"  Durchschnittlicher Rotationsfluss: {props['avg_rotational_flow']:.3f}")
    print(f"  Kantenmaß κ = |V|/|E|: {props['kappa']:.3f}")
    print(f"  Netzwerk-Durchmesser: {props['diameter']:.1f}")
    
    # 3. Ad-hoc Pfadwahl basierend auf Synapsen
    print("\n3. AD-HOC PFADWAHL BASIEREND AUF SYNAPSENVERKNÜPFUNGEN")
    print("-" * 80)
    
    # Erstelle komplexeres Netzwerk
    n_complex = 12
    adj_complex = np.zeros((n_complex, n_complex), dtype=int)
    
    # Ring mit Abkürzungen
    for i in range(n_complex):
        adj_complex[i, (i + 1) % n_complex] = 1
        if i % 3 == 0:
            adj_complex[i, (i + 3) % n_complex] = 1
    
    # Synaptische Stärken (einige Verbindungen stärker als andere)
    synaptic_weights = adj_complex.astype(float)
    for i in range(n_complex):
        if (i + 1) % n_complex < n_complex:
            # Abkürzungen haben höhere Gewichte
            if adj_complex[i, (i + 3) % n_complex] == 1:
                synaptic_weights[i, (i + 3) % n_complex] = 2.0
    
    start = 0
    end = 6
    
    print(f"\nSuche Pfad von Knoten {start} zu Knoten {end}...")
    
    # Wähle Pfad basierend auf Synapsenverknüpfung
    path = processor.select_path_by_synaptic_strength(
        adj_complex, start, end,
        synaptic_weights=synaptic_weights,
        rotation_preference=True
    )
    
    if path:
        print(f"Gewählter Pfad: {' -> '.join(map(str, path))}")
        print(f"Pfadlänge: {len(path) - 1} Schritte")
    else:
        print("Kein Pfad gefunden")
    
    # 4. Rotationsfluss visualisieren
    print("\n4. ROTATIONSFLUSS-ANALYSE")
    print("-" * 80)
    
    positions_complex = processor.generate_circular_layout(n_complex)
    flow = processor.compute_rotational_flow(adj_complex, positions_complex)
    
    # Finde Kanten mit höchstem und niedrigstem Fluss
    edges_with_flow = adj_complex == 1
    flows = flow[edges_with_flow]
    
    print(f"\nRotationsfluss-Statistiken:")
    print(f"  Durchschnitt: {np.mean(flows):.3f}")
    print(f"  Maximum: {np.max(flows):.3f}")
    print(f"  Minimum: {np.min(flows):.3f}")
    print(f"  Standardabweichung: {np.std(flows):.3f}")
    
    # Zeige einige Kanten mit ihrem Fluss
    print("\nBeispiel-Kanten mit Rotationsfluss:")
    count = 0
    for i in range(n_complex):
        for j in range(n_complex):
            if adj_complex[i, j] == 1 and count < 5:
                print(f"  Kante {i} -> {j}: Fluss = {flow[i, j]:.3f}")
                count += 1
    
    # 5. Vergleich mit Standard-Graphprofil
    print("\n5. VERGLEICH: ROTATION VS. STANDARD-PROFIL")
    print("-" * 80)
    
    D, L, kappa = calculator.compute_profile(adj_complex)
    stats = calculator.get_profile_statistics(D, L, kappa)
    
    print("\nStandard-Graphprofil:")
    print(f"  Kantenmaß κ: {kappa:.3f}")
    print(f"  Durchmesser: {stats['diameter']:.1f}")
    print(f"  Durchschnittlicher kürzester Weg: {stats['avg_shortest_path']:.2f}")
    print(f"  Maximaler längster Weg: {stats['max_longest_path']}")
    
    print("\nRotations-erweiterte Metriken:")
    print(f"  Rotationseffizienz: {props['rotation_efficiency']:.3f}")
    print(f"  → Gibt an, wie gut die Struktur rechtsdrehende Verarbeitung unterstützt")
    
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print()
    print("Das Gehirn verarbeitet Informationen rechtsherum (im Uhrzeigersinn),")
    print("was von oben betrachtet einer negativen Winkelrichtung entspricht.")
    print()
    print("Bei der Ad-hoc Informationsselektion wird der Weg durch das Gehirn")
    print("nach Bedarf gewählt, passend zur aktuellen Synapsenverknüpfung.")
    print()
    print("Die Rotationsfluss-Analyse quantifiziert, wie stark jede Verbindung")
    print("zur rechtsdrehenden Informationsverarbeitung beiträgt.")
    print("=" * 80)


if __name__ == "__main__":
    main()
