#!/usr/bin/env python
"""
Demonstration der Gehirn-Informationsverarbeitung mit Drehrichtung.

Das Gehirn hat von oben betrachtet eine Drehrichtung zur allgemeinen
Verarbeitung von Informationen. Die Drehrichtung ist bei Menschen rechtsherum.
Also von oben in negativer Winkelrichtung. Bei der ad-hoc Informationsselektion
wird nach Bedarf der entsprechende Weg durch das Gehirn gewählt, wie es zur
aktuellen Synapsenverknüpfung passt.
"""

import numpy as np
import sys
import os

# Füge src zum Path hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.graph_profile import GraphProfileCalculator


def print_section(title):
    """Hilfsfunktion für formatierte Ausgabe."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_basic_directionality():
    """Demonstration der grundlegenden Drehrichtung."""
    print_section("1. GRUNDLEGENDE DREHRICHTUNG")
    
    calculator = GraphProfileCalculator()
    
    # Einfaches Feed-forward Netzwerk (wie sensorische Verarbeitung)
    print("\nFeed-forward Netzwerk (sensorische Verarbeitung):")
    print("  Input -> Hidden -> Output")
    
    adj_feedforward = np.array([
        [0, 1, 1, 0, 0],  # Input-Schicht (2 Neuronen)
        [0, 0, 0, 1, 1],  # Input -> Hidden
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],  # Output-Schicht (2 Neuronen)
        [0, 0, 0, 0, 0]
    ])
    
    result = calculator.compute_bidirectional_profile(adj_feedforward)
    
    print(f"\n  Vorwärts (clockwise/rechtsherum):")
    print(f"    - Durchschnittliche Weglänge: {result['combined']['avg_path_forward']:.2f}")
    print(f"    - Durchmesser: {result['combined']['diameter_forward']:.0f}")
    
    print(f"\n  Rückwärts (counter-clockwise/linksherum):")
    print(f"    - Durchschnittliche Weglänge: {result['combined']['avg_path_backward']:.2f}")
    print(f"    - Durchmesser: {result['combined']['diameter_backward']:.0f}")
    
    print(f"\n  Direktionalitäts-Ratio: {result['combined']['directionality_ratio']:.2f}")
    print(f"    (> 1.0 bedeutet: Vorwärts-dominant)")


def demo_cortical_layers():
    """Demonstration mit kortikaler Schichtenstruktur."""
    print_section("2. KORTIKALE SCHICHTENSTRUKTUR (6 LAYER)")
    
    calculator = GraphProfileCalculator()
    
    # Vereinfachte kortikale Säule mit 6 Schichten
    # Jede Schicht hat 4 Neuronen
    n_per_layer = 4
    n_layers = 6
    n_total = n_per_layer * n_layers
    
    adj = np.zeros((n_total, n_total), dtype=int)
    
    print("\nErstelle kortikale Säulenstruktur:")
    print(f"  - 6 Schichten (Layer 1-6)")
    print(f"  - {n_per_layer} Neuronen pro Schicht")
    print(f"  - Gesamt: {n_total} Neuronen")
    
    # Typische kortikale Verbindungen
    # Layer 4 erhält Input (von Thalamus, hier simuliert durch Layer 1)
    # Layer 2/3 -> Layer 5
    # Layer 5 -> Output
    # Layer 6 -> zurück zu Layer 4 (Feedback)
    
    for layer in range(n_layers - 1):
        start = layer * n_per_layer
        end = start + n_per_layer
        next_start = end
        next_end = next_start + n_per_layer
        
        # Feed-forward Verbindungen
        for i in range(start, end):
            # Jeder Knoten verbindet zu 2 Neuronen in nächster Schicht
            if next_start < n_total:
                num_targets = min(2, n_total - next_start)
                if num_targets > 0:
                    targets = np.random.choice(range(next_start, min(next_end, n_total)), 
                                              size=num_targets, replace=False)
                    for t in targets:
                        adj[i, t] = 1
    
    # Feedback von Layer 6 zu Layer 4
    layer_6_start = 5 * n_per_layer
    layer_4_start = 3 * n_per_layer
    layer_4_end = layer_4_start + n_per_layer
    for i in range(layer_6_start, layer_6_start + n_per_layer):
        feedback_target = np.random.choice(range(layer_4_start, layer_4_end))
        adj[i, feedback_target] = 1
    
    print("\nAnalysiere Informationsfluss...")
    analysis = calculator.analyze_brain_information_flow(adj, 'forward')
    
    print(f"\n  Primäre Richtung: {analysis['primary_direction']}")
    print(f"\n  Primärprofil (Forward):")
    print(f"    - Durchschnittliche Weglänge: {analysis['primary_profile']['avg_path_length']:.2f}")
    print(f"    - Durchmesser: {analysis['primary_profile']['diameter']:.0f}")
    print(f"    - Kantenmaß κ: {analysis['primary_profile']['kappa']:.4f}")
    
    print(f"\n  Sekundärprofil (Feedback):")
    print(f"    - Durchschnittliche Weglänge: {analysis['secondary_profile']['avg_path_length']:.2f}")
    print(f"    - Durchmesser: {analysis['secondary_profile']['diameter']:.0f}")
    
    print(f"\n  Effizienz:")
    print(f"    - Forward: {analysis['information_flow_efficiency']['primary']:.4f}")
    print(f"    - Feedback: {analysis['information_flow_efficiency']['secondary']:.4f}")
    
    print(f"\n  Adaptivitäts-Score: {analysis['adaptivity_score']:.2f}")
    
    print(f"\n  Interpretation:")
    # Formatiere Interpretation mit Einrückung
    interpretation = analysis['interpretation']
    for line in interpretation.split('. '):
        if line:
            print(f"    {line.strip()}.")


def demo_adhoc_path_selection():
    """Demonstration der ad-hoc Informationsselektion."""
    print_section("3. AD-HOC INFORMATIONSSELEKTION")
    
    calculator = GraphProfileCalculator()
    
    print("\nKomplexes Netzwerk mit mehreren Pfaden:")
    print("  - Mehrere alternative Routen zwischen Knoten")
    print("  - Simuliert flexible Pfadwahl im Gehirn")
    
    # Netzwerk mit redundanten Pfaden
    adj = np.array([
        [0, 1, 1, 0, 0, 0],  # Knoten 0: 2 Ausgänge
        [0, 0, 0, 1, 1, 0],  # Knoten 1: 2 Ausgänge
        [0, 0, 0, 1, 1, 0],  # Knoten 2: 2 Ausgänge
        [0, 0, 0, 0, 0, 1],  # Knoten 3: 1 Ausgang
        [0, 0, 0, 0, 0, 1],  # Knoten 4: 1 Ausgang
        [0, 0, 0, 0, 0, 0]   # Knoten 5: Ziel
    ])
    
    # Berechne beide Richtungen
    D_fwd, L_fwd, _ = calculator.compute_profile_with_direction(adj, 'forward')
    D_bwd, L_bwd, _ = calculator.compute_profile_with_direction(adj, 'backward')
    
    print("\n  Von Knoten 0 zu Knoten 5:")
    print(f"    - Kürzester Weg (forward): {D_fwd[0, 5]:.0f} Schritte")
    print(f"    - Längster Weg (forward): {L_fwd[0, 5]:.0f} Schritte")
    print(f"    - Anzahl möglicher Pfade: {int(L_fwd[0, 5] - D_fwd[0, 5]) + 1}")
    
    print("\n  Bei ad-hoc Selektion:")
    print("    1. Direkte Route (kürzester Weg): 0 -> 1/2 -> 3/4 -> 5")
    print("    2. Alternative Routen nutzen andere Synapsen")
    print("    3. Gehirn wählt basierend auf aktueller Aktivierung")
    
    # Analyse der Flexibilität
    analysis = calculator.analyze_brain_information_flow(adj, 'forward')
    
    print(f"\n  Netzwerk-Adaptivität: {analysis['adaptivity_score']:.2f}")
    if analysis['adaptivity_score'] < 1.5:
        print("    → Hohes Maß an Flexibilität für alternative Pfade")
    else:
        print("    → Strukturierte, gerichtete Verarbeitung")


def demo_comparison_scenarios():
    """Vergleich verschiedener Gehirn-Szenarien."""
    print_section("4. VERGLEICH: VERSCHIEDENE VERARBEITUNGSTYPEN")
    
    calculator = GraphProfileCalculator()
    
    scenarios = [
        ("Sensorischer Kortex (Feed-forward)", np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])),
        ("Präfrontaler Kortex (Rekurrent)", np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]
        ])),
        ("Hippocampus (Stark vernetzt)", np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0]
        ]))
    ]
    
    print("\nVergleich der Verarbeitungscharakteristiken:\n")
    print(f"{'Region':<35} {'Adaptivität':<15} {'Interpretation'}")
    print("-" * 80)
    
    for name, adj in scenarios:
        analysis = calculator.analyze_brain_information_flow(adj, 'forward')
        adaptivity = analysis['adaptivity_score']
        
        # Kurze Interpretation
        if adaptivity > 1.5:
            interp = "Feed-forward dominant"
        elif adaptivity > 0.7:
            interp = "Balanciert"
        else:
            interp = "Rekurrent dominant"
        
        print(f"{name:<35} {adaptivity:>6.2f}         {interp}")


def main():
    """Hauptfunktion."""
    print("\n" + "=" * 80)
    print("  GEHIRN-INFORMATIONSVERARBEITUNG MIT DREHRICHTUNG")
    print("  Demonstration der Implementierung")
    print("=" * 80)
    print("\nDas Gehirn hat von oben betrachtet eine Drehrichtung zur allgemeinen")
    print("Verarbeitung von Informationen. Die Drehrichtung ist bei Menschen")
    print("rechtsherum (negative Winkelrichtung = clockwise).")
    print("\nBei der ad-hoc Informationsselektion wird nach Bedarf der entsprechende")
    print("Weg durch das Gehirn gewählt, wie es zur aktuellen Synapsenverknüpfung passt.")
    
    try:
        demo_basic_directionality()
        demo_cortical_layers()
        demo_adhoc_path_selection()
        demo_comparison_scenarios()
        
        print_section("ZUSAMMENFASSUNG")
        print("\nDie Implementierung ermöglicht:")
        print("  ✓ Analyse der Informationsfluss-Richtung (clockwise/counter-clockwise)")
        print("  ✓ Berechnung bidirektionaler Profile (Forward/Feedback)")
        print("  ✓ Charakterisierung der Netzwerk-Adaptivität")
        print("  ✓ Simulation verschiedener kortikaler Strukturen")
        print("  ✓ Quantifizierung der ad-hoc Pfadselektions-Flexibilität")
        
        print("\nAnwendungen:")
        print("  • Neurowissenschaftliche Forschung")
        print("  • KI-Modell-Architektur-Analyse")
        print("  • Gehirn-Computer-Interface-Design")
        print("  • Neuromorphe Computing-Optimierung")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\nFehler bei Ausführung: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
