#!/usr/bin/env python3
"""
Experimente zur Gehirn-Informationsverarbeitung mit Rotationsrichtung.

Generiert Experimente und Visualisierungen für das neue Kapitel in graphs.tex:
- Rotationsrichtung (rechtsherum/clockwise) im Gehirn
- Ad-hoc Informationsselektion basierend auf Synapsenverknüpfungen
- Vergleichsstudien verschiedener neuronaler Strukturen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys
import os

# Setze Backend für nicht-interaktive Umgebung
matplotlib.use('Agg')

# Füge src zum Path hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.graph_profile import GraphProfileCalculator
from src.brain_information_processing import BrainInformationProcessor


def create_output_directory():
    """Erstellt Ausgabeverzeichnis für Plots."""
    output_dir = Path(__file__).parent.parent / 'science'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def experiment_1_rotation_direction():
    """
    Experiment 1: Demonstration der Rotationsrichtung im Gehirn.
    
    Zeigt wie das Gehirn Informationen rechtsherum (im Uhrzeigersinn) verarbeitet.
    """
    print("=" * 80)
    print("EXPERIMENT 1: ROTATIONSRICHTUNG IM GEHIRN")
    print("=" * 80)
    
    processor = BrainInformationProcessor()
    calculator = GraphProfileCalculator()
    
    # Verschiedene Netzwerkgrößen testen
    network_sizes = [8, 12, 16, 20, 24]
    rotation_efficiencies = []
    avg_flows = []
    kappas = []
    diameters = []
    
    for n in network_sizes:
        # Erstelle Ring-Netzwerk (simuliert kortikale Regionen)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj[i, (i + 1) % n] = 1  # Ring
            adj[i, (i + 2) % n] = 1  # Überspringende Verbindungen
        
        positions = processor.generate_circular_layout(n)
        props = processor.analyze_rotational_properties(adj, positions)
        
        rotation_efficiencies.append(props['rotation_efficiency'])
        avg_flows.append(props['avg_rotational_flow'])
        kappas.append(props['kappa'])
        diameters.append(props['diameter'])
        
        print(f"n={n:3d}: Rotation-Effizienz={props['rotation_efficiency']:.3f}, "
              f"κ={props['kappa']:.3f}, Durchmesser={props['diameter']:.1f}")
    
    # Erstelle Visualisierung
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Rotationsrichtung im Gehirn: Rechtsherum (Uhrzeigersinn)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Rotationseffizienz vs. Netzwerkgröße
    axes[0, 0].plot(network_sizes, rotation_efficiencies, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Anzahl Neuronen', fontsize=11)
    axes[0, 0].set_ylabel('Rotationseffizienz', fontsize=11)
    axes[0, 0].set_title('Rotationseffizienz (rechtsherum)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Baseline')
    axes[0, 0].legend()
    
    # Plot 2: Durchschnittlicher Rotationsfluss
    axes[0, 1].plot(network_sizes, avg_flows, 's-', color='green', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Anzahl Neuronen', fontsize=11)
    axes[0, 1].set_ylabel('Durchschn. Rotationsfluss', fontsize=11)
    axes[0, 1].set_title('Rotationsfluss-Analyse', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Kantenmaß κ
    axes[1, 0].plot(network_sizes, kappas, '^-', color='purple', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Anzahl Neuronen', fontsize=11)
    axes[1, 0].set_ylabel('Kantenmaß κ = |V|/|E|', fontsize=11)
    axes[1, 0].set_title('Kantenmaß (Strukturelle Effizienz)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Netzwerk-Durchmesser
    axes[1, 1].plot(network_sizes, diameters, 'd-', color='orange', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Anzahl Neuronen', fontsize=11)
    axes[1, 1].set_ylabel('Durchmesser (Hops)', fontsize=11)
    axes[1, 1].set_title('Netzwerk-Durchmesser', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichere als SVG
    output_dir = create_output_directory()
    svg_path = output_dir / 'brain_rotation_direction.svg'
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"\nSVG gespeichert: {svg_path}")
    
    plt.close()
    
    return {
        'rotation_efficiencies': rotation_efficiencies,
        'avg_flows': avg_flows,
        'kappas': kappas,
        'diameters': diameters
    }


def experiment_2_adhoc_path_selection():
    """
    Experiment 2: Ad-hoc Informationsselektion.
    
    Demonstriert wie das Gehirn Pfade basierend auf aktueller Synapsenverknüpfung wählt.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: AD-HOC INFORMATIONSSELEKTION")
    print("=" * 80)
    
    processor = BrainInformationProcessor()
    calculator = GraphProfileCalculator()
    
    # Verschiedene Netzwerk-Komplexitäten
    complexities = []
    adaptivity_scores = []
    path_efficiencies = []
    
    network_configs = [
        ("Feed-forward (sensorisch)", 5, 'feedforward'),
        ("Moderat rekurrent", 6, 'moderate'),
        ("Stark rekurrent", 6, 'recurrent'),
        ("Vollständig vernetzt", 5, 'complete')
    ]
    
    for name, n, config_type in network_configs:
        # Erstelle verschiedene Netzwerk-Typen
        if config_type == 'feedforward':
            adj = np.array([
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ])
        elif config_type == 'moderate':
            adj = np.zeros((n, n), dtype=int)
            for i in range(n-1):
                adj[i, i+1] = 1
            # Einige Rückverbindungen
            adj[n-1, 0] = 1
            adj[n-2, 1] = 1
        elif config_type == 'recurrent':
            adj = np.zeros((n, n), dtype=int)
            for i in range(n):
                adj[i, (i+1) % n] = 1
                adj[i, (i+2) % n] = 1
            # Zusätzliche rekurrente Verbindungen
            for i in range(n//2):
                adj[(i+3) % n, i] = 1
        else:  # complete
            adj = np.ones((n, n), dtype=int)
            np.fill_diagonal(adj, 0)
        
        # Analysiere Informationsfluss
        analysis = calculator.analyze_brain_information_flow(adj, 'forward')
        
        complexities.append(name)
        adaptivity_scores.append(analysis['adaptivity_score'])
        
        # Berechne Pfad-Effizienz
        D = analysis['primary_profile']['D']
        avg_path = np.mean(D[D < np.inf]) if np.any(D < np.inf) else 0
        path_efficiencies.append(1.0 / avg_path if avg_path > 0 else 0)
        
        print(f"{name:30s}: Adaptivität={analysis['adaptivity_score']:.2f}, "
              f"Pfad-Effizienz={path_efficiencies[-1]:.3f}")
    
    # Erstelle Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ad-hoc Informationsselektion: Flexibilität der Pfadwahl', 
                 fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(complexities))
    
    # Plot 1: Adaptivitäts-Score
    bars1 = axes[0].bar(x_pos, adaptivity_scores, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[0].set_xlabel('Netzwerk-Typ', fontsize=11)
    axes[0].set_ylabel('Adaptivitäts-Score', fontsize=11)
    axes[0].set_title('Netzwerk-Adaptivität\n(höher = mehr strukturierte Verarbeitung)', fontsize=12)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(complexities, rotation=15, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    axes[0].legend()
    
    # Werte über den Balken anzeigen
    for i, (bar, val) in enumerate(zip(bars1, adaptivity_scores)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Pfad-Effizienz
    bars2 = axes[1].bar(x_pos, path_efficiencies, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[1].set_xlabel('Netzwerk-Typ', fontsize=11)
    axes[1].set_ylabel('Pfad-Effizienz (1/Ø-Distanz)', fontsize=11)
    axes[1].set_title('Informations-Effizienz\n(höher = kürzere durchschnittliche Pfade)', fontsize=12)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(complexities, rotation=15, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Werte über den Balken anzeigen
    for i, (bar, val) in enumerate(zip(bars2, path_efficiencies)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Speichere als SVG
    output_dir = create_output_directory()
    svg_path = output_dir / 'brain_adhoc_path_selection.svg'
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"\nSVG gespeichert: {svg_path}")
    
    plt.close()
    
    return {
        'complexities': complexities,
        'adaptivity_scores': adaptivity_scores,
        'path_efficiencies': path_efficiencies
    }


def experiment_3_directional_comparison():
    """
    Experiment 3: Vergleich Forward vs. Backward Informationsfluss.
    
    Zeigt den Unterschied zwischen rechtsdrehender (forward) und 
    linksdrehender (backward) Verarbeitung.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: RICHTUNGSVERGLEICH (FORWARD VS. BACKWARD)")
    print("=" * 80)
    
    calculator = GraphProfileCalculator()
    
    # Test verschiedene Netzwerk-Typen
    network_types = []
    forward_efficiencies = []
    backward_efficiencies = []
    directionality_ratios = []
    
    test_networks = [
        ("Feed-forward\n(sensorisch)", np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])),
        ("Reziprok\n(assoziativ)", np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]
        ])),
        ("Ring\n(zyklisch)", np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0]
        ])),
        ("Vollständig\n(stark vernetzt)", np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0]
        ]))
    ]
    
    for name, adj in test_networks:
        result = calculator.compute_bidirectional_profile(adj)
        
        network_types.append(name)
        forward_efficiencies.append(result['combined']['avg_path_forward'])
        backward_efficiencies.append(result['combined']['avg_path_backward'])
        directionality_ratios.append(result['combined']['directionality_ratio'])
        
        print(f"{name.replace(chr(10), ' '):25s}: "
              f"Forward={result['combined']['avg_path_forward']:.2f}, "
              f"Backward={result['combined']['avg_path_backward']:.2f}, "
              f"Ratio={result['combined']['directionality_ratio']:.2f}")
    
    # Erstelle Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Richtungsvergleich: Rechtsherum (Forward) vs. Linksherum (Backward)', 
                 fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(network_types))
    width = 0.35
    
    # Plot 1: Forward vs. Backward Effizienz
    bars1 = axes[0].bar(x_pos - width/2, forward_efficiencies, width, 
                        label='Forward (rechtsherum)', color='#2E86AB')
    bars2 = axes[0].bar(x_pos + width/2, backward_efficiencies, width, 
                        label='Backward (linksherum)', color='#F18F01')
    
    axes[0].set_xlabel('Netzwerk-Typ', fontsize=11)
    axes[0].set_ylabel('Durchschn. Pfadlänge', fontsize=11)
    axes[0].set_title('Vergleich Pfadlängen\n(niedriger = effizienter)', fontsize=12)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(network_types, fontsize=9)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Direktionalitäts-Ratio
    colors = ['#2E86AB' if r > 1.0 else '#F18F01' for r in directionality_ratios]
    bars3 = axes[1].bar(x_pos, directionality_ratios, color=colors)
    axes[1].set_xlabel('Netzwerk-Typ', fontsize=11)
    axes[1].set_ylabel('Direktionalitäts-Ratio', fontsize=11)
    axes[1].set_title('Direktionalitäts-Ratio (Forward/Backward)\n(>1 = Forward-dominant)', fontsize=12)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(network_types, fontsize=9)
    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Neutral (1.0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Werte über den Balken anzeigen
    for bar, val in zip(bars3, directionality_ratios):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Speichere als SVG
    output_dir = create_output_directory()
    svg_path = output_dir / 'brain_directional_comparison.svg'
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"\nSVG gespeichert: {svg_path}")
    
    plt.close()
    
    return {
        'network_types': network_types,
        'forward_efficiencies': forward_efficiencies,
        'backward_efficiencies': backward_efficiencies,
        'directionality_ratios': directionality_ratios
    }


def convert_svg_to_pdf():
    """Konvertiert alle generierten SVG-Dateien zu PDF."""
    print("\n" + "=" * 80)
    print("KONVERTIERUNG: SVG → PDF")
    print("=" * 80)
    
    output_dir = create_output_directory()
    svg_files = list(output_dir.glob('brain_*.svg'))
    
    if not svg_files:
        print("Keine SVG-Dateien gefunden!")
        return
    
    try:
        import subprocess
        
        for svg_path in svg_files:
            pdf_path = svg_path.with_suffix('.pdf')
            
            # Versuche inkscape, dann cairosvg, dann rsvg-convert
            converters = [
                ['inkscape', str(svg_path), '--export-filename', str(pdf_path)],
                ['cairosvg', str(svg_path), '-o', str(pdf_path)],
                ['rsvg-convert', '-f', 'pdf', '-o', str(pdf_path), str(svg_path)]
            ]
            
            converted = False
            for cmd in converters:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"✓ {svg_path.name} → {pdf_path.name}")
                    converted = True
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if not converted:
                print(f"⚠ Konnte {svg_path.name} nicht konvertieren (kein Konverter verfügbar)")
                print(f"  SVG-Datei verfügbar unter: {svg_path}")
        
    except Exception as e:
        print(f"Fehler bei Konvertierung: {e}")
        print("SVG-Dateien wurden erstellt, können aber manuell konvertiert werden.")


def main():
    """Hauptfunktion: Führt alle Experimente aus."""
    print("\n" + "=" * 80)
    print("  GEHIRN-ROTATIONS-EXPERIMENTE")
    print("  Für: science/graphs.tex (Neues Kapitel)")
    print("=" * 80)
    print()
    
    try:
        # Führe Experimente durch
        results1 = experiment_1_rotation_direction()
        results2 = experiment_2_adhoc_path_selection()
        results3 = experiment_3_directional_comparison()
        
        # Konvertiere SVG zu PDF
        convert_svg_to_pdf()
        
        print("\n" + "=" * 80)
        print("  ALLE EXPERIMENTE ERFOLGREICH ABGESCHLOSSEN")
        print("=" * 80)
        print()
        print("Generierte Dateien im Verzeichnis 'science/':")
        print("  1. brain_rotation_direction.svg (+ .pdf)")
        print("  2. brain_adhoc_path_selection.svg (+ .pdf)")
        print("  3. brain_directional_comparison.svg (+ .pdf)")
        print()
        print("Diese Dateien können nun in graphs.tex eingebunden werden.")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nFehler bei Ausführung der Experimente: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
