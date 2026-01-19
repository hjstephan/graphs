"""
Experimente zur Graphprofilverteilung im Gehirnmaßstab.
Simuliert 86 * 10^9 Knoten durch skalierbare Sampling-Strategie.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Füge src zum Path hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph_profile import GraphProfileCalculator


class BrainScaleExperiments:
    """
    Experimente zur Graphprofilverteilung im neuronalen Maßstab.
    
    Strategie: Verwende repräsentative Subgraphen verschiedener Größen
    und extrapoliere auf Gehirn-Skala (86 * 10^9 Knoten).
    """
    
    # Konstanten
    BRAIN_NEURONS = 86_000_000_000  # 86 Milliarden Neuronen
    SYNAPSES_PER_NEURON_AVG = 7000  # Durchschnittlich 7000 Synapsen pro Neuron
    
    def __init__(self, output_dir: str = "src/results/brain_scale"):
        self.calculator = GraphProfileCalculator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ergebnisse speichern
        self.results = {
            'metadata': {
                'target_neurons': self.BRAIN_NEURONS,
                'avg_synapses': self.SYNAPSES_PER_NEURON_AVG,
                'timestamp': datetime.now().isoformat()
            },
            'profiles': []
        }
    
    def generate_graph(self, n: int, graph_type: str, **kwargs) -> np.ndarray:
        """
        Generiert verschiedene Graphtypen.
        
        Args:
            n: Anzahl Knoten
            graph_type: Typ des Graphen
            **kwargs: Zusätzliche Parameter
            
        Returns:
            Adjazenzmatrix
        """
        if graph_type == "random_sparse":
            # Erdős-Rényi mit niedriger Dichte (wie neuronale Netze)
            p = kwargs.get('p', self.SYNAPSES_PER_NEURON_AVG / n)
            adj = (np.random.random((n, n)) < p).astype(int)
            np.fill_diagonal(adj, 0)
            return adj
        
        elif graph_type == "scale_free":
            # Barabási-Albert (Power-Law Verteilung, typisch für biologische Netze)
            m = kwargs.get('m', 5)  # Neue Kanten pro Knoten
            adj = self._barabasi_albert(n, m)
            return adj
        
        elif graph_type == "small_world":
            # Watts-Strogatz (hoher Clustering, kurze Pfade)
            k = kwargs.get('k', 6)  # Anzahl Nachbarn
            p = kwargs.get('p', 0.1)  # Rewiring-Wahrscheinlichkeit
            adj = self._watts_strogatz(n, k, p)
            return adj
        
        elif graph_type == "hierarchical":
            # Hierarchisches Netzwerk (wie kortikale Strukturen)
            levels = kwargs.get('levels', 3)
            adj = self._hierarchical_network(n, levels)
            return adj
        
        elif graph_type == "modular":
            # Modulares Netzwerk (Communities wie Hirnregionen)
            num_modules = kwargs.get('num_modules', 10)
            adj = self._modular_network(n, num_modules)
            return adj
        
        elif graph_type == "cortical_column":
            # Simuliert kortikale Säule (layered structure)
            layers = kwargs.get('layers', 6)
            adj = self._cortical_column(n, layers)
            return adj
        
        else:
            raise ValueError(f"Unbekannter Graphtyp: {graph_type}")
    
    def _barabasi_albert(self, n: int, m: int) -> np.ndarray:
        """Barabási-Albert Scale-Free Netzwerk."""
        adj = np.zeros((n, n), dtype=int)
        
        # Start mit vollständigem Graph von m Knoten
        for i in range(m):
            for j in range(i + 1, m):
                adj[i, j] = adj[j, i] = 1
        
        # Füge Knoten hinzu mit preferential attachment
        degrees = np.sum(adj, axis=1)
        
        for new_node in range(m, n):
            # Wähle m Knoten proportional zu ihrem Grad
            # Nur aus existierenden Knoten wählen
            probs = degrees[:new_node] / np.sum(degrees[:new_node])
            targets = np.random.choice(new_node, size=min(m, new_node), 
                                      replace=False, p=probs)
            
            for target in targets:
                adj[new_node, target] = 1
                adj[target, new_node] = 1
                degrees[target] += 1
            
            degrees[new_node] = len(targets)
        
        # Mache gerichtet (wie Synapsen)
        return np.triu(adj, k=1)
    
    def _watts_strogatz(self, n: int, k: int, p: float) -> np.ndarray:
        """Watts-Strogatz Small-World Netzwerk."""
        adj = np.zeros((n, n), dtype=int)
        
        # Erzeuge Ring-Gitter
        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n
                adj[i, neighbor] = 1
        
        # Rewiring mit Wahrscheinlichkeit p
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if np.random.random() < p:
                    neighbor = (i + j) % n
                    # Entferne alte Kante
                    adj[i, neighbor] = 0
                    # Füge neue zufällige Kante hinzu
                    new_neighbor = np.random.randint(0, n)
                    while new_neighbor == i or adj[i, new_neighbor] == 1:
                        new_neighbor = np.random.randint(0, n)
                    adj[i, new_neighbor] = 1
        
        return adj
    
    def _hierarchical_network(self, n: int, levels: int) -> np.ndarray:
        """Hierarchisches Netzwerk mit mehreren Ebenen."""
        adj = np.zeros((n, n), dtype=int)
        
        # Teile Knoten in Ebenen
        nodes_per_level = n // levels
        
        for level in range(levels):
            start = level * nodes_per_level
            end = start + nodes_per_level if level < levels - 1 else n
            
            # Verbindungen innerhalb der Ebene (dicht)
            for i in range(start, end):
                # Verbinde zu ~20% der Knoten in gleicher Ebene
                num_connections = max(1, int(0.2 * (end - start)))
                targets = np.random.choice(range(start, end), 
                                          size=num_connections, replace=False)
                for t in targets:
                    if t != i:
                        adj[i, t] = 1
            
            # Verbindungen zur nächsten Ebene (weniger dicht)
            if level < levels - 1:
                next_start = end
                next_end = next_start + nodes_per_level if level < levels - 2 else n
                for i in range(start, end):
                    # Verbinde zu ~5% der Knoten in nächster Ebene
                    num_connections = max(1, int(0.05 * (next_end - next_start)))
                    targets = np.random.choice(range(next_start, next_end), 
                                              size=num_connections, replace=False)
                    for t in targets:
                        adj[i, t] = 1
        
        return adj
    
    def _modular_network(self, n: int, num_modules: int) -> np.ndarray:
        """Modulares Netzwerk (Communities)."""
        adj = np.zeros((n, n), dtype=int)
        
        nodes_per_module = n // num_modules
        
        for module in range(num_modules):
            start = module * nodes_per_module
            end = start + nodes_per_module if module < num_modules - 1 else n
            
            # Dichte Verbindungen innerhalb des Moduls
            p_internal = 0.3
            for i in range(start, end):
                for j in range(i + 1, end):
                    if np.random.random() < p_internal:
                        adj[i, j] = 1
            
            # Sparse Verbindungen zwischen Modulen
            p_external = 0.01
            for other_module in range(module + 1, num_modules):
                other_start = other_module * nodes_per_module
                other_end = other_start + nodes_per_module if other_module < num_modules - 1 else n
                
                for i in range(start, end):
                    for j in range(other_start, other_end):
                        if np.random.random() < p_external:
                            adj[i, j] = 1
        
        return adj
    
    def _cortical_column(self, n: int, layers: int = 6) -> np.ndarray:
        """Simuliert kortikale Säulenstruktur mit 6 Schichten."""
        adj = np.zeros((n, n), dtype=int)
        
        nodes_per_layer = n // layers
        
        # Typische kortikale Konnektivität:
        # Layer 4 erhält Thalamus-Input
        # Layer 2/3 -> Layer 5
        # Layer 5 -> Output
        # Layer 6 -> zurück zu Thalamus
        
        for layer in range(layers):
            start = layer * nodes_per_layer
            end = start + nodes_per_layer if layer < layers - 1 else n
            
            # Intra-layer Verbindungen
            for i in range(start, end):
                num_conn = np.random.randint(3, 8)
                targets = np.random.choice(range(start, end), size=num_conn, replace=False)
                for t in targets:
                    if t != i:
                        adj[i, t] = 1
            
            # Inter-layer Verbindungen (feed-forward)
            if layer < layers - 1:
                next_start = end
                next_end = next_start + nodes_per_layer if layer < layers - 2 else n
                
                for i in range(start, end):
                    # Jeder Knoten projiziert zu 1-3 Knoten in nächster Schicht
                    num_conn = np.random.randint(1, 4)
                    targets = np.random.choice(range(next_start, next_end), 
                                              size=num_conn, replace=False)
                    for t in targets:
                        adj[i, t] = 1
        
        return adj
    
    def run_scalability_experiment(self):
        """
        Führt Experimente mit zunehmenden Graphgrößen durch.
        Extrapoliert auf Gehirn-Skala.
        """
        print("=" * 80)
        print("SKALIERBARKEITS-EXPERIMENT: Von kleinen zu großen Graphen")
        print("=" * 80)
        
        # Größen von 100 bis ... 200 Knoten (kann erweitert werden)
        sizes = [100, 200]
        graph_types = ["random_sparse", "scale_free", "small_world"]
        
        results = {}
        
        for graph_type in graph_types:
            print(f"\n--- Graphtyp: {graph_type} ---")
            results[graph_type] = []
            
            for n in sizes:
                print(f"  n = {n:5d} Knoten... ", end="", flush=True)
                
                # Generiere Graph
                adj = self.generate_graph(n, graph_type)
                
                # Berechne Profil
                import time
                start = time.perf_counter()
                D, L, kappa = self.calculator.compute_profile(adj)
                elapsed = time.perf_counter() - start
                
                # Statistiken
                stats = self.calculator.get_profile_statistics(D, L, kappa)
                
                result = {
                    'n': int(n),
                    'm': int(np.sum(adj)),
                    'kappa': float(kappa),
                    'diameter': float(stats['diameter']),
                    'avg_shortest_path': float(stats['avg_shortest_path']),
                    'max_longest_path': float(stats['max_longest_path']),
                    'time_seconds': float(elapsed)
                }
                
                results[graph_type].append(result)
                
                print(f"κ={kappa:.3f}, diameter={stats['diameter']:.1f}, "
                      f"Zeit={elapsed:.3f}s")
                
                # Speichere Zwischenergebnisse
                self.results['profiles'].append({
                    'graph_type': graph_type,
                    **result
                })
        
        # Extrapolation auf Gehirn-Skala
        print("\n" + "=" * 80)
        print("EXTRAPOLATION AUF GEHIRN-SKALA (86 Milliarden Knoten)")
        print("=" * 80)
        
        for graph_type in graph_types:
            print(f"\n{graph_type}:")
            
            # Fitte Potenzgesetz für Laufzeit: T(n) = a * n^b
            ns = np.array([r['n'] for r in results[graph_type]])
            times = np.array([r['time_seconds'] for r in results[graph_type]])
            
            # Log-Log-Regression
            log_n = np.log(ns)
            log_t = np.log(times)
            b, log_a = np.polyfit(log_n, log_t, 1)
            
            # Extrapoliere auf 86 Milliarden
            # ACHTUNG: Dies ist theoretisch, nicht praktisch durchführbar!
            time_brain_scale = np.exp(log_a) * (self.BRAIN_NEURONS ** b)
            
            # Konvertiere in verständliche Einheiten
            time_years = time_brain_scale / (365.25 * 24 * 3600)
            
            print(f"  Geschätzte Laufzeit: {time_brain_scale:.2e} Sekunden")
            print(f"                     = {time_years:.2e} Jahre")
            print(f"  (Annahme: O(n^{b:.2f}) Skalierung)")
            
            # Schätze κ und Durchmesser
            kappas = np.array([r['kappa'] for r in results[graph_type]])
            diameters = np.array([r['diameter'] for r in results[graph_type]])
            
            # Für κ: sollte relativ stabil bleiben
            avg_kappa = np.mean(kappas)
            
            # Für Durchmesser: log(n) Wachstum bei Small-World
            if graph_type == "small_world":
                diameter_brain = np.log(self.BRAIN_NEURONS) / np.log(ns[-1]) * diameters[-1]
            else:
                # Konservative Schätzung
                diameter_brain = diameters[-1] * np.sqrt(self.BRAIN_NEURONS / ns[-1])
            
            print(f"  Geschätztes κ: {avg_kappa:.4f}")
            print(f"  Geschätzter Durchmesser: {diameter_brain:.1f}")
        
        return results
    
    def run_distribution_experiment(self, samples_per_type: int = 20):
        """
        Berechnet Graphprofilverteilung über viele Instanzen.
        """
        print("\n" + "=" * 80)
        print(f"VERTEILUNGS-EXPERIMENT: {samples_per_type} Samples pro Typ")
        print("=" * 80)
        
        n = 100  # Kleine Graphen für statistische Analyse
        
        graph_configs = [
            ("random_sparse", {'p': 0.01}),
            ("random_sparse", {'p': 0.05}),
            ("scale_free", {'m': 3}),
            ("scale_free", {'m': 5}),
            ("small_world", {'k': 6, 'p': 0.1}),
            ("small_world", {'k': 10, 'p': 0.3}),
            ("hierarchical", {'levels': 3}),
            ("modular", {'num_modules': 10}),
            ("cortical_column", {'layers': 6}),
        ]
        
        distribution = {}
        
        for graph_type, params in graph_configs:
            config_name = f"{graph_type}_{params}"
            print(f"\n{config_name}:")
            
            kappas = []
            diameters = []
            avg_paths = []
            
            for i in range(samples_per_type):
                print(f"  Sample {i+1}/{samples_per_type}... ", end="", flush=True)
                
                adj = self.generate_graph(n, graph_type, **params)
                D, L, kappa = self.calculator.compute_profile(adj)
                stats = self.calculator.get_profile_statistics(D, L, kappa)
                
                kappas.append(kappa)
                diameters.append(stats['diameter'])
                avg_paths.append(stats['avg_shortest_path'])
                
                print(f"κ={kappa:.3f}")
            
            distribution[config_name] = {
                'kappa': {
                    'mean': float(np.mean(kappas)),
                    'std': float(np.std(kappas)),
                    'min': float(np.min(kappas)),
                    'max': float(np.max(kappas)),
                    'values': [float(k) for k in kappas]
                },
                'diameter': {
                    'mean': float(np.mean(diameters)),
                    'std': float(np.std(diameters)),
                    'values': [float(d) for d in diameters]
                },
                'avg_shortest_path': {
                    'mean': float(np.mean(avg_paths)),
                    'std': float(np.std(avg_paths)),
                    'values': [float(a) for a in avg_paths]
                }
            }
            
            print(f"  → κ: {distribution[config_name]['kappa']['mean']:.3f} "
                  f"± {distribution[config_name]['kappa']['std']:.3f}")
            print(f"  → Durchmesser: {distribution[config_name]['diameter']['mean']:.1f} "
                  f"± {distribution[config_name]['diameter']['std']:.1f}")
        
        # Speichere Verteilung
        with open(self.output_dir / "distribution.json", 'w') as f:
            json.dump(distribution, f, indent=2)
        
        return distribution
    
    def visualize_results(self, distribution: Dict):
        """Erstellt Visualisierungen der Graphprofilverteilung."""
        print("\nErstelle Visualisierungen...")
        
        # 1. κ-Verteilung
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = list(distribution.keys())
        kappa_means = [distribution[c]['kappa']['mean'] for c in configs]
        kappa_stds = [distribution[c]['kappa']['std'] for c in configs]
        
        x = np.arange(len(configs))
        ax.bar(x, kappa_means, yerr=kappa_stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Graph-Konfiguration', fontsize=12)
        ax.set_ylabel('Kantenmaß κ = |V| / |E|', fontsize=12)
        ax.set_title('Graphprofilverteilung: Kantenmaß über verschiedene Graphtypen', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kappa_distribution.png', dpi=300)
        plt.close()
        
        # 2. Durchmesser-Verteilung
        fig, ax = plt.subplots(figsize=(12, 6))
        
        diameter_means = [distribution[c]['diameter']['mean'] for c in configs]
        diameter_stds = [distribution[c]['diameter']['std'] for c in configs]
        
        ax.bar(x, diameter_means, yerr=diameter_stds, capsize=5, alpha=0.7, color='orange')
        ax.set_xlabel('Graph-Konfiguration', fontsize=12)
        ax.set_ylabel('Durchmesser', fontsize=12)
        ax.set_title('Graphprofilverteilung: Durchmesser über verschiedene Graphtypen', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diameter_distribution.png', dpi=300)
        plt.close()
        
        # 3. κ vs. Durchmesser Scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for config in configs:
            kappas = distribution[config]['kappa']['values']
            diameters = distribution[config]['diameter']['values']
            ax.scatter(kappas, diameters, alpha=0.6, label=config, s=50)
        
        ax.set_xlabel('Kantenmaß κ', fontsize=12)
        ax.set_ylabel('Durchmesser', fontsize=12)
        ax.set_title('Graphprofilverteilung: κ vs. Durchmesser', 
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kappa_vs_diameter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualisierungen gespeichert in: {self.output_dir}")
    
    def save_results(self):
        """Speichert alle Ergebnisse als JSON."""
        output_file = self.output_dir / "brain_scale_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nErgebnisse gespeichert: {output_file}")


def main():
    """Hauptfunktion für Experimente."""
    experiments = BrainScaleExperiments()
    
    print("GRAPHPROFIL-EXPERIMENTE IM GEHIRN-MASSSTAB")
    print("=" * 80)
    print(f"Ziel: {experiments.BRAIN_NEURONS:,} Neuronen")
    print(f"      ~{experiments.SYNAPSES_PER_NEURON_AVG:,} Synapsen pro Neuron")
    print(f"      ≈{experiments.BRAIN_NEURONS * experiments.SYNAPSES_PER_NEURON_AVG / 1e15:.1f} Billiarden Synapsen")
    print("=" * 80)
    
    # 1. Skalierbarkeits-Experiment
    scalability_results = experiments.run_scalability_experiment()
    
    # 2. Verteilungs-Experiment
    distribution = experiments.run_distribution_experiment(samples_per_type=20)
    
    # 3. Visualisierungen
    experiments.visualize_results(distribution)
    
    # 4. Speichere Ergebnisse
    experiments.save_results()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTE ABGESCHLOSSEN")
    print("=" * 80)


if __name__ == "__main__":
    main()