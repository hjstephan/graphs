"""
Erstellt SVG-Visualisierungen aus den Experimentdaten.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setze SVG als Backend
import matplotlib
matplotlib.use('SVG')

# Lade Daten
data_file = Path("brain_scale/distribution.json")
with open(data_file, 'r') as f:
    distribution = json.load(f)

output_dir = Path("brain_scale")
output_dir.mkdir(parents=True, exist_ok=True)

# Konfigurationen extrahieren
configs = list(distribution.keys())

# Daten extrahieren
kappa_means = [distribution[c]['kappa']['mean'] for c in configs]
kappa_stds = [distribution[c]['kappa']['std'] for c in configs]
diameter_means = [distribution[c]['diameter']['mean'] for c in configs]
diameter_stds = [distribution[c]['diameter']['std'] for c in configs]

# Farben für verschiedene Graphtypen
colors = {
    'random_sparse': '#E74C3C',
    'scale_free': '#3498DB',
    'small_world': '#2ECC71',
    'hierarchical': '#F39C12',
    'modular': '#9B59B6',
    'cortical_column': '#1ABC9C'
}

def get_color(config_name):
    """Bestimme Farbe basierend auf Graphtyp."""
    for key in colors:
        if config_name.startswith(key):
            return colors[key]
    return '#95A5A6'

# Bereite Labels vor (kürze sie für bessere Lesbarkeit)
def shorten_label(label):
    """Kürzt Labels für bessere Lesbarkeit."""
    label = label.replace("random_sparse", "RS")
    label = label.replace("scale_free", "SF")
    label = label.replace("small_world", "SW")
    label = label.replace("hierarchical", "Hier")
    label = label.replace("modular", "Mod")
    label = label.replace("cortical_column", "CC")
    label = label.replace("{'p': ", "(p=")
    label = label.replace("{'m': ", "(m=")
    label = label.replace("{'k': ", "(k=")
    label = label.replace("'p': ", "p=")
    label = label.replace("{'levels': ", "(levels=")
    label = label.replace("{'num_modules': ", "(mod=")
    label = label.replace("{'layers': ", "(layers=")
    label = label.replace("}", ")")
    return label

short_labels = [shorten_label(c) for c in configs]

# ============================================================================
# GRAFIK 1: Kappa vs. Durchmesser (Scatter)
# ============================================================================
print("Erstelle Grafik 1: κ vs. Durchmesser...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plotte jeden Graphtyp
for i, config in enumerate(configs):
    kappas = distribution[config]['kappa']['values']
    diameters = distribution[config]['diameter']['values']
    color = get_color(config)
    label = shorten_label(config)
    
    ax.scatter(kappas, diameters, alpha=0.6, s=80, 
              color=color, label=label, edgecolors='black', linewidth=0.5)

# Formatierung
ax.set_xlabel('Kantenmaß κ = |V| / |E|', fontsize=14, fontweight='bold')
ax.set_ylabel('Durchmesser', fontsize=14, fontweight='bold')
ax.set_title('Graphprofilverteilung: Kantenmaß vs. Durchmesser', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
         frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

# Setze logarithmische Skala für bessere Sichtbarkeit bei großen Wertebereichen
ax.set_xlim(0, max(kappa_means) * 1.2)
ax.set_ylim(0, max(diameter_means) * 1.2)

plt.tight_layout()
plt.savefig(output_dir / 'kappa_vs_diameter.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()
print(f"  → Gespeichert: {output_dir / 'kappa_vs_diameter.svg'}")

# ============================================================================
# GRAFIK 2: Durchmesser über verschiedene Graphtypen (Balkendiagramm)
# ============================================================================
print("Erstelle Grafik 2: Durchmesser-Verteilung...")

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(configs))
bar_colors = [get_color(c) for c in configs]

bars = ax.bar(x, diameter_means, yerr=diameter_stds, capsize=5, 
             alpha=0.8, color=bar_colors, edgecolor='black', linewidth=1.2,
             error_kw={'linewidth': 2, 'ecolor': 'darkred'})

# Werte über Balken anzeigen
for i, (bar, mean, std) in enumerate(zip(bars, diameter_means, diameter_stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
           f'{mean:.1f}',
           ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Graph-Konfiguration', fontsize=14, fontweight='bold')
ax.set_ylabel('Durchmesser', fontsize=14, fontweight='bold')
ax.set_title('Graphprofilverteilung: Durchmesser über verschiedene Graphtypen\n(Mittelwert ± Standardabweichung, n=20 Samples)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Y-Achse bei 0 starten
ax.set_ylim(0, max(diameter_means) * 1.3)

plt.tight_layout()
plt.savefig(output_dir / 'diameter_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()
print(f"  → Gespeichert: {output_dir / 'diameter_distribution.svg'}")

# ============================================================================
# GRAFIK 3: Kappa über verschiedene Graphtypen (Balkendiagramm)
# ============================================================================
print("Erstelle Grafik 3: κ-Verteilung...")

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(configs))

bars = ax.bar(x, kappa_means, yerr=kappa_stds, capsize=5, 
             alpha=0.8, color=bar_colors, edgecolor='black', linewidth=1.2,
             error_kw={'linewidth': 2, 'ecolor': 'darkblue'})

# Werte über Balken anzeigen
for i, (bar, mean, std) in enumerate(zip(bars, kappa_means, kappa_stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
           f'{mean:.3f}',
           ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Graph-Konfiguration', fontsize=14, fontweight='bold')
ax.set_ylabel('Kantenmaß κ = |V| / |E|', fontsize=14, fontweight='bold')
ax.set_title('Graphprofilverteilung: Kantenmaß über verschiedene Graphtypen\n(Mittelwert ± Standardabweichung, n=20 Samples)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Y-Achse bei 0 starten
ax.set_ylim(0, max(kappa_means) * 1.3)

# Markiere biologisch relevanten Bereich (κ ≈ 0.15-0.20)
ax.axhspan(0.15, 0.20, alpha=0.1, color='green', label='Biologischer Bereich')
ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / 'kappa_distribution.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()
print(f"  → Gespeichert: {output_dir / 'kappa_distribution.svg'}")

# ============================================================================
# BONUS: Kombinierte Übersichts-Grafik
# ============================================================================
print("Erstelle Bonus-Grafik: Kombinierte Übersicht...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Links: Kappa
x = np.arange(len(configs))
ax1.bar(x, kappa_means, yerr=kappa_stds, capsize=5, 
       alpha=0.8, color=bar_colors, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Graph-Konfiguration', fontsize=12, fontweight='bold')
ax1.set_ylabel('Kantenmaß κ', fontsize=12, fontweight='bold')
ax1.set_title('(a) Kantenmaß κ = |V| / |E|', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax1.grid(axis='y', alpha=0.3)
ax1.axhspan(0.15, 0.20, alpha=0.1, color='green')
ax1.set_ylim(0, max(kappa_means) * 1.2)

# Rechts: Durchmesser
ax2.bar(x, diameter_means, yerr=diameter_stds, capsize=5, 
       alpha=0.8, color=bar_colors, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Graph-Konfiguration', fontsize=12, fontweight='bold')
ax2.set_ylabel('Durchmesser', fontsize=12, fontweight='bold')
ax2.set_title('(b) Durchmesser', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(diameter_means) * 1.2)

plt.suptitle('Graphprofilverteilung: Vollständige Charakterisierung\n(n=20 Samples pro Konfiguration)', 
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'combined_overview.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()
print(f"  → Gespeichert: {output_dir / 'combined_overview.svg'}")

# ============================================================================
# Statistik-Ausgabe
# ============================================================================
print("\n" + "="*80)
print("STATISTIK-ZUSAMMENFASSUNG")
print("="*80)

print("\nGraphtyp mit niedrigstem κ:")
min_kappa_idx = np.argmin(kappa_means)
print(f"  {short_labels[min_kappa_idx]}: κ = {kappa_means[min_kappa_idx]:.4f} ± {kappa_stds[min_kappa_idx]:.4f}")

print("\nGraphtyp mit höchstem κ:")
max_kappa_idx = np.argmax(kappa_means)
print(f"  {short_labels[max_kappa_idx]}: κ = {kappa_means[max_kappa_idx]:.4f} ± {kappa_stds[max_kappa_idx]:.4f}")

print("\nGraphtyp mit kleinstem Durchmesser:")
min_diam_idx = np.argmin(diameter_means)
print(f"  {short_labels[min_diam_idx]}: Ø = {diameter_means[min_diam_idx]:.1f} ± {diameter_stds[min_diam_idx]:.1f}")

print("\nGraphtyp mit größtem Durchmesser:")
max_diam_idx = np.argmax(diameter_means)
print(f"  {short_labels[max_diam_idx]}: Ø = {diameter_means[max_diam_idx]:.1f} ± {diameter_stds[max_diam_idx]:.1f}")

print("\nBiologisch relevante Graphtypen (κ ∈ [0.15, 0.20]):")
for i, (label, kappa) in enumerate(zip(short_labels, kappa_means)):
    if 0.15 <= kappa <= 0.20:
        print(f"  {label}: κ = {kappa:.4f}, Ø = {diameter_means[i]:.1f}")

print("\nGraphtypen mit perfekter Reproduzierbarkeit (σ_κ = 0):")
for i, (label, std) in enumerate(zip(short_labels, kappa_stds)):
    if std < 1e-10:
        print(f"  {label}: σ_κ = {std:.2e}")

print("\n" + "="*80)
print("ALLE SVG-GRAFIKEN ERFOLGREICH ERSTELLT!")
print("="*80)
