import matplotlib.pyplot as plt
import numpy as np

print("Génération du graphique de publication (PDF)...")

# 1. Extraction des données (simulées via ton modèle v4)
heures = ['08:00\n(PBLH: 313m)', '14:00\n(PBLH: 1037m)', '22:00\n(PBLH: 170m)']
trafic = [2950, 1650, 1050]
c_instante = [284.86, 144.07, 125.11]  # Le modèle naïf
c_accumule = [284.86, 144.07, 162.42]  # Le modèle avec inversion nocturne

# 2. Configuration de la figure
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#fdfdfd')
ax1.set_facecolor('#fdfdfd')

# 3. L'Axe principal (Gauche) : Les Concentrations de CO2
color_c1 = '#d62728' # Rouge vif pour le modèle dynamique
color_c2 = '#ff9896' # Rouge pastel pour le modèle naïf

ax1.set_xlabel('Heure locale (Hanoi) - Janvier 2023', fontsize=12, fontweight='bold')
ax1.set_ylabel('Concentration de polluants (µg/m³)', color=color_c1, fontsize=12, fontweight='bold')

# Courbe avec accumulation (Ton résultat majeur)
line1, = ax1.plot(heures, c_accumule, color=color_c1, marker='o', linestyle='-', 
                  linewidth=3, markersize=10, label='Modèle v4 (avec accumulation nocturne)')

# Courbe instantanée (Pour prouver la différence)
line2, = ax1.plot(heures, c_instante, color=color_c2, marker='s', linestyle='--', 
                  linewidth=2, markersize=8, label='Modèle v3 (sans mémoire atmosphérique)')

ax1.tick_params(axis='y', labelcolor=color_c1)
ax1.set_ylim(0, 350)
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

# 4. Le Second Axe (Droite) : Le Trafic Routier
ax2 = ax1.twinx()  
color_t = '#1f77b4' # Bleu classique
ax2.set_ylabel('Volume de trafic (Véhicules/h)', color=color_t, fontsize=12, fontweight='bold')

# On met les barres en transparence (alpha=0.2) derrière les courbes
bar1 = ax2.bar(heures, trafic, color=color_t, alpha=0.2, width=0.4, label='Trafic routier')
ax2.tick_params(axis='y', labelcolor=color_t)
ax2.set_ylim(0, 4000)

# 5. La légende combinée
lines = [line1, line2, bar1]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=10, framealpha=0.9)

# 6. L'Annotation scientifique ("The Money Shot")
# On pointe exactement la différence à 22h00
ax1.annotate('Effet d\'Inversion\n(+30% d\'accumulation)', 
             xy=(2, 162.42), xytext=(1.2, 220),
             arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold', color='darkred',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1.5))

plt.title('Impact de l\'inversion thermique nocturne sur la qualité de l\'air\n(Rue Phố Viên - Hanoi)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# 7. Sauvegarde au format Vectoriel (indispensable pour les articles)
fichier_sortie = 'Figure1_Inversion_Hanoi.pdf'
plt.savefig(fichier_sortie, dpi=300, bbox_inches='tight')
print(f"Graphique généré et sauvegardé sous : {fichier_sortie}")

plt.show()