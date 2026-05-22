def simuler_microclimat_rue(vehicules_par_heure, vent_m_s, hauteur_bat_m, largeur_rue_m, nb_arbres):
    """
    Simule mathématiquement la qualité de l'air dans une rue.
    """
    print(f"SIMULATION DU SEGMENT DE RUE")
    print(f"Conditions : {vehicules_par_heure} véhicules/h, Vent: {vent_m_s} m/s, Arbres: {nb_arbres}")
    
    # On imagine une émission moyenne de 150 g/km, à 50 km/h
    emission_moyenne_g_s_par_voiture = (150 * 50) / 3600 
    emissions_totales_g_s = (vehicules_par_heure / 3600) * emission_moyenne_g_s_par_voiture
    
    # Plus les bâtiments sont hauts, plus le vent est bloqué
    ratio_canyon = hauteur_bat_m / largeur_rue_m
    vent_effectif = vent_m_s / (1 + ratio_canyon)
    vent_effectif = max(vent_effectif, 0.1) # On évite la division par zéro (vent nul)
    
    # Concentration dans l'air (g/m3)
    concentration_co2 = emissions_totales_g_s / (largeur_rue_m * vent_effectif)
    
    # ABSORPTION PAR LES ARBRES
    absorption_g_s = nb_arbres * 0.0007 # Un arbre absorbe environ 0.0007 g/s
    
    # BILAN FINAL
    bilan_net_g_s = max(0, emissions_totales_g_s - absorption_g_s)
    
    # Plus la rue est encaissée et polluée, plus il fait chaud
    augmentation_temp = (ratio_canyon * 0.5) + (bilan_net_g_s * 10)
    
    print("\nRÉSULTATS MATHÉMATIQUES :")
    print(f"Émissions du trafic   : {emissions_totales_g_s:.4f} g/s")
    print(f"Vitesse vent au sol   : {vent_effectif:.2f} m/s") # freiné par les batiments
    print(f"Absorption végétale   : -{absorption_g_s:.4f} g/s")
    print(f"Bilan Carbone net     : {bilan_net_g_s:.4f} g/s")
    print(f"Augmentation UHI estimée : +{augmentation_temp:.2f} °C")
    
    return bilan_net_g_s

# Faisons un test avec une rue de Hanoi, valeurs choisis arbitrairement
simuler_microclimat_rue(
    vehicules_par_heure=2000, 
    vent_m_s=2.0, 
    hauteur_bat_m=30, # Bâtiments de 30m
    largeur_rue_m=15, # Rue de 15m
    nb_arbres=5       # Peu d'arbres
)