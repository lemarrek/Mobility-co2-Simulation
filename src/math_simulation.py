# Source : EEA (2023), "EMEP/EEA air pollutant emission inventory guidebook"
EMISSION_FACTORS = {
    "moto":  72,
    "car":   150,
    "bus":   800,
    "hgv":   1200,
}

SPEED_KMH = 30  # Vitesse en ville

# Source : Nowak et al. (2013), "Carbon storage and sequestration by trees
#          in urban and community areas of the United States"
#          Moyenne : ~21 kg CO₂ / arbre / an → 0.000 000 665 g/s
TREE_CO2_G_PER_S = 21_000 / (365 * 24 * 3600)  # ≈ 6.65e-7 g/s


def simuler_microclimat_rue(
    flotte: dict,          # {"moto": N, "car": N, "bus": N, "hgv": N}  [veh/h]
    vent_m_s: float,       # Vitesse du vent [m/s]
    hauteur_bat_m: float,  # Hauteur moy des bâtiments [m]
    largeur_rue_m: float,  # Largeur du canyon [m]
    longueur_seg_m: float, # Longueur du segment étudié [m]
    densite_arbres_km: float,  # Arbres par kilomètre linéaire de rue
):

    # ÉMISSIONS PAR CATÉGORIE : Q_traffic = Σ (N_i · E_i)
    # N_i  = débit en veh/s,  E_i = émission en g/s par véhicule
    emissions_par_cat = {}
    total_emit_g_s = 0.0

    for categorie, nb_veh_h in flotte.items():
        ef_g_km = EMISSION_FACTORS[categorie]
        debit_veh_s = nb_veh_h / 3600
        # Conversion g/km → g/s via la vitesse
        emit_g_s = debit_veh_s * (ef_g_km * SPEED_KMH / 3600)
        emissions_par_cat[categorie] = emit_g_s
        total_emit_g_s += emit_g_s

    # Ratio H/W défini dans Oke (1988) "Street design and urban canopy layer climate"
    ratio_hw = hauteur_bat_m / largeur_rue_m
    vent_effectif = max(vent_m_s / (1 + ratio_hw), 0.1)

    # ABSORPTION VÉGÉTALE
    nb_arbres_sur_segment = (densite_arbres_km / 1000) * longueur_seg_m
    absorption_g_s = nb_arbres_sur_segment * TREE_CO2_G_PER_S

    # BILAN
    bilan_net_g_s = max(0.0, total_emit_g_s - absorption_g_s)

    # ESTIMATION UHI
    delta_T = (ratio_hw * 0.5) + (bilan_net_g_s * 10)

    # Rapport
    print("=" * 55)
    print("MICROCLIMAT DE RUE — Bilan CO₂")
    print(f"Segment : {longueur_seg_m} m | Canyon H/W = {ratio_hw:.2f}")
    print(f"Vent ambiant : {vent_m_s} m/s → sol : {vent_effectif:.2f} m/s")
    print(f"Nombre d'arbres sur segment : {nb_arbres_sur_segment:.1f}")
    print("-" * 55)
    print("Émissions par catégorie :")
    for cat, val in emissions_par_cat.items():
        pct = val / total_emit_g_s * 100 if total_emit_g_s > 0 else 0
        print(f"  {cat:<8} {val:.5f} g/s  ({pct:.1f}%)")
    print(f"  TOTAL    {total_emit_g_s:.5f} g/s")
    print(f"Absorption  -{absorption_g_s:.5f} g/s")
    print(f"Bilan net    {bilan_net_g_s:.5f} g/s")
    print(f"ΔT UHI estimé : +{delta_T:.2f} °C")
    print("=" * 55)

    return {
        "emissions_par_cat": emissions_par_cat,
        "total_emit_g_s": total_emit_g_s,
        "absorption_g_s": absorption_g_s,
        "bilan_net_g_s": bilan_net_g_s,
        "delta_T_uhi": delta_T,
    }


# Test : rue de Hanoi (estimation terrain)
simuler_microclimat_rue(
    flotte={
        "moto": 1500,  # dominantes à Hanoi
        "car":  400,
        "bus":  80,
        "hgv":  20,
    },
    vent_m_s=2.0,
    hauteur_bat_m=30,
    largeur_rue_m=15,
    longueur_seg_m=200,       # segment de 200 m
    densite_arbres_km=25,     # ~5 arbres sur 200 m
)