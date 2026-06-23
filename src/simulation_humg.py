# =============================================
# SIMULATION MICROCLIMAT URBAIN — HUMG, HANOI
# PBLH dynamique ERA5 + accumulation temporelle
# =============================================
import xarray as xr
import numpy as np

# Constantes sourcées
EMISSION_FACTORS = {
    "moto": 72, "car": 150, "bus": 800, "hgv": 1200,
}
# Source : EEA (2023)
SPEED_KMH       = 30
# Nowak et al. (2013)
TREE_CO2_G_PER_S = 21_000 / (365 * 24 * 3600)

# Paramètres HUMG 
HUMG = {
    "hauteur_bat_m":     12,
    "largeur_rue_m":     14,
    "longueur_seg_m":   200,
    "densite_arbres_km": 30,
}

# Flotte par scénario horaire (heure locale Hanoi)
FLOTTE = {
    8:  {"moto": 2200, "car": 600, "bus": 120, "hgv": 30},
    14: {"moto": 1200, "car": 350, "bus": 80,  "hgv": 20},
    22: {"moto": 800,  "car": 200, "bus": 40,  "hgv": 10},
}

# Mapping heure locale → heure UTC pour ERA5
HEURE_UTC = {8: 1, 14: 7, 22: 15}


def charger_era5(fichier: str) -> dict:
    """Extrait les moyennes PBLH et vent pour chaque scénario horaire."""
    ds  = xr.open_dataset(fichier)
    td  = 'valid_time'
    lat, lon = 21.0, 105.8

    blh = ds['blh'].sel(latitude=lat, longitude=lon, method='nearest')
    u10 = ds['u10'].sel(latitude=lat, longitude=lon, method='nearest')
    v10 = ds['v10'].sel(latitude=lat, longitude=lon, method='nearest')
    ws  = np.sqrt(u10**2 + v10**2)

    era5 = {}
    for h_loc, h_utc in HEURE_UTC.items():
        blh_h  = blh.where(blh[td].dt.hour == h_utc, drop=True)
        ws_h   = ws.where(ws[td].dt.hour == h_utc,   drop=True)
        blh_jan = blh_h.where(blh_h[td].dt.month == 1, drop=True)
        ws_jan  = ws_h.where(ws_h[td].dt.month  == 1, drop=True)
        era5[h_loc] = {
            "pblh_m":   round(float(blh_jan.mean()), 0),
            "vent_m_s": round(float(ws_jan.mean()),  2),
        }
        print(f"  {h_loc:02d}h locale ({h_utc:02d}h UTC) — "
              f"PBLH: {era5[h_loc]['pblh_m']:.0f} m, "
              f"Vent: {era5[h_loc]['vent_m_s']:.2f} m/s")
    ds.close()
    return era5


def taux_retention(pblh_m: float, vent_m_s: float) -> float:
    """
    Calcule le taux de rétention des polluants entre deux pas de temps.

    Rretention augmente quand la PBLH est basse (inversion stable)
    et quand le vent est faible (pas de ventilation).
    Borné entre 0.0 (dispersion totale) et 0.85 (accumulation forte).

    Inspiré de Deng et al. (2023, Atmosphere) — stratification stable
    et cisaillement vertical faible comme conditions d'accumulation.
    """
    # Normalisation : PBLH de référence = 1000 m, vent de référence = 3 m/s
    facteur_pblh = max(0.0, 1.0 - pblh_m / 1000.0)   # élevé si PBLH basse
    facteur_vent = max(0.0, 1.0 - vent_m_s / 3.0)     # élevé si vent faible
    retention    = 0.85 * facteur_pblh * facteur_vent
    return round(min(retention, 0.85), 3)


def simuler_dynamique(era5: dict) -> dict:
    """
    Modèle dynamique avec mémoire temporelle.

    Équation récurrente (Ct) :
        Ct = Qt / (W × Heff × Ueff) + Ct-1 × Rretention

    Où :
        Qt       = émissions totales à l'heure t (g/s)
        W        = largeur de rue (m)
        Heff     = hauteur effective de mélange (m)
        Ueff     = vent effectif au sol (m/s)
        Rretention = taux de rétention fonction de PBLH et vent
    """
    p          = HUMG
    ratio_hw   = p["hauteur_bat_m"] / p["largeur_rue_m"]
    h_canyon   = p["hauteur_bat_m"] * (1 + ratio_hw)
    nb_arbres  = (p["densite_arbres_km"] / 1000) * p["longueur_seg_m"]
    absorption = nb_arbres * TREE_CO2_G_PER_S

    resultats   = {}
    C_precedent = 0.0  # concentration initiale nulle (début de journée)

    for heure in [8, 14, 22]:
        flotte = FLOTTE[heure]
        e5     = era5[heure]

        # Émissions
        total_emit = sum(
            (nb / 3600) * (EMISSION_FACTORS[cat] * SPEED_KMH / 3600)
            for cat, nb in flotte.items()
        )
        emissions_par_cat = {
            cat: (nb / 3600) * (EMISSION_FACTORS[cat] * SPEED_KMH / 3600)
            for cat, nb in flotte.items()
        }

        # Dispersion
        vent_eff  = max(e5["vent_m_s"] / (1 + ratio_hw), 0.1)
        h_vert    = max(h_canyon, min(e5["pblh_m"], 10 * h_canyon))
        C_instant = total_emit / (p["largeur_rue_m"] * h_vert * vent_eff)

        # Accumulation temporelle
        R   = taux_retention(e5["pblh_m"], e5["vent_m_s"])
        C_t = C_instant + C_precedent * R

        # Bilan
        bilan_net   = max(0.0, total_emit - absorption)
        delta_T     = (ratio_hw * 0.5) + (bilan_net * 10)

        resultats[heure] = {
            "pblh_m":         e5["pblh_m"],
            "vent_eff":        round(vent_eff, 2),
            "h_vert_m":        round(h_vert, 1),
            "total_veh":       sum(flotte.values()),
            "total_emit_g_s":  round(total_emit, 5),
            "absorption_g_s":  round(absorption, 5),
            "bilan_net_g_s":   round(bilan_net, 5),
            "retention":       R,
            "C_instant":       round(C_instant * 1e6, 2),
            "C_accumulee":     round(C_t * 1e6, 2),
            "delta_T":         round(delta_T, 2),
            "emissions_cat":   emissions_par_cat,
        }

        C_precedent = C_t  # mémoire pour le prochain pas de temps

    return resultats


# ============================================================
# EXÉCUTION
# ============================================================
print("Chargement ERA5...")
era5 = charger_era5('era5_hanoi_v2.nc')

print("\n" + "=" * 65)
print("SIMULATION DYNAMIQUE — Segment HUMG, Hanoi | Janvier 2023")
print("=" * 65)

res = simuler_dynamique(era5)

for h in [8, 14, 22]:
    r = res[h]
    print(f"\n{'─'*65}")
    print(f"  {h:02d}:00 locale  |  PBLH: {r['pblh_m']:.0f} m  |  "
          f"Rétention: {r['retention']:.2f}")
    print(f"{'─'*65}")
    print(f"  Hauteur mélange   : {r['h_vert_m']:>7} m")
    print(f"  Vent effectif     : {r['vent_eff']:>7} m/s")
    print(f"  Véhicules/heure   : {r['total_veh']:>7}")
    print(f"  Émissions trafic  : {r['total_emit_g_s']:>9.5f} g/s")
    print(f"  Concentration     : {r['C_instant']:>9.2f} µg/m³  (instantanée)")
    print(f"  Concentration     : {r['C_accumulee']:>9.2f} µg/m³  (avec accumulation)")
    print(f"  ΔT UHI estimé     : +{r['delta_T']:>6.2f} °C")

print(f"\n\n{'='*65}")
print("TABLEAU RÉCAPITULATIF")
print(f"{'='*65}")
print(f"  {'Heure':<8} {'PBLH':>6} {'Rét.':>5} {'Émiss.':>8} "
      f"{'C_inst':>8} {'C_accum':>8} {'ΔT':>6}")
print(f"  {'':8} {'(m)':>6} {'':>5} {'(g/s)':>8} "
      f"{'(µg/m³)':>8} {'(µg/m³)':>8} {'(°C)':>6}")
print(f"  {'─'*63}")
for h in [8, 14, 22]:
    r = res[h]
    print(f"  {h:02d}:00   {r['pblh_m']:>6.0f} {r['retention']:>5.2f} "
          f"{r['total_emit_g_s']:>8.5f} {r['C_instant']:>8.2f} "
          f"{r['C_accumulee']:>8.2f} +{r['delta_T']:>5.2f}")