import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd

# ============================================================
# CONSTANTES DU MODELE (reprises des constantes sourcées)
# ============================================================
EMISSION_FACTORS = {"moto": 72, "car": 150, "bus": 800, "hgv": 1200}
SPEED_KMH        = 30
TREE_CO2_G_PER_S = 21_000 / (365 * 24 * 3600)

DENSITE_ARBRES_KM_DEFAUT = 20
BUFFER_M = 25

ERA5_SCENARIOS = {
    8:  {"pblh_m": 313,  "vent_m_s": 1.84},
    14: {"pblh_m": 1037, "vent_m_s": 2.17},
    22: {"pblh_m": 170,  "vent_m_s": 1.90},
}

LARGEUR_PAR_TYPE = {
    "primary": 20, "secondary": 16, "tertiary": 12,
    "residential": 8, "living_street": 6, "unclassified": 8, "service": 5,
}
LARGEUR_DEFAUT = 10


def estimer_largeur(highway_tag) -> float:
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0]
    return LARGEUR_PAR_TYPE.get(highway_tag, LARGEUR_DEFAUT)


FLOTTE_BASE_PAR_TYPE = {
    "primary":       {"moto": 800, "car": 200, "bus": 40, "hgv": 10},
    "secondary":     {"moto": 600, "car": 150, "bus": 30, "hgv": 8},
    "tertiary":      {"moto": 400, "car": 100, "bus": 15, "hgv": 4},
    "residential":   {"moto": 150, "car": 30,  "bus": 0,  "hgv": 0},
    "living_street": {"moto": 60,  "car": 10,  "bus": 0,  "hgv": 0},
    "unclassified":  {"moto": 150, "car": 30,  "bus": 0,  "hgv": 0},
    "service":       {"moto": 40,  "car": 5,   "bus": 0,  "hgv": 0},
}
FLOTTE_DEFAUT_BASE = {"moto": 100, "car": 20, "bus": 0, "hgv": 0}

FACTEUR_TRAFIC_HORAIRE = {8: 1.00, 14: 0.60, 22: 0.35}


def estimer_flotte(highway_tag, heure) -> dict:
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0]
    base = FLOTTE_BASE_PAR_TYPE.get(highway_tag, FLOTTE_DEFAUT_BASE)
    facteur = FACTEUR_TRAFIC_HORAIRE[heure]
    return {cat: nb * facteur for cat, nb in base.items()}


def hauteur_batiment(row) -> float:
    levels = row.get('building:levels')
    if pd.notna(levels):
        try:
            return float(levels) * 3.0
        except (ValueError, TypeError):
            pass
    return 12.0


def simuler_rue(longueur_m, largeur_m, hauteur_bat_m, densite_arbres_km,
                 flotte, pblh_m, vent_m_s, retention=0.0, c_precedent=0.0):
    if largeur_m <= 0 or longueur_m <= 0:
        return None

    ratio_hw   = hauteur_bat_m / largeur_m
    h_canyon   = hauteur_bat_m * (1 + ratio_hw)
    vent_eff   = max(vent_m_s / (1 + ratio_hw), 0.1)
    h_vert     = max(h_canyon, min(pblh_m, 10 * h_canyon))

    total_emit = sum(
        (nb / 3600) * (EMISSION_FACTORS[cat] * SPEED_KMH / 3600)
        for cat, nb in flotte.items()
    )

    concentration_instant = total_emit / (largeur_m * h_vert * vent_eff)
    concentration = concentration_instant + c_precedent * retention

    nb_arbres  = (densite_arbres_km / 1000) * longueur_m
    absorption = nb_arbres * TREE_CO2_G_PER_S
    bilan_net  = max(0.0, total_emit - absorption)
    delta_T    = (ratio_hw * 0.5) + (bilan_net * 10)

    return {
        "ratio_hw":      round(ratio_hw, 2),
        "h_canyon_m":    round(h_canyon, 1),
        "vent_eff":      round(vent_eff, 2),
        "total_emit":    round(total_emit, 5),
        "concentration": round(concentration * 1e6, 2),
        "delta_T":       round(delta_T, 2),
    }


def taux_retention(pblh_m, vent_m_s) -> float:
    facteur_pblh = max(0.0, 1.0 - pblh_m / 1000.0)
    facteur_vent = max(0.0, 1.0 - vent_m_s / 3.0)
    return round(min(0.85 * facteur_pblh * facteur_vent, 0.85), 3)


print("Chargement du reseau et des batiments...")
G         = ox.load_graphml('data/processed/humg_network.graphml')
buildings = gpd.read_file('data/processed/humg_buildings.gpkg')
edges     = ox.graph_to_gdfs(G, nodes=False)

print(f"  {len(edges)} rues (brutes), {len(buildings)} batiments")

buildings = buildings.to_crs(epsg=3857)
buildings['hauteur_m'] = buildings.apply(hauteur_batiment, axis=1)

edges_proj = edges.to_crs(epsg=3857)
edges_proj = edges_proj.reset_index()
edges_proj['paire_uv'] = edges_proj.apply(
    lambda r: tuple(sorted([r['u'], r['v']])), axis=1
)
edges_proj = edges_proj.drop_duplicates(subset='paire_uv').reset_index(drop=True)

print(f"  Apres deduplication : {len(edges_proj)} rues uniques")

print("Calcul des hauteurs de batiments adjacentes...")
infos_rues = []
for idx, rue in edges_proj.iterrows():
    longueur_m = rue['length']
    highway    = rue.get('highway')
    largeur_m  = estimer_largeur(highway)

    zone    = rue['geometry'].buffer(BUFFER_M)
    voisins = buildings[buildings.geometry.intersects(zone)]
    hauteur_moy = voisins['hauteur_m'].mean() if len(voisins) > 0 else 12.0

    nom = rue.get('name', None)
    if isinstance(nom, list):
        nom = nom[0]
    if nom is None or pd.isna(nom):
        nom = f"Ruelle sans nom (segment {idx})"

    infos_rues.append({
        "idx": idx, "nom_rue": nom,
        "highway": highway if not isinstance(highway, list) else highway[0],
        "longueur_m": longueur_m, "largeur_m": largeur_m,
        "hauteur_bat_m": hauteur_moy, "nb_batiments": len(voisins),
    })

resultats_par_scenario = {}
c_precedent_par_rue = {info["idx"]: 0.0 for info in infos_rues}

for heure in [8, 14, 22]:
    print(f"\nSimulation scenario {heure:02d}h00...")
    era5 = ERA5_SCENARIOS[heure]
    retention = taux_retention(era5["pblh_m"], era5["vent_m_s"])

    lignes = []
    for info in infos_rues:
        flotte = estimer_flotte(info["highway"], heure)
        res = simuler_rue(
            longueur_m=info["longueur_m"], largeur_m=info["largeur_m"],
            hauteur_bat_m=info["hauteur_bat_m"],
            densite_arbres_km=DENSITE_ARBRES_KM_DEFAUT, flotte=flotte,
            pblh_m=era5["pblh_m"], vent_m_s=era5["vent_m_s"],
            retention=retention, c_precedent=c_precedent_par_rue[info["idx"]],
        )
        if res:
            c_precedent_par_rue[info["idx"]] = res["concentration"] / 1e6
            lignes.append({**info, **res, "heure": heure, "pblh_m": era5["pblh_m"]})

    df_h = pd.DataFrame(lignes)
    resultats_par_scenario[heure] = df_h

    print(f"  Concentration moyenne : {df_h['concentration'].mean():.2f} µg/m3")
    print(f"  Concentration max     : {df_h['concentration'].max():.2f} µg/m3 "
          f"({df_h.loc[df_h['concentration'].idxmax(), 'nom_rue']})")

df_complet = pd.concat(resultats_par_scenario.values(), ignore_index=True)
df_complet.to_csv('data/raw/resultats_spatiaux_humg_3scenarios.csv', index=False)
print(f"\nSauvegarde : resultats_spatiaux_humg_3scenarios.csv ({len(df_complet)} lignes)")

for heure, df_h in resultats_par_scenario.items():
    edges_h = edges_proj.copy()
    edges_h = edges_h.merge(
        df_h[["idx", "concentration", "delta_T", "ratio_hw"]],
        left_index=True, right_on="idx", how="inner"
    )
    edges_h.to_file(f'data/processed/humg_edges_{heure}h.gpkg', driver='GPKG')
    print(f"Sauvegarde : humg_edges_{heure}h.gpkg")

print("\nTermine. Pret pour la cartographie multi-scenarios.")