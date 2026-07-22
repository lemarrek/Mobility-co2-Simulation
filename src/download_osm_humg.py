import osmnx as xn
import osmnx as ox
import geopandas as gpd

# Coordonnées HUMG (Hanoi University of Mining and Geology)
POINT_HUMG = (21.0697, 105.7708)  # 18 Phố Viên, Đức Thắng, Bắc Từ Liêm — HUMG
RAYON_M    = 1500  # rayon d'étude autour de l'université

print("Téléchargement du réseau routier...")
G = ox.graph_from_point(
    POINT_HUMG,
    dist=RAYON_M,
    network_type='drive',
    simplify=True
)
print(f"  → {len(G.nodes)} nœuds, {len(G.edges)} arêtes (rues)")

print("\nTéléchargement des bâtiments...")
tags = {'building': True}
buildings = ox.features_from_point(
    POINT_HUMG,
    tags=tags,
    dist=RAYON_M
)
print(f"  → {len(buildings)} bâtiments trouvés")

# Sauvegarde pour réutilisation
ox.save_graphml(G, 'data/processed/humg_network.graphml')
buildings.to_file('data/processed/humg_buildings.gpkg', driver='GPKG')
print("\nFichiers sauvegardés : humg_network.graphml, humg_buildings.gpkg")

# --- Aperçu de la première rue ---
edges = ox.graph_to_gdfs(G, nodes=False)
print(f"\n=== APERÇU PREMIÈRE RUE ===")
premiere = edges.iloc[0]
print(f"  Nom        : {premiere.get('name', 'sans nom')}")
print(f"  Longueur   : {premiere['length']:.1f} m")
print(f"  Type voie  : {premiere.get('highway', 'inconnu')}")
print(f"  Géométrie  : {premiere['geometry'].geom_type}")

# --- Aperçu des colonnes disponibles pour les bâtiments ---
print(f"\n=== COLONNES BÂTIMENTS DISPONIBLES ===")
print(buildings.columns.tolist())
if 'height' in buildings.columns:
    print(f"\n  Bâtiments avec hauteur OSM renseignée : "
          f"{buildings['height'].notna().sum()} / {len(buildings)}")