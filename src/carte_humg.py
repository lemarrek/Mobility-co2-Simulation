import json
import geopandas as gpd
import folium
import branca.colormap as cm

print("Chargement des resultats spatialises...")

SCENARIOS = {
    "08h00": {"fichier": "humg_edges_8h.gpkg",  "label": "Pointe matin (PBLH 313 m)"},
    "14h00": {"fichier": "humg_edges_14h.gpkg", "label": "Apres-midi (PBLH 1037 m)"},
    "22h00": {"fichier": "humg_edges_22h.gpkg", "label": "Inversion soir (PBLH 170 m)"},
}

gdfs = {}
vmin_global = float("inf")
vmax_global = float("-inf")

for cle, info in SCENARIOS.items():
    gdf = gpd.read_file(info["fichier"]).to_crs(epsg=4326)
    gdfs[cle] = gdf
    vmin_global = min(vmin_global, gdf["concentration"].min())
    vmax_global = max(vmax_global, gdf["concentration"].max())
    print(f"  {cle} : {len(gdf)} rues chargees")

print(f"  Echelle globale : {vmin_global:.1f} - {vmax_global:.1f} ug/m3")

colormap = cm.LinearColormap(
    colors=['#2ecc71', '#f1c40f', '#e74c3c'],
    vmin=vmin_global,
    vmax=vmax_global,
)

premier_gdf = list(gdfs.values())[0]
centre = premier_gdf.geometry.union_all().centroid
lat_centre, lon_centre = centre.y, centre.x

m = folium.Map(
    location=[lat_centre, lon_centre],
    zoom_start=15,
    tiles='CartoDB positron',
)

groupes_js = {}

for cle, gdf in gdfs.items():
    fg = folium.FeatureGroup(name=cle, show=(cle == "08h00"))

    for _, rue in gdf.iterrows():
        if rue['concentration'] is None or rue.geometry is None:
            continue

        coords = [(pt[1], pt[0]) for pt in rue.geometry.coords]
        couleur = colormap(rue['concentration'])

        nom = rue.get('name', 'Sans nom')
        if isinstance(nom, list):
            nom = nom[0]
        if nom is None:
            nom = "Ruelle sans nom"

        popup_html = f"""
        <b>{nom}</b><br>
        Type : {rue.get('highway', 'inconnu')}<br>
        Ratio H/W : {rue['ratio_hw']:.2f}<br>
        Concentration CO2 : <b>{rue['concentration']:.1f} ug/m3</b><br>
        Delta T (UHI) : +{rue['delta_T']:.2f} C
        """

        folium.PolyLine(
            locations=coords,
            color=couleur,
            weight=5,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fg)

    fg.add_to(m)
    groupes_js[cle] = fg.get_name()

colormap.caption = 'Concentration CO2 estimee (microg/m3)'
colormap.add_to(m)

folium.Marker(
    location=[21.0697, 105.7708],
    popup="Hanoi University of Mining and Geology (HUMG)",
    icon=folium.Icon(color='blue', icon='graduation-cap', prefix='fa'),
).add_to(m)

noms_js = json.dumps(list(SCENARIOS.keys()))
groupes_js_str = json.dumps(groupes_js)
labels_js = json.dumps({k: v["label"] for k, v in SCENARIOS.items()})
map_name = m.get_name()

selector_html = f"""
<div id="scenario-selector" style="
    position: fixed; top: 12px; left: 60px; z-index: 9999;
    background: white; padding: 10px 14px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.25); font-family: sans-serif;
">
  <div style="font-size: 13px; font-weight: 600; margin-bottom: 6px;">
    Scenario horaire
  </div>
  <div id="scenario-buttons" style="display: flex; gap: 6px;"></div>
  <div id="scenario-label" style="font-size: 11px; color: #555; margin-top: 6px;"></div>
</div>

<script>
(function() {{
  var groupesMap = {groupes_js_str};
  var labelsMap = {labels_js};
  var heures = {noms_js};

  function init() {{
    var mapInstance = window["{map_name}"];
    if (!mapInstance) {{
      console.error("Carte non trouvee : {map_name}");
      setTimeout(init, 200);
      return;
    }}

    function switchScenario(heure) {{
      heures.forEach(function(h) {{
        var grp = window[groupesMap[h]];
        if (!grp) {{
          console.error("Groupe non trouve pour", h, groupesMap[h]);
          return;
        }}
        if (h === heure) {{
          if (!mapInstance.hasLayer(grp)) mapInstance.addLayer(grp);
        }} else {{
          if (mapInstance.hasLayer(grp)) mapInstance.removeLayer(grp);
        }}
      }});
      document.getElementById('scenario-label').innerText = labelsMap[heure];
      heures.forEach(function(h) {{
        var btn = document.getElementById('btn-' + h);
        if (btn) {{
          btn.style.background = (h === heure) ? '#378ADD' : '#eee';
          btn.style.color = (h === heure) ? 'white' : '#333';
        }}
      }});
    }}

    var container = document.getElementById('scenario-buttons');
    container.innerHTML = '';
    heures.forEach(function(h) {{
      var btn = document.createElement('button');
      btn.id = 'btn-' + h;
      btn.innerText = h;
      btn.style.cssText = 'padding:5px 10px;border:none;border-radius:5px;cursor:pointer;font-size:12px;background:#eee;';
      btn.onclick = function() {{ switchScenario(h); }};
      container.appendChild(btn);
    }});
    switchScenario('08h00');
  }}

  if (document.readyState === 'complete') {{
    init();
  }} else {{
    window.addEventListener('load', init);
  }}
}})();
</script>
"""

m.get_root().html.add_child(folium.Element(selector_html))

m.save('carte_pollution_humg_3scenarios.html')
print("\nCarte sauvegardee : carte_pollution_humg_3scenarios.html")
print("Ouvre ce fichier dans ton navigateur et utilise les boutons en haut a gauche.")