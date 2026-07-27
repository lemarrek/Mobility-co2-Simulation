#import "lib.typ": rapport

// ── Mise en page : éviter les grands blancs ──────────────────
#set figure(placement: none)
#set par(justify: true)

#show: rapport.with(
  nom: "LE MARREC Kelig",
  titre: "Artificial Intelligence and GIS Applied to Urban Mobility Simulation and CO₂ Emissions Modelling",
  année: 4, 
  filière: "INFO",
  période: [April 6, 2026 -- July 27, 2026],
  lang: "en",
  entreprise: (
    nom: "Hanoi University of Mining and Geology (HUMG)",
    adresse: [
      Dong Ngac, Bac Tu Liem, \
      Hanoi, Vietnam
    ],
    logo: image("/images/logo_HUMG.png", height: 4em),
  ),
  responsable: (
    nom: "Prof. Nguyen Gia Trong",
    fonction: "Lecturer-Researcher, Department of Higher Geodesy",
    email: "nguyengiatrong@humg.edu.vn"
  ),
  tuteur: (
    nom: "Eric Gascard",
    email: "eric.gascard@grenoble-inp.fr"
  ),
  référent: (
    nom: "Jean-François Méhaut",
    email: "jean-francois.mehaut@univ-grenoble-alpes.fr"
  ),
  résumé: (
  en: [
    This report presents the work carried out during a 16-week internship (April 6 to
    July 27, 2026) at the Hanoi University of Mining and Geology (HUMG), Vietnam,
    under the supervision of Prof. Nguyen Gia Trong. The objective was to develop from
    scratch a simulation system for traffic-related urban CO₂ emissions, combining
    Artificial Intelligence and Geographic Information Systems (GIS). 
    
    The main
    deliverable is HUCODT (_Hanoi Urban CO₂ Digital Twin_), a fully open-source,
    Python-based urban air quality digital twin integrating four modules: YOLOv8-based
    vehicle detection, linear regression CO₂ prediction, an ERA5-forced atmospheric
    dispersion engine coupled with Oke's (1988) urban street-canyon model, and an
    interactive GIS visualisation layer built with Folium. Applied across 501 road
    segments around the HUMG campus, the system demonstrates that simulated nocturnal
    CO₂ concentrations (19.16 µg/m³) exceed afternoon values (14.87 µg/m³) despite a
    65% traffic reduction, driven by the collapse of the planetary boundary layer height
    (PBLH) to 170 m.
    *Keywords:* GIS, CO₂ emissions, urban traffic, digital twin,
    YOLOv8, ERA5, street canyon, Hanoi.
  ],
  fr: [
    Ce rapport présente le travail effectué lors d'un stage de 16 semaines (du 6 avril
    au 27 juillet 2026) à la Hanoi University of Mining and Geology (HUMG), Vietnam,
    sous la direction du Prof. Nguyen Gia Trong. L'objectif était de développer depuis
    zéro un système de simulation des émissions de CO₂ liées au trafic urbain, en
    croisant Intelligence Artificielle et Systèmes d'Information Géographique (SIG).
    
    Le livrable principal est HUCODT (_Hanoi Urban CO₂ Digital Twin_), un jumeau
    numérique open-source Python intégrant quatre modules : détection de véhicules
    par YOLOv8, prédiction des émissions par régression linéaire, moteur de dispersion
    atmosphérique couplé aux données ERA5 et au modèle de canyon urbain d'Oke (1988),
    et visualisation SIG interactive via Folium. Appliqué sur 501 segments routiers
    autour du campus HUMG, le système démontre que les concentrations nocturnes en
    CO₂ (19,16 µg/m³) dépassent les valeurs de l'après-midi (14,87 µg/m³) malgré
    une réduction de 65 % du trafic, en raison de l'effondrement de la couche limite
    atmosphérique (PBLH) à 170 m. 
    
    *Mots-clés :* SIG, CO₂, jumeau numérique,
    YOLOv8, ERA5, canyon urbain, Hanoï.
  ]
  ),
  glossaire: [
    / HUCODT: Hanoi Urban CO₂ Digital Twin — cadre logiciel développé pendant le stage.
    / GIS: Geographic Information System — Système d'Information Géographique.
    / PBLH: Planetary Boundary Layer Height — hauteur de la couche limite atmosphérique.
    / ERA5: Réanalyse atmosphérique mondiale (ECMWF), accessible via Copernicus.
    / YOLOv8: You Only Look Once v8 — modèle de détection d'objets en temps réel.
    / OSMnx: Bibliothèque Python pour les réseaux routiers OpenStreetMap.
    / GeoPackage: Format géospatial ouvert basé sur SQLite (.gpkg).
    / NetCDF: Network Common Data Form — format scientifique multidimensionnel.
    / H/W: Ratio Hauteur/Largeur du canyon urbain — paramètre clé du modèle d'Oke (1988).
    / UHI: Urban Heat Island — Îlot de Chaleur Urbain.
    / ML: Machine Learning — Apprentissage automatique.
    / venv: Environnement virtuel Python pour isoler les dépendances.
    / COCO: Common Objects in Context — jeu de données de référence pour YOLO.
  ],
)