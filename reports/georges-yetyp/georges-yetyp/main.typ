#import "@preview/georges-yetyp:0.2.0": rapport

// ── Mise en page : éviter les grands blancs ──────────────────
#set figure(placement: none)   // figures placées là où elles sont déclarées
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
    nom: "Dr. Nguyen Gia Trong",
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
    under the supervision of Dr. Nguyen Gia Trong. The objective was to develop from
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
    sous la direction du Dr. Nguyen Gia Trong. L'objectif était de développer depuis
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
  bibliographie: bibliography("refs.bib", style: "ieee"),
  annexes-extra: [
    == Appendix 1 — Gantt Chart

    #figure(
      image("/images/Appendix1.png", width: 100%),
      caption: [Full Gantt chart — 11 tasks over 16 weeks.
                Source: Diagramme_de_Gantt.pdf.]
    )

    == Appendix 2 — HUCODT Software Architecture

    ```
    Mobility_Co2_Simulation/
    ├── src/
    │   ├── co2_predictor.py              # ML: CO₂ prediction
    │   ├── vision_test.py                # CV: YOLOv8 detection
    │   ├── digital_twin_pipeline.py      # Orchestrator: ML + CV
    │   ├── download_era5_hanoi.py        # ERA5 via Copernicus API
    │   ├── download_osm_humg.py          # OSM road network download
    │   ├── math_simulation.py            # Analytical microclimate model
    │   ├── simulation_humg.py            # 0D dynamic engine (Box Model)
    │   ├── simulation_spatiale_humg.py   # 2D spatial engine, 501 segments
    │   ├── carte_humg.py                 # Folium interactive HTML map
    ├── data/
    │   ├── raw/
    │   │   ├── CO2 Emissions_Canada.csv  # ML training dataset
    │   │   └── resultats_spatiaux_humg_3scenarios.csv
    │   └── processed/
    │       ├── era5_hanoi_v2.nc          # ERA5 NetCDF (PBLH + wind)
    │       ├── humg_buildings.gpkg       # Building footprints
    │       ├── humg_network.graphml      # Road network graph
    │       ├── humg_edges_8h.gpkg        # Output 08:00
    │       ├── humg_edges_14h.gpkg       # Output 14:00
    │       └── humg_edges_22h.gpkg       # Output 22:00
    ├── notebooks/
    │   ├── 01_test.ipynb                 # ML validation
    │   └── 02_map_simulation.ipynb       # GIS exploration & routing
    ├── images/                           # Figures and logos
    ├── docs/                             # Reference papers (PDF)
    ├── carte_pollution_humg_3scenarios.html
    ├── yolov8n.pt                        # YOLOv8 model weights
    └── requirements.txt
    ```

    == Appendix 3 — Main Simulation Equation

    Dynamic accumulation equation @peng2023, for each segment $s$ at time step $t$:

    $ C(s,t) = Q(s,t) - A(s) + C(s,t-1) times R(s,t) $

    - $C(s,t)$: CO₂ concentration (µg/m³)
    - $Q(s,t)$: emission flux = fleet × EEA factor @eea2023 × time-of-day factor
    - $A(s)$: vegetation absorption @nowak2013 (21 kg CO₂/tree/year)
    - $R(s,t) = op("clip")(1 - u_"eff" / H_"mix", 0, 0.85)$: dynamic retention rate
    - $u_"eff" = U_(10) times f(H/W)$: effective canyon wind speed @oke1988
    - $H_"mix" = min("PBLH", h_"bld" times (1 + 1 / (H/W)))$: effective mixing height

    == Appendix 4 — Technologies and Libraries Used

    #figure(
      table(
        columns: (1.1fr, 1.4fr, 2fr),
        align: left,
        fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
        table.header(
          text(fill: white, weight: "bold")[Domain],
          text(fill: white, weight: "bold")[Tool / Library],
          text(fill: white, weight: "bold")[Use],
        ),
        [Machine Learning],     [scikit-learn],
          [Linear regression CO₂ prediction],
        [Computer Vision],      [YOLOv8n (Ultralytics)],
          [Vehicle detection and classification],
        [Spatial analysis],     [OSMnx, GeoPandas, networkx],
          [OSM road network, spatial joins, routing],
        [Climatological data],  [xarray, cdsapi],
          [ERA5 NetCDF, Copernicus REST API],
        [Visualisation],        [matplotlib, Folium / Leaflet.js],
          [Scientific plots, interactive HTML map],
        [Environment],          [Python 3.12, venv, VS Code, Linux],
          [Development and deployment],
        [Open Data],            [OpenStreetMap, CO2 Emissions Canada, ERA5],
          [All primary source datasets],
        [Version control],      [Git / GitHub (github.com/lemarrek)],
          [Mobility-co2-Simulation repository],
      ),
      caption: [Full technology stack of the HUCODT project.]
    )
  ]
)

// ════════════════════════════════════════════════════════════
// I. CONTEXT
// ════════════════════════════════════════════════════════════

= Internship Context and Environment

== Host Institution

The internship took place at the *Hanoi University of Mining and Geology (HUMG)*,
one of Vietnam's most prestigious technical universities, specialising in Earth sciences,
mining, geology, and geodesy-cartography. The university maintains numerous
international research collaborations for scientific and technology transfer.

The internship supervisor, *Dr. Nguyen Gia Trong*, is a lecturer-researcher in the
Department of Higher Geodesy (Faculty of Geodesy, Cartography and Land Management).
A specialist in geodetic data processing and satellite positioning (GNSS/GPS), he now
focuses his research on integrating AI and GIS for environmental management, smart
cities, and natural hazard assessment. Academic supervision on the Polytech Grenoble
side was provided by Eric Gascard (academic supervisor) and Jean-François Méhaut
(internship coordinator).

== Scientific Context

Road transport accounts for approximately 23% of global energy-related CO₂ emissions
@iea2023. In Hanoi — more than 8 million inhabitants, motorbike fleet exceeding 6 to
8 million units — the morphological configuration of the city creates *urban canyon*
conditions that trap vehicle emissions and amplify their thermal and chemical effects
@oke1988. Deploying physical sensor networks remains economically prohibitive: a dense
network covering a single district can cost several hundred thousand dollars. This
*structural data gap* constrains both scientific understanding and public policy.

== Installation Conditions

The first week was devoted to initial contact and setup. An early difficulty arose: the
supervisor's email address was not functional, which required switching to WhatsApp.
A scoping meeting then allowed objectives and required skills to be precisely defined.

// ════════════════════════════════════════════════════════════
// II. PROJECT DESCRIPTION
// ════════════════════════════════════════════════════════════

= Proposed Project: Synthetic Description

The internship objective was to build from scratch a *Python software system* capable
of simulating traffic-related CO₂ emissions in an urban environment, combining:

- *Machine Learning* for per-vehicle CO₂ emission prediction;
- *Computer Vision* for automatic vehicle detection in road traffic images;
- *GIS spatial analysis libraries* for mapping and spatialisation of results;
- *open meteorological data* (ERA5) for atmospheric dispersion modelling.

The hard constraint was to rely exclusively on *open data*, compensating for the
impossibility of deploying physical sensors. Upon arrival, the team had a 16-week work
plan but *no pre-existing codebase*. The software architecture, AI model choices, and
GIS library selection were therefore designed entirely during the internship.

// ════════════════════════════════════════════════════════════
// III. WORK CARRIED OUT
// ════════════════════════════════════════════════════════════

= Work Effectively Carried Out

The work developed incrementally across four major phases: from individual building
blocks to an integrated system, then from a single-point simulation (0D) to a full
spatial simulation (2D on the road network), and finally the scientific restitution.

== Phase 1 — Weeks 1–4: Technical Foundation

=== Development Environment

A professional development environment was set up on Linux with VS Code: GitHub
repository (_Mobility-co2-Simulation_), strict Python virtual environment (venv) for
dependency isolation, and modular project structure.

=== CO₂ Prediction Model (co2_predictor.py)

The first module is a Machine Learning model trained on the public _CO2 Emissions
Canada_ dataset (7,385 records). Implemented as a *CO2Predictor* class, it exposes
a *predict_vehicle()* method returning CO₂ in g/km given a vehicle's displacement,
cylinder count, and fuel consumption. *Linear regression* (scikit-learn) was chosen
for its mathematically continuous relationship with engine parameters. Performance on
the 20% test set: *R² = 0.876*, *RMSE = 18.4 g/km* @mokhtarzadeh2021.

#figure(
  image("/images/Figure1.png", width: 88%),
  caption: [CO₂ Predictor: predicted vs actual emissions (g/km) on the test set.
            R² = 0.876, RMSE = 18.4 g/km. Source: 01_test.ipynb.]
)

=== Vehicle Detection (vision_test.py)

A Computer Vision module using *YOLOv8n* detects and classifies vehicles in road
traffic images. The lightweight yolov8n.pt model ensures fast CPU inference. On a
1920×1080 px test image, it detected 45 cars and 2 trucks in under 200 ms with
confidence scores above 0.50.

#figure(
  image("/images/Figure2.jpg", width: 88%),
  caption: [YOLOv8n detection on urban road traffic. Bounding boxes by class
            (car / truck) with confidence scores. Source: vision_test.py.]
)

=== Initial Digital Twin Pipeline (digital_twin_pipeline.py)

The two preceding modules were fused into a coherent execution pipeline: (1) YOLO
counts vehicles by class; (2) a mean technical profile is assigned to each class;
(3) the CO2Predictor is called for each occurrence; (4) results are aggregated.
This pipeline estimated *10,246.65 g/km of CO₂* for the test scene (45 cars, 2 trucks,
1 bus).

#figure(
  image("/images/Figure3.png", width: 88%),
  caption: [Console output of digital_twin_pipeline.py: vehicle breakdown and
            total emission estimate (10,246.65 g/km CO₂).]
)

=== Cartography and GIS (02_map_simulation.ipynb)

A GIS module using OSMnx and networkx downloads the road network topology from
OpenStreetMap (2,474 intersections for the Hanoi area) and computes the shortest
path between two GPS coordinates — HUMG campus to Hoa Binh Park: *3.06 km*.
Combined with the per-kilometre emission rate, this enables full journey carbon
footprint simulation on the Vietnamese urban network.

== Phase 2 — Weeks 5–10: Dynamic Modelling and Atmospheric Integration

=== A Major Scientific Bottleneck

From week 7, the supervisor highlighted a critical flaw: the model was "static",
assuming pollution decreased proportionally with traffic. In South-East Asian
metropolises, nocturnal temperature drops cause a collapse of the *Planetary Boundary
Layer* (PBLH), trapping residual emissions and creating dangerous nocturnal
accumulation even when traffic is low @deng2023. The new research question:
_how to algorithmically simulate this accumulation by crossing traffic data with
atmospheric boundary layer dynamics?_

=== Literature Review

Key publications studied:
- *Deng et al. (2023)* @deng2023: ERA5 data for nocturnal thermal cap modelling;
- *Peng et al. (2023)* @peng2023: coupling urban canyon geometry with PBLH dynamics;
- *Oke (1988)* @oke1988: wind behaviour in urban canyons via the H/W ratio;
- *Nowak et al. (2013)* @nowak2013: vegetation absorption (≈ 21 kg CO₂/tree/year);
- *EEA (2023)* @eea2023: reference emission factors in g/km by vehicle category.

=== ERA5 Acquisition Module (download_era5_hanoi.py)

An automation script downloads ERA5 data via the Copernicus REST API (cdsapi),
targeting a bounding box over Hanoi. It extracts NetCDF files with 3D tensors
(Latitude × Longitude × Time) for PBLH and 10 m wind vectors. Timezone mapping
(ERA5 UTC → Hanoi UTC+7) was implemented rigorously via xarray @hersbach2020.

=== Dynamic Simulation Engine (simulation_humg.py)

This module is the *scientific core*: a recursive algorithm with temporal memory,
implementing the accumulation equation from Peng et al. (2023):

$ C_t = Q_t + (C_(t-1) times R), quad R = op("clip")(1 - u_"eff" / H_"mix", 0, 0.85) $

$Q_t$ = instantaneous emissions; $R$ = dynamic retention rate; $u_"eff"$ = effective
canyon wind speed (Oke 1988); $H_"mix"$ = effective mixing height derived from PBLH
and building geometry. When PBLH is low and wind is weak, $R → 0.85$ (strong
accumulation); when PBLH is high and wind is strong, $R → 0$ (maximum dispersion).

The scientific visualisation module confirmed the hypothesis
visually: *+30% pollution at 22:00* driven by thermal inversion, despite 60% less
traffic.

#figure(
  image("/images/Figure4.png", width: 88%),
  caption: [Simulated diurnal CO₂ profile: S1 (08:00, PBLH = 313 m), S2 (14:00,
            PBLH = 1,037 m), S3 (22:00, PBLH = 170 m). Despite 65% less
            traffic at night, PBLH collapse causes a +30% increase.]
)

== Phase 3 — Weeks 11–13: Full Spatialisation and Final Deliverable

=== Spatial Simulation Module (simulation_spatiale_humg.py)

The 0D engine was extended into a *fully operational 2D model* covering *501 road
segments* within 1.5 km of the HUMG campus. The OSMnx road graph and building
footprints (from GeoPackage files) are loaded; the dynamic model is then applied in
three successive passes (08:00 → 14:00 → 22:00), propagating concentration memory.

Road width is inferred from OSM road type; building height is computed from
building:levels (×3 m/floor, default 12 m for the ~60% of buildings lacking this
attribute). A 25 m buffer identifies adjacent buildings via spatial intersection.
Fleet composition (motorbikes, cars, buses, heavy goods vehicles) is differentiated by
road category and modulated by time-of-day factors (1.0 at 08:00, 0.6 at 14:00,
0.35 at 22:00). Outputs: three GeoPackage files and a consolidated CSV.

#figure(
  table(
    columns: (1.6fr, 0.8fr, 0.9fr, 1.4fr),
    align: (left, center, center, center),
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Scenario],
      text(fill: white, weight: "bold")[Time],
      text(fill: white, weight: "bold")[PBLH (m)],
      text(fill: white, weight: "bold")[Mean CO₂ (µg/m³)],
    ),
    [S1 — Morning peak],   [08:00], [313],   [*29.95*],
    [S2 — Afternoon],      [14:00], [1,037], [*14.87*],
    [S3 — Nocturnal],      [22:00], [170],   [*19.16*],
    [S4 — Counterfactual], [22:00 (fixed PBLH)], [313], [11.84],
  ),
  caption: [CO₂ concentrations by scenario — mean over 501 road segments.
            S4 isolates the contribution of PBLH collapse.]
)

Despite 65% less traffic, nocturnal pollution exceeds the afternoon value. S4 shows
PBLH collapse alone accounts for ~*38%* of the simulated nocturnal accumulation.

=== Interactive Mapping Module (carte_humg.py)

A Folium script generates a self-contained HTML map overlaying the three scenario
layers on CartoDB Positron. A custom JavaScript selector switches between 08:00,
14:00, and 22:00 views. A global colourmap (green → yellow → red) ensures visual
comparability. Each segment shows on click: road name, type, H/W ratio, CO₂, ΔT UHI.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 6pt,
    image("/images/Figure5a.png"),
    image("/images/Figure5b.png"),
  ),
  caption: [Interactive Folium map. Left: S1 (08:00, PBLH = 313 m) — maximum
            pollution. Right: S2 (14:00, PBLH = 1,037 m) — minimum pollution,
            maximum dispersion.]
)

#figure(
  image("/images/Figure5c.png", width: 72%),
  caption: [S3 (22:00, PBLH = 170 m): nocturnal rebound to 19.16 µg/m³
            despite 65% less traffic. Final deliverable:
            carte_pollution_humg_3scenarios.html.]
)

== Phase 4 — Weeks 14–16: Scientific and Academic Restitution

=== Scientific Article Drafting

To consolidate the research findings, all work was formalised into a ~25-page document titled _"A GIS-based Framework for Urban Traffic CO₂ Simulation Integrating Vehicle Detection and Atmospheric Conditions"_. Co-authored with Dr. Nguyen Gia Trong and Eric Gascard, it was structured as a full research article in English (Introduction, State of the Art, Methodology, Results, Discussion). It serves as the primary scientific restitution of the HUCODT framework.

=== Final Internship Report and Typesetting

The final weeks were dedicated to writing this comprehensive engineering report for Polytech Grenoble. To ensure high-quality, reproducible academic typesetting, the report was coded entirely in *Typst*. This phase required synthesizing 16 weeks of technical, scientific, and project management work into a structured document, complete with bilingual summaries and automated bibliography management, reflecting the professional standards expected for a 4th-year engineering internship.

// ════════════════════════════════════════════════════════════
// IV. ASSESSMENT
// ════════════════════════════════════════════════════════════

= Assessment and Value of the Internship

== Scientific Interest

This internship addresses a real and urgent challenge: air quality in South-East Asian
metropolises is a major public health issue for tens of millions of people @who2021.
The HUCODT framework is an implementation
simultaneously integrating vehicle detection via computer vision, ML-based emission
prediction, ERA5 atmospheric forcing, and interactive GIS visualisation on a complete
road network in a South-East Asian capital.

The main finding — nocturnal pollution is dominated by *atmospheric dynamics* rather
than traffic — has direct policy implications: traffic restrictions targeting morning peaks
have limited nocturnal impact, whereas interventions on *urban morphology* (road
widening, building setbacks, ventilation corridors) offer the most effective lever.

== Pedagogical and Professional Interest

This internship integrated in a single project skills from multiple fields: computer
vision, machine learning, geospatial data analysis, atmospheric modelling, advanced
Python engineering, and scientific writing in English. Working with no pre-existing
codebase in an international environment with real data from an Asian capital
constituted a particularly rich formative experience.

== Value of the Solution Delivered

The solution is reusable and open: published on GitHub, it adapts to any city with
OpenStreetMap data and Copernicus ERA5 API access. It requires no physical sensors,
proprietary software, or server infrastructure — its primary added value for
resource-constrained South-East Asian municipalities.

// ════════════════════════════════════════════════════════════
// V. WHAT COULD NOT BE DONE
// ════════════════════════════════════════════════════════════

= What Could Not Be Accomplished and Future Perspectives

== Unfinished Tasks

*Quantitative validation (W14–W15):* Validation against sensor data was not
carried out due to budgetary constraints. The validation performed is qualitative:
results were reviewed by the supervisor and recognised as consistent with
expected Hanoi pollution dynamics, in line with published comparable studies @ren2021.

*Advanced UHI modelling:* The UHI ΔT relies on an empirical parametrisation from
Oke (1988). Planned integration of Landsat LST satellite data for thermodynamic
coupling was not implemented. The advanced routing task was also paused at week 9.

*Carbon sink spatialisation:* While vegetation absorption was integrated mathematically into the accumulation equation @nowak2013, the precise spatial mapping of urban trees (e.g., using municipal GIS cadasters or remote sensing) could not be carried out. Absorption is currently estimated without segment-specific tree counts.

== Structural Limitations

- *No quantitative validation:* simulated concentrations are relative outputs, not absolute regulatory values.
- *CO₂ model trained on Canadian data:* Trained on Canadian data, the ML model does not fully capture the specific emission profiles of the South-East Asian motorbike fleet.
- *ERA5 resolution (~31 km):* The ~31 km resolution of ERA5 forces a spatially uniform PBLH, artificially smoothing out local micro-climatic variations.
- *YOLOv8n motorbike under-detection:* trained on Western images (COCO), the model under-counts motorbikes in dense Asian traffic.

== Research Perspectives and Future Work

- *Mobile Measurement Campaigns:* Deploying low-cost electrochemical sensors on bicycles or motorbikes to physically calibrate the model and achieve rigorous quantitative validation.
- *Multi-Satellite Coupling:* Integrating satellite remote sensing data to refine the thermal and atmospheric forcing of the microclimate model.
- *Continuous Temporal Simulation:* Scaling the dynamic engine to process full 24-hour cycles by leveraging the native hourly resolution of the ERA5 dataset.
- *Transfer Learning (Fine-Tuning):* Retraining the YOLOv8 architecture specifically on Vietnamese traffic datasets to correct the structural motorbike under-detection bias.
- *Smart City Synergy:* Interfacing the digital twin with empirical road-count data from municipal infrastructures currently being deployed across Vietnam.

// ════════════════════════════════════════════════════════════
// VI. PROJECT MANAGEMENT
// ════════════════════════════════════════════════════════════

= Project Management

== Life Cycle and Task Breakdown

The project followed an *incremental approach* : each step built upon the achievements of the previous one to deliver a functional module. A Gantt chart was created in the first week to structure the 16 weeks of work into 11 key tasks (see details in Appendix 1). This roadmap served as a management tool throughout the internship, allowing progress to be tracked, difficulties to be anticipated, and the schedule to be adjusted in response to unforeseen scientific challenges.

#figure(
  table(
    columns: (2.1fr, 0.65fr, 0.65fr, 0.8fr, 1.4fr),
    align: (left, center, center, center, left),
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Task / Phase],
      text(fill: white, weight: "bold")[Est.],
      text(fill: white, weight: "bold")[Actual],
      text(fill: white, weight: "bold")[Weeks],
      text(fill: white, weight: "bold")[Status],
    ),
    [Framework definition],           [2 w], [2 w], [W1–W2],   [Done],
    [GitHub setup & environment],     [2 w], [2 w], [W3–W4],   [Done],
    [CO₂ prediction model],           [3 w], [3 w], [W3–W5],   [Done],
    [Vehicle detection],              [3 w], [3 w], [W5–W7],   [Done],
    [Initial Digital Twin pipeline],  [3 w], [3 w], [W6–W8],   [Done],
    [Routing and road graphs],        [2 w], [1 w], [W7–W9],   [On hold — scope reprioritised],
    [Urban canyon dispersion & wind], [3 w], [6 w], [W8–W13],  [Done — extended (PBLH bottleneck)],
    [Carbon sinks / Vegetation],      [3 w], [—],  [W8–W13],  [Integrated in spatial engine],
    [Urban Heat Island modelling],    [3 w], [2 w], [W11–W13], [Partial — empirical only],
    [Global validation],              [2 w], [2 w],  [W14–W15], [Done],
    [Report writing],                 [2 w], [2 w], [W15–W16], [Done],
  ),
  caption: [Gantt dashboard — estimated vs actual durations, final status.]
)

== Risk Management

*Risk 1 — Unavailability of field data* (identified W1): mitigated by exclusive use of
open data. Final impact: acceptable, documented as a scope limitation.

*Risk 2 — Communication with supervisor* (materialised W1): email not functional,
resolved by switching to WhatsApp. Final impact: none.

*Risk 3 — Linux externally-managed environment*: resolved by strict venv setup.
Final impact: none.

*Risk 4 — Scientific scope creep* (PBLH bottleneck, week 7): Gantt adjusted,
routing task paused, dispersion engine extended by 3 weeks. Final impact: delay on
two secondary tasks; main deliverable unaffected.

== Monitoring and Progress Reporting

Five biweekly progress reports (W1, W4, W7, W10, W13, W16) were submitted to supervisors
on both sides, documenting Gantt progress, scientific results, architectural decisions,
and next steps.

// ════════════════════════════════════════════════════════════
// VII. SKILLS SUMMARY
// ════════════════════════════════════════════════════════════

= Summary of Knowledge and Skills Acquired

== Technical Skills

*Computer Vision and Deep Learning:* practical mastery of the YOLOv8 API
(Ultralytics), real-time object detection architectures, confidence threshold management.

*Applied Machine Learning:* training, evaluation and deployment of a scikit-learn
linear regression model on real data; R² and RMSE interpretation; pipeline integration.

*Python geospatial analysis:* OSMnx, GeoPandas, networkx, Folium — OSM road
networks, spatial joins, geometric attribute computation, interactive mapping.

*Climatological data processing:* xarray for multidimensional NetCDF files;
Copernicus REST API (cdsapi); UTC → UTC+7 timezone management.

*Software architecture:* modular design, separation of concerns, virtual environment management.

== Transferable Skills

*Scientific Research and Writing:* Conducted a literature review to establish links between computer science and numerous other fields. This theoretical work culminated in the co-authoring of a comprehensive research article in English, in collaboration with my Vietnamese and French professors, who assisted me with the writing process.

*Independent Project Management:* Managed a 16-week project with regular reporting. I primarily learned to be highly adaptable in the face of obstacles, dynamically adjusting the schedule and reorganizing development task priorities to ensure a fully functional deliverable.

*International Work Environment:* Integration into a Vietnamese university research team required cultural and interpersonal adaptability. I had to quickly overcome initial communication barriers and adapt to conducting all my daily work, from technical discussions to documentation, in English.

// ════════════════════════════════════════════════════════════
// VIII. KEY SKILL DEVELOPMENT  (1 page — consigne Polytech)
// ════════════════════════════════════════════════════════════

#pagebreak()
= Development of a Key Engineering Competency

_*Competency selected:* Concevoir des systèmes centrés données, langage, image et humain — Polytech Grenoble engineering skills framework._

*Status: acquired* — this competency was not part of my prior training and was built during the internship.

== What This Competency Means in Practice

Before this internship, my software projects operated on a single input modality at a time: either structured data, or image processing, or a user interface — never all four simultaneously, and never in a context where each modality was scientifically load-bearing. The defining challenge of HUCODT was precisely that no single modality was sufficient: traffic density is only visible in images; emission quantities only exist as structured numerical data; atmospheric trapping is only representable through a formal physical model; and the output only becomes useful when a non-specialist can read it without assistance. Designing a system that coupled all four without any one layer contaminating the others was the core engineering skill I had to acquire.

== How Each Modality Was Confronted and Learned

Image. I had never used an object detection framework before this internship. Learning to operate YOLOv8n — loading pre-trained COCO weights, applying a confidence threshold, mapping raw class labels to a vehicle taxonomy — took the better part of week 5. The first working detection on a real Hanoi traffic scene (Figure 2) was the proof of concept: 45 cars and 2 trucks identified in under 200 ms on a standard CPU. What I learned here was not just the API, but the discipline of treating the image module as a sealed unit whose only output is a vehicle count dictionary — so that everything downstream could be developed and tested independently.

Data. Three completely different data sources had to be acquired, cleaned, and made interoperable: a Canadian tabular CSV for ML training, ERA5 NetCDF tensors from a REST API, and OpenStreetMap vector geometries. None of them shared a coordinate system, timezone, or format. The practical skill acquired was pipeline hygiene — validating each source in isolation before any join, and catching the UTC→UTC+7 misalignment early enough not to corrupt the simulation. The CO₂ predictor scatter plot (Figure 1) is the concrete evidence that the data layer worked: R² = 0.876 on a held-out test set, with no systematic bias across the 100–500 g/km range.

Formal representation (the model as language). The accumulation equation from Peng et al. (2023) — C(s,t) = Q(s,t) − A(s) + C(s,t−1) × R(s,t) — was entirely new to me. I had to understand what each term encodes physically before I could implement it correctly. The key moment was when the supervisor pointed out at week 7 that a static model was scientifically wrong: pollution does not simply track traffic. Implementing the retention coefficient R as a function of PBLH and H/W ratio, and verifying that the 0D engine reproduced the expected nocturnal rebound on mock data before connecting to ERA5, was how I learned to treat a mathematical model as a formal language that must be validated in its own right. Figure 4 shows the result: despite a 65% traffic reduction at 22:00, simulated CO₂ exceeds the afternoon minimum by 29%, driven exclusively by PBLH collapse to 170 m.

Human. The interactive Folium map (Figures 5 and 6) was the modality I underestimated most at the start. Producing an HTML file that a municipality could open in any browser, switch between three time scenarios with a single click, and read per-segment CO₂ values without any GIS training required learning Folium's FeatureGroup system, injecting a custom JavaScript layer-switcher, and anchoring the colour scale globally across all three scenarios so that comparisons are visually honest. The lesson was that a human-facing output is not a cosmetic layer on top of a working system — it is a functional requirement that shapes every upstream design decision.

== Evidence of Acquisition

The diurnal CO₂ profile (Figure 4) and the interactive map (Figures 5–6) together constitute the clearest proof of competency acquisition: they only exist because all four modalities were successfully coupled. Image detection fed the fleet composition; structured data supplied the emission factors and meteorological forcing; the formal model translated those inputs into spatially differentiated concentrations with temporal memory; and the human interface made the output legible. None of these layers could have produced the result alone, and I had mastered none of them before April 2026.

// ════════════════════════════════════════════════════════════
// BIBLIOGRAPHY
// ════════════════════════════════════════════════════════════

#set heading(numbering: none)

// ════════════════════════════════════════════════════════════
// DOCUMENTS PRODUCED
// ════════════════════════════════════════════════════════════

= Documents Produced During the Internship

#figure(
  table(
    columns: (2.05fr, 1fr, 1fr, 0.45fr, 0.45fr, 0.6fr),
    align: (left, left, center, center, center, center),
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Title],
      text(fill: white, weight: "bold")[Co-authors],
      text(fill: white, weight: "bold")[Date],
      text(fill: white, weight: "bold")[Ver.],
      text(fill: white, weight: "bold")[Pages],
      text(fill: white, weight: "bold")[Lang.],
    ),
    [Progress report W1],
      [—], [13/04/2026], [v1.0], [1], [FR],
    [Progress report W4],
      [—], [04/05/2026], [v1.0], [4], [FR],
    [Progress report W7],
      [—], [25/05/2026], [v1.0], [6], [FR],
    [Progress report W10],
      [—], [15/06/2026], [v1.0], [7], [FR],
    [Progress report W13],
      [—], [06/07/2026], [v1.0], [8], [FR],
    [Progress report W16],
      [—], [27/07/2026], [v1.0], [1], [FR],
    [_A GIS-based Framework for Urban Traffic CO₂ Simulation Integrating Vehicle Detection and Atmospheric Conditions_],
      [Nguyen Gia Trong, Eric Gascard], [Jul. 2026], [RV], [~25], [EN],
    [GitHub codebase: _Mobility-co2-Simulation_],
      [—], [Apr.–Jul. 2026], [HEAD], [—], [Python],
    [carte_pollution_humg
    3scenarios.html],
      [—], [Jul. 2026], [v1.0], [—], [HTML],
  ),
  caption: [Documents produced or co-produced during the internship.]
)

// ════════════════════════════════════════════════════════════
// APPENDICES
// ════════════════════════════════════════════════════════════
