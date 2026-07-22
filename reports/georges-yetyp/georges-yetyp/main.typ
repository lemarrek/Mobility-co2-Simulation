#import "@preview/georges-yetyp:0.2.0": rapport

#show: rapport.with(
  nom: "LE MARREC Kelig",
  titre: "Artificial Intelligence and GIS Applied to Urban Mobility Simulation and CO₂ Emissions Modelling",
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
    email: "jean-francois.mehaut@grenoble-inp.fr"
  ),
  résumé: [
    This report presents the work carried out during a 16-week internship (April 4 to July 27, 2026) at the Hanoi University of Mining and Geology (HUMG), Vietnam, under the supervision of Prof. Nguyen Gia Trong. 

    The main deliverable is the HUCODT (_Hanoi Urban CO₂ Digital Twin_) software framework — a fully open-source, Python-based urban air quality digital twin integrating four modules: YOLOv8-based vehicle detection, linear regression CO₂ prediction, an ERA5-forced atmospheric dispersion engine coupled with Oke's (1988) urban street-canyon model, and an interactive GIS visualisation layer built with Folium. Applied across 501 road segments around the HUMG campus, the system demonstrates that simulated nocturnal CO₂ concentrations (19.16 µg/m³) exceed afternoon values (14.87 µg/m³) despite a 65% traffic reduction, driven by the collapse of the planetary boundary layer height (PBLH) to 170 m.

    *Keywords:* GIS, CO₂ emissions, urban traffic, digital twin, YOLOv8, ERA5, street canyon, Hanoi.
  ],
  glossaire: [
    / HUCODT: Hanoi Urban CO₂ Digital Twin — name of the software framework developed during the internship.
    / GIS: Geographic Information System. Tool for capturing, storing, analysing and displaying spatial data.
    / PBLH: Planetary Boundary Layer Height — altitude up to which atmospheric turbulence disperses pollutants.
    / ERA5: Fifth generation global atmospheric reanalysis from the European Centre for Medium-Range Weather Forecasts (ECMWF), accessible via Copernicus.
    / YOLOv8: You Only Look Once v8 — real-time object detection model developed by Ultralytics.
    / OSMnx: Python library for downloading and analysing road networks from OpenStreetMap.
    / GeoPackage: Open geospatial file format based on SQLite (.gpkg).
    / NetCDF: Network Common Data Form — standard format for multidimensional scientific data (climatology).
    / H/W: Height-to-Width ratio of an urban canyon — key parameter in Oke's (1988) model.
    / UHI: Urban Heat Island. Localised temperature increase in dense urban areas.
    / ML: Machine Learning.
    / venv: Python virtual environment for isolating project dependencies.
    / COCO: Common Objects in Context — benchmark dataset for training object detection models.
    / Open Data: Data freely available for use, reuse and redistribution without major restriction.
  ]
)

// ============================================================
// ABSTRACT
// ============================================================

= Abstract

This report presents the work carried out during a 16-week internship (April 4 to July 27, 2026) at the Hanoi University of Mining and Geology (HUMG), Vietnam, under the supervision of Prof. Nguyen Gia Trong. The objective was to develop from scratch a simulation system for traffic-related urban CO₂ emissions, combining Artificial Intelligence and Geographic Information Systems (GIS).

The main deliverable is the HUCODT (_Hanoi Urban CO₂ Digital Twin_) software framework — a fully open-source, Python-based urban air quality digital twin integrating four modules: YOLOv8-based vehicle detection, linear regression CO₂ prediction, an ERA5-forced atmospheric dispersion engine coupled with Oke's (1988) urban street-canyon model, and an interactive GIS visualisation layer built with Folium. Applied across 501 road segments around the HUMG campus, the system demonstrates that simulated nocturnal CO₂ concentrations (19.16 µg/m³) exceed afternoon values (14.87 µg/m³) despite a 65% traffic reduction, driven by the collapse of the planetary boundary layer height (PBLH) to 170 m.

*Keywords:* GIS, CO₂ emissions, urban traffic, digital twin, YOLOv8, ERA5, street canyon, Hanoi.

// ============================================================
// 1. CONTEXT
// ============================================================

= Internship Context and Environment

== Host institution

The internship took place at the *Hanoi University of Mining and Geology (HUMG)*, one of Vietnam's most prestigious technical universities, specialising in Earth sciences, mining, geology, and geodesy-cartography. The university maintains numerous international collaborations for scientific research and technology transfer.

The internship supervisor, *Dr. Nguyen Gia Trong*, is a lecturer-researcher in the Department of Higher Geodesy (Faculty of Geodesy, Cartography and Land Management). A specialist in geodetic data processing and satellite positioning (GNSS/GPS), he now focuses his research on integrating AI and GIS for environmental management, smart cities, and natural hazard assessment. Academic supervision on the Polytech Grenoble side was provided by Eric Gascard (academic supervisor) and Jean-François Méhaut (internship coordinator).

== Scientific context

Road transport accounts for approximately 23% of global energy-related CO₂ emissions @iea2023. In Hanoi — a city of more than 8 million inhabitants with a motorbike fleet exceeding 6 to 8 million units — the city's morphological configuration creates *urban canyon* conditions that trap emissions and amplify their thermal and chemical effects @oke1988. Deploying physical sensor networks remains economically prohibitive: a dense network covering a single district can cost several hundred thousand dollars. This *structural data gap* constrains both scientific understanding and public policy on air pollution.

== Installation conditions

The first week was devoted to initial contact and setup. An early difficulty arose: the supervisor's email address was not functional, which required switching to WhatsApp to establish first contact. A scoping meeting then allowed the expected skills and precise objectives to be defined.

// ============================================================
// 2. PROJECT DESCRIPTION
// ============================================================

= Proposed Project: Synthetic Description

The internship objective was to develop from scratch a *Python software system* capable of simulating traffic-related CO₂ emissions in an urban environment, combining:

- *Machine Learning algorithms* for per-vehicle CO₂ emission prediction;
- *Computer Vision* for automatic vehicle detection in road traffic images;
- *spatial analysis libraries (GIS)* for mapping and spatialisation of results;
- *open meteorological data* for modelling atmospheric dispersion.

The application scope is centred on urban mobility simulation and CO₂ emissions modelling at neighbourhood scale in Hanoi, with the hard constraint of relying exclusively on open data, to compensate for the impossibility of deploying physical sensors on the ground. Upon arrival, the team had a 16-week work plan but *no pre-existing codebase*. The software architecture, choice of AI models, and selection of GIS libraries were therefore designed entirely during the internship.

// ============================================================
// 3. WORK CARRIED OUT
// ============================================================

= Work Effectively Carried Out

The work developed incrementally across three major phases, following a coherent scientific and technical progression: from individual building blocks to an integrated system, then from a single-point simulation (0D) to a full spatial simulation (2D on the road network).

== Phase 1 — Weeks 1 to 4: Technical foundation and first building blocks

=== Development environment

The first phase began with setting up a professional development environment on Linux with VS Code: creation of a GitHub repository (_Mobility-co2-Simulation_), configuration of a strict Python virtual environment (venv) for dependency isolation, and structuring the project into distinct modules.

=== CO₂ prediction model (co2_predictor.py)

The first module developed is a Machine Learning model trained on the public _CO2 Emissions Canada_ dataset (7,385 records). The model is implemented as a *CO2Predictor* class with a constructor that loads and cleans the data, and a *predict_vehicle()* method that takes a vehicle's characteristics (engine displacement, number of cylinders, fuel consumption) and returns its CO₂ emission in g/km.

The chosen algorithm is *linear regression* (scikit-learn), justified by the mathematically continuous and strongly linear relationship between a combustion engine's characteristics and its emissions. Performance on the test set (20% of the data): *R² = 0.876* and *RMSE = 18.4 g/km* — consistent with published work on similar datasets @mokhtarzadeh2021.

#figure(
  image("/images/Figure1.png"),
  caption: [Scatter plot: predicted vs actual CO₂ emissions on the test set (g/km). R² = 0.876. Source: execution of 01_test.ipynb.]
)

=== Vehicle detection with Computer Vision (vision_test.py)

A Computer Vision module using the *YOLOv8n* model was integrated to automatically detect and classify vehicles in road traffic images. The lightweight yolov8n.pt model was chosen to ensure fast inference on standard CPU hardware. On a test image (1920×1080 px), the model detected 45 cars and 2 trucks in under 200 ms, with confidence scores above 0.50 for the majority of detections.

#figure(
  image("/images/Figure1.png"),
  caption: [YOLOv8n detection on urban road traffic (vision_test.py). Detection bounding boxes by class (car / truck) with confidence scores.]
)

=== Initial Digital Twin pipeline (digital_twin_pipeline.py)

The critical step of this phase was the *fusion of the two previous building blocks* into a coherent execution pipeline. The script orchestrates: (1) YOLO analysis of the image to count vehicles by class; (2) assignment of a mean technical profile to each detected class; (3) injection into the CO2Predictor; (4) aggregation to provide a global emission estimate for the scene. This pipeline produced an estimate of *10,246.65 g/km of CO₂* for the test scene.

#figure(
  rect(width: 100%, height: 5cm, fill: rgb("#D9D9D9"), stroke: none)[
    #align(center + horizon)[
      #text(fill: rgb("#555555"), style: "italic")[
        [ FIGURE 3 — Insert here the console output of digital_twin_pipeline.py ]
      ]
    ]
  ],
  caption: [Execution of digital_twin_pipeline.py: detected vehicle breakdown and total emission estimate (10,246.65 g/km CO₂).]
)

=== Cartography and GIS (02_map_simulation.ipynb)

In parallel, a first GIS module was developed using OSMnx and networkx. The script queries the OpenStreetMap API, downloads the road network topology as a graph (nodes and edges), and implements a routing algorithm computing the shortest path between two GPS coordinates (HUMG campus → Hoa Binh Park, distance: *3.06 km*).

== Phase 2 — Weeks 5 to 10: Dynamic modelling and atmospheric integration

=== Identification of a major scientific bottleneck

From week 7 onwards, the supervisor directed the work towards a major scientific challenge: the emissions model was until then "static" — it assumed that pollution decreased proportionally with traffic. However, in South-East Asian metropolises, the drop in nocturnal temperatures causes a collapse of the *Planetary Boundary Layer* (PBLH). This phenomenon traps residual emissions and creates a dangerous accumulation of pollutants at night, even when traffic is low @deng2023.

The new research question: _how to algorithmically simulate nocturnal pollution accumulation by crossing traffic data with atmospheric boundary layer dynamics?_

=== Literature review

A scientific literature review was conducted on the following publications:

- *Deng et al. (2023)* @deng2023: demonstrates the importance of ERA5 Copernicus data for modelling the nocturnal thermal "cap";
- *Peng et al. (2023)* @peng2023: provides the theoretical framework for coupling urban canyon geometry with dynamic PBLH variations;
- *Oke (1988)* @oke1988: models wind behaviour in urban canyons via the H/W ratio;
- *Nowak et al. (2013)* @nowak2013: quantifies vegetation absorption (≈ 21 kg of CO₂ per tree per year).

=== ERA5 data acquisition module (download_era5.py)

An automation script downloads ERA5 meteorological data via the Copernicus REST API (cdsapi). It extracts NetCDF files containing 3D tensors (Latitude × Longitude × Time) for PBLH and 10-metre wind vectors. Rigorous timezone management (ERA5 in UTC → local Hanoi time UTC+7) is a critical step implemented via xarray @hersbach2020.

=== Dynamic simulation engine (simulation_humg.py v4)

This module is the *scientific core* of the project: a recursive algorithm with temporal memory management, implementing the accumulation equation from Peng et al. (2023):

$ C_t = Q_t + (C_(t-1) times R) $

$Q_t$ represents instantaneous emissions, $C_(t-1)$ the concentration at the previous time step, and $R$ a *dynamic retention rate*. When wind is weak and PBLH is low, $R$ tends towards 0.85 (strong accumulation); when wind is strong and PBLH is high, $R$ tends towards 0 (maximum dispersion).

The scientific visualisation module (plot_results.py) demonstrated the validity of the hypothesis visually: a *+30% increase in pollution at 22:00* driven exclusively by thermal inversion, despite a 60% drop in traffic.

#figure(
  rect(width: 100%, height: 8cm, fill: rgb("#D9D9D9"), stroke: none)[
    #align(center + horizon)[
      #text(fill: rgb("#555555"), style: "italic")[
        [ FIGURE 4 — Insert here the diurnal CO₂ vs Traffic vs PBLH chart \
        (screenshot from plot_results.py, Twin Axes matplotlib architecture) ]
      ]
    ]
  ],
  caption: [Simulated diurnal profile: CO₂ concentration (curve, left axis) overlaid with traffic factor (bars, right axis) and PBLH. Generated by plot_results.py.]
)

== Phase 3 — Weeks 11 to 13: Full spatialisation and final deliverable

=== Spatial simulation module (simulation_spatiale_humg.py)

This script is the major extension of the 0D engine into a *fully operational 2D model* covering 501 road segments (1.5 km radius around the HUMG campus). It loads the OSMnx road graph and building footprints from GeoPackage files, then applies the dynamic model in three successive passes (08:00 → 14:00 → 22:00), propagating concentration memory between scenarios.

Geometric parameters are estimated from OpenStreetMap: road width inferred from OSM road type, building height computed from the building:levels attribute (×3 m/floor, default: 12 m). A 25 m geospatial buffer around each segment identifies adjacent buildings via intersection query. The vehicle fleet is modulated by a time-of-day factor (1.0 at 08:00, 0.6 at 14:00, 0.35 at 22:00).

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Scenario],
      text(fill: white, weight: "bold")[Time],
      text(fill: white, weight: "bold")[PBLH (m)],
      text(fill: white, weight: "bold")[Mean CO₂ (µg/m³)],
    ),
    [S1 — Morning peak],    [08:00],             [313],   [*29.95*],
    [S2 — Afternoon],       [14:00],             [1,037], [*14.87*],
    [S3 — Nocturnal],       [22:00],             [170],   [*19.16*],
    [S4 — Counterfactual],  [22:00 (fixed PBLH)],[313],   [11.84],
  ),
  caption: [Simulated CO₂ concentrations by scenario — mean over 501 road segments. S4 is the counterfactual scenario (constant PBLH) that isolates the effect of nocturnal PBLH collapse.]
)

These results confirm the central hypothesis: despite 65% less traffic, nocturnal pollution exceeds the afternoon value. The counterfactual scenario S4 shows that PBLH collapse alone accounts for approximately *38%* of the simulated nocturnal accumulation.

=== Interactive mapping module (carte_humg.py)

A script generates a self-contained interactive HTML map using *Folium*. A dynamic JavaScript selector allows switching between the three scenarios (08:00 / 14:00 / 22:00). A global linear colourmap (green → yellow → red) ensures visual comparability across time slots. Each segment displays on click its name, road type, H/W ratio, CO₂ concentration, and UHI ΔT.

The final deliverable *carte_pollution_humg_3scenarios.html* is a standalone file, with no server dependency, openable directly in any web browser.

#figure(
  rect(width: 100%, height: 8cm, fill: rgb("#D9D9D9"), stroke: none)[
    #align(center + horizon)[
      #text(fill: rgb("#555555"), style: "italic")[
        [ FIGURE 5a — Insert here the Folium map, scenario 08:00 \
        (screenshot from carte_pollution_humg_3scenarios.html) ]
      ]
    ]
  ],
  caption: [Interactive Folium map, scenario S1 (08:00, PBLH = 313 m). Road segments coloured by simulated CO₂ concentration (green → red). CartoDB Positron basemap.]
)

#figure(
  rect(width: 100%, height: 8cm, fill: rgb("#D9D9D9"), stroke: none)[
    #align(center + horizon)[
      #text(fill: rgb("#555555"), style: "italic")[
        [ FIGURE 5b — Insert here the Folium map, scenario 14:00 \
        (maximum dispersion, PBLH = 1,037 m) ]
      ]
    ]
  ],
  caption: [Scenario S2 (14:00): minimum concentration (14.87 µg/m³), PBLH = 1,037 m, maximum atmospheric dispersion.]
)

#figure(
  rect(width: 100%, height: 8cm, fill: rgb("#D9D9D9"), stroke: none)[
    #align(center + horizon)[
      #text(fill: rgb("#555555"), style: "italic")[
        [ FIGURE 5c — Insert here the Folium map, scenario 22:00 \
        (nocturnal rebound despite traffic reduction) ]
      ]
    ]
  ],
  caption: [Scenario S3 (22:00): nocturnal rebound to 19.16 µg/m³ despite 65% less traffic. PBLH collapsed to 170 m.]
)

=== Scientific article

All work was formalised in a scientific article written in English: _"A GIS-based Framework for Urban Traffic CO₂ Simulation Integrating Vehicle Detection and Atmospheric Conditions"_, co-authored with Prof. Nguyen Gia Trong and Eric Gascard. It presents the HUCODT framework, the methodology, the results of the three scenarios, a critical discussion of limitations, and a comparison with the existing literature.

// ============================================================
// 4. ASSESSMENT
// ============================================================

= Assessment and Value of the Internship

== Scientific interest

This internship is of notable scientific value. It addresses a real and urgent challenge: air quality in South-East Asian metropolises is a major public health issue for tens of millions of people @who2021. Furthermore, the HUCODT framework is, to our knowledge, the first open-source implementation simultaneously integrating vehicle detection via computer vision, ML-based emission prediction, ERA5 atmospheric forcing, and interactive GIS visualisation on a complete road network in a South-East Asian capital city.

The main finding — that nocturnal pollution is dominated by atmospheric dynamics rather than traffic — has direct implications for public policy: traffic restrictions targeting the morning peak have limited impact on the nocturnal exposure of residents, whereas interventions on urban morphology (road widening, building setbacks, ventilation corridors) would offer the most effective lever.

== Pedagogical and professional interest

On a personal level, this internship made it possible to integrate in a single project skills from multiple fields: computer vision, machine learning, geospatial data analysis, atmospheric modelling, advanced Python software engineering, and scientific writing in English. The constraint of working with no pre-existing codebase, in an international environment, with real data from an Asian capital, constituted a particularly rich formative experience.

== Value of the solution delivered

The solution is reusable and open: published in full on GitHub, it can be adapted to any city with OpenStreetMap data and access to the Copernicus ERA5 API. It requires neither physical sensors, nor proprietary software, nor server infrastructure. Its scalability and reproducibility are its primary added value for resource-constrained municipalities in South-East Asia.

// ============================================================
// 5. WHAT COULD NOT BE DONE AND FUTURE WORK
// ============================================================

= What Could Not Be Accomplished and Future Perspectives

== Unfinished tasks

Two tasks from the initial Gantt chart remained planned for weeks 14 to 16:

*Quantitative validation (S14–S15):* Validation against in-situ sensor data was not carried out. This limitation was anticipated from the outset due to budgetary constraints. The validation performed is qualitative: results were reviewed by the academic supervisor and recognised as consistent with the pollution dynamics expected in Hanoi.

*Advanced UHI modelling:* The UHI temperature delta (ΔT_UHI) relies on an empirical parametrisation from Oke (1988). The planned improvement — integrating Landsat LST satellite data — was not implemented. The advanced routing task was also paused after week 9 to focus resources on model spatialisation.

== Structural limitations

- *Absence of quantitative validation:* simulated concentrations are relative outputs, not absolute regulatory values.
- *CO₂ model trained on Canadian data:* small-engine Vietnamese motorbikes may deviate from the linear model.
- *ERA5 spatial resolution (~31 km):* PBLH is spatially uniform across the study area, potentially underestimating intra-urban heterogeneity.
- *Motorbike under-detection by YOLOv8n:* trained on COCO (primarily Western images), the model under-counts motorbikes in dense Asian traffic.

== Research perspectives

- Quantitative validation via mobile measurement campaigns with low-cost electrochemical sensors.
- Integration of Sentinel-5P (tropospheric NO₂) and Landsat LST to improve atmospheric and thermal forcing.
- Extension to a full 24-hour diurnal simulation at hourly ERA5 resolution.
- Fine-tuning of YOLOv8 on Vietnamese road traffic images to improve motorbike detection.
- Integration of municipal road-count data as it becomes available under Vietnam's Smart City programmes.

// ============================================================
// 6. PROJECT MANAGEMENT
// ============================================================

= Project Management

== Life cycle and task breakdown

The project followed an *incremental development model*: each phase produces a functional deliverable that forms the basis of the next. This approach is suited to an R&D project where scientific constraints evolve over time (introduction of the PBLH bottleneck at week 7). The initial planning was formalised in a *Gantt chart* established at week 1, structuring 11 tasks over 16 weeks (see Appendix 1).

#figure(
  table(
    columns: (2fr, 0.7fr, 0.7fr, 1.5fr),
    align: (left, center, center, left),
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Task / Phase],
      text(fill: white, weight: "bold")[Duration],
      text(fill: white, weight: "bold")[Weeks],
      text(fill: white, weight: "bold")[Final status],
    ),
    [Framework definition],              [2 wks], [S1–S2],   [Done],
    [GitHub setup & dev environment],    [2 wks], [S3–S4],   [Done],
    [CO₂ prediction model],              [3 wks], [S3–S5],   [Done],
    [Vehicle detection],                 [3 wks], [S5–S7],   [Done],
    [Initial Digital Twin pipeline],     [3 wks], [S6–S8],   [Done],
    [Routing and road graphs],           [2 wks], [S7–S9],   [On hold],
    [Urban canyon dispersion & wind],    [3 wks], [S8–S13],  [Done],
    [Carbon sinks / Vegetation abs.],   [3 wks], [S8–S13],  [Integrated],
    [Urban Heat Island modelling],       [3 wks], [S11–S13], [Partial],
    [Global validation],                 [2 wks], [S14–S15], [Qual. only],
    [Report writing],                    [2 wks], [S15–S16], [In progress],
  ),
  caption: [Gantt dashboard — status of the 11 tasks at the end of the internship. Full diagram in Appendix 1.]
)

== Risk management

*Risk 1 — Unavailability of field data:* identified at S1, mitigated by exclusive use of open data (OpenStreetMap, ERA5, CO2 Emissions Canada). Final impact: acceptable, documented as a scope limitation.

*Risk 2 — Communication with supervisor:* email not functional at S1, resolved by switching to WhatsApp. Final impact: none.

*Risk 3 — Linux restrictions (environment externally managed):* resolved by setting up a strict venv. Final impact: none.

*Risk 4 — Scientific scope creep (introduction of PBLH bottleneck):* Gantt adjusted, advanced routing task paused. Final impact: delay on two secondary tasks, no impact on the main deliverable.

== Monitoring and progress reporting

Five *biweekly progress reports* were written and sent to supervisors on both sides (S1, S4, S7, S10, S13), documenting progress against the Gantt, scientific results obtained, architectural decisions made, and next steps. These reports structured the regular reviews with the supervisor.

// ============================================================
// 7. SKILLS SUMMARY
// ============================================================

= Summary of Knowledge and Skills Acquired

== Technical skills

*Computer Vision and Deep Learning:* practical mastery of the YOLOv8 API (Ultralytics), understanding of real-time object detection architectures, management of confidence thresholds and class filtering.

*Applied Machine Learning:* training, evaluation and deployment of a scikit-learn linear regression model on a real dataset; interpretation of R² and RMSE metrics; integration into an operational pipeline.

*Python geospatial analysis:* mastery of OSMnx, GeoPandas, networkx and Folium for downloading OSM road networks, performing spatial joins, computing geometric attributes and producing interactive maps.

*Climatological data processing:* use of xarray to manipulate multidimensional NetCDF files; access to the Copernicus REST API (cdsapi); timezone management (UTC → UTC+7).

*Software architecture:* modular design decoupling the computation engine from the visualisation layer; mock-up to production approach to validate mathematical logic before connecting to production APIs.

== Transferable skills

*Scientific research:* autonomous literature review on interdisciplinary topics (urban meteorology, atmospheric dispersion, computer vision, GIS); writing a complete scientific article in English to academic publication standards.

*Autonomous project management:* 16-week planning, structured biweekly monitoring, dynamic schedule adjustment in response to scientific and technical unforeseen events.

*Work in an international context:* integration into a Vietnamese research team, adaptation to different communication styles, daily work in English.

// ============================================================
// 8. KEY SKILL DEVELOPMENT
// ============================================================

= Development of a Key Skill: Modular Software Architecture

_Engineering competency: Designing an architectured, maintainable and scalable software solution (Domain "Software design and development")._

== Context and challenge

Upon arrival, no code existed. The temptation to write a monolithic script that "does everything" was real. However, the supervisor expected an engineering-quality system — reusable and extensible. The challenge was therefore to design a *modular software architecture* from the outset, applying the principle of separation of concerns.

== Concrete implementation

The HUCODT framework is structured into five decoupled modules, each encapsulating a single responsibility:

#figure(
  table(
    columns: (1.2fr, 1.5fr, 2fr),
    align: left,
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Module],
      text(fill: white, weight: "bold")[Python script],
      text(fill: white, weight: "bold")[Responsibility],
    ),
    [Climatological data],    [download_era5.py],              [ERA5 acquisition via cdsapi → 3D NetCDF],
    [Road network & GIS],     [02_map_simulation.ipynb],       [OSMnx, GeoPackage, road graph],
    [Detection + prediction], [vision_test.py + co2_predictor.py], [YOLO → CO2Predictor (ML)],
    [Spatial simulation],     [simulation_spatiale_humg.py],   [2D engine, 501 segments, 3 scenarios],
    [Visualisation],          [carte_humg.py],                 [Folium HTML, JS scenario selector],
  ),
  caption: [HUCODT modular architecture: five decoupled modules, each with a single responsibility.]
)

The simulation engine (simulation_spatiale_humg.py) contains no visualisation code. It produces standardised GeoPackages consumed independently by carte_humg.py. This means the visualisation layer can be replaced (for example with a QGIS integration) without touching the computation engine.

The *mock-up to production approach* was also a key architectural choice: all the mathematical logic of the differential equations was first validated with hard-coded data, before connecting the code to the ERA5 production API. This approach allowed independent debugging of the scientific logic and the data integration.

== Measurable outcome

- Each module is independently testable (notebooks 01_test.ipynb and 02_map_simulation.ipynb).
- Scaling from 1 segment (0D) to 501 segments (2D) required *no refactoring* of the ML module or the visualisation module.
- The GitHub codebase is reusable by other researchers for any city with OSM data and ERA5 access.

// ============================================================
// BIBLIOGRAPHY
// ============================================================

= Bibliography

#bibliography("refs.bib", style: "ieee")

// ============================================================
// DOCUMENTS PRODUCED
// ============================================================

= Documents Produced During the Internship

#figure(
  table(
    columns: (2.5fr, 1.5fr, 0.8fr, 0.5fr, 0.7fr),
    align: (left, left, center, center, center),
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Title],
      text(fill: white, weight: "bold")[Co-authors],
      text(fill: white, weight: "bold")[Date],
      text(fill: white, weight: "bold")[Pages],
      text(fill: white, weight: "bold")[Language],
    ),
    [Progress report W1],  [—], [13/04/2026], [1],   [FR],
    [Progress report W4],  [—], [04/05/2026], [4],   [FR],
    [Progress report W7],  [—], [25/05/2026], [6],   [FR],
    [Progress report W10], [—], [15/06/2026], [7],   [FR],
    [Progress report W13], [—], [06/07/2026], [8],   [FR],
    [Article: _A GIS-based Framework for Urban Traffic CO₂ Simulation..._],
      [Nguyen Gia Trong, Eric Gascard], [Jul. 2026], [~25], [EN],
    [GitHub codebase : Mobility-co2-Simulation],
      [—], [Apr.–Jul. 2026], [—], [Python],
    [carte_pollution_humg_3
    scenarios.html],
      [—], [Jul. 2026], [—], [HTML/JS],
  ),
  caption: [List of documents produced or co-produced during the internship.]
)

// ============================================================
// APPENDICES
// ============================================================

= Appendices

== Appendix 1 — Gantt Chart

#figure(
  rect(width: 100%, height: 8cm, fill: rgb("#D9D9D9"), stroke: none)[
    #align(center + horizon)[
      #text(fill: rgb("#555555"), style: "italic")[
        [ APPENDIX 1 — Insert here the high-resolution Gantt chart \
        (source file: Diagramme_de_Gantt.pdf) ]
      ]
    ]
  ],
  caption: [Gantt chart of the project over 16 weeks (11 tasks). Source: Diagramme_de_Gantt.pdf.]
)

== Appendix 2 — HUCODT Software Architecture

```
Mobility_Co2_Simulation/
├── src/
│   ├── co2_predictor.py             # ML module: CO₂ prediction
│   ├── vision_test.py               # CV module: YOLOv8 detection
│   ├── digital_twin_pipeline.py     # Orchestrator
│   ├── download_era5.py             # Data module: Copernicus API
│   ├── simulation_spatiale_humg.py  # 2D engine: dispersion + GIS
│   └── carte_humg.py                # Visualisation module: Folium
├── data/
│   ├── raw/                         # CSV, NetCDF
│   └── processed/                   # GeoPackage outputs
├── notebooks/
│   ├── 01_test.ipynb                # ML model validation
│   └── 02_map_simulation.ipynb      # GIS exploration
└── requirements.txt
```

== Appendix 3 — Main Simulation Equation

Dynamic accumulation equation @peng2023, applied to each road segment $s$ at time step $t$:

$ C(s,t) = Q(s,t) - A(s) + C(s, t-1) times R(s,t) $

Where:

- $C(s,t)$: CO₂ concentration at segment $s$, time $t$ (µg/m³)
- $Q(s,t)$: instantaneous emission flux (fleet × EEA factor × time-of-day factor)
- $A(s)$: vegetation absorption @nowak2013 (21 kg CO₂/tree/year)
- $R(s,t) = "clip"(1 - u_"eff" \/ H_"mix",\ 0,\ 0.85)$: dynamic retention rate
- $u_"eff"$: effective wind speed in the canyon $= U_10 times f(H\/W)$, after @oke1988
- $H_"mix"$: effective mixing height $= min("PBLH",\ h_"bld" times (1 + 1\/(H\/W)))$

== Appendix 4 — Technologies and Libraries Used

#figure(
  table(
    columns: (1.2fr, 1.5fr, 2fr),
    align: left,
    fill: (_, row) => if row == 0 { rgb("#1F3864") } else { white },
    table.header(
      text(fill: white, weight: "bold")[Domain],
      text(fill: white, weight: "bold")[Tool / Library],
      text(fill: white, weight: "bold")[Use],
    ),
    [Machine Learning],      [scikit-learn],                         [Linear regression CO₂],
    [Computer Vision],       [YOLOv8n (Ultralytics)],               [Vehicle detection],
    [Spatial analysis],      [OSMnx, GeoPandas, networkx],          [OSM road network, spatial joins],
    [Climatological data],   [xarray, cdsapi],                      [ERA5 NetCDF, Copernicus API],
    [Visualisation],         [matplotlib, Folium],                   [Charts, interactive HTML map],
    [Environment],           [Python 3.12, venv, VS Code, Linux],   [Dev & deployment],
    [Open Data],             [OpenStreetMap, CO2 Emissions Canada, ERA5], [Source datasets],
    [Version control],       [Git / GitHub],                         [github.com/lemarrek],
  ),
  caption: [Full technology stack of the HUCODT project.]
)
