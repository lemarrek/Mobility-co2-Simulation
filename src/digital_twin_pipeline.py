from ultralytics import YOLO
from co2_predictor import CO2Predictor

print("--- INITIALISATION DU JUMEAU NUMÉRIQUE ---")

# Démarrage de l'Analysis Centre (Ton modèle de Machine Learning)
chemin_donnees = "../data/raw/CO2 Emissions_Canada.xls"
simulateur_co2 = CO2Predictor(chemin_donnees)

# Démarrage de la Caméra (YOLO)
model = YOLO('yolov8n.pt')
chemin_image = "../data/raw/traffic_jam.jpg"

print("\n--- ANALYSE EN TEMPS RÉEL ---")
# YOLO regarde l'image
resultats = model(chemin_image, verbose=False)

# On compte les véhicules détectés
nombre_vehicules = 0
for r in resultats:
    boxes = r.boxes
    for box in boxes:
        # On récupère le nom de ce que YOLO a vu
        classe_id = int(box.cls[0])
        nom_classe = model.names[classe_id]
        
        # Si c'est un véhicule routier, on augmente le compteur
        if nom_classe in ['car', 'truck', 'bus', 'motorcycle']:
            nombre_vehicules += 1

print(f"YOLO a détecté : {nombre_vehicules} véhicules sur cette portion de route.")

# Calcul des émissions totales
if nombre_vehicules > 0:
    # Pour la simulation, on imagine un véhicule moyen (Moteur 2.0L, 4 Cylindres, Conso 8.5L/100)
    co2_moyen_par_vehicule = simulateur_co2.predict_vehicle(engine_size=2.0, cylinders=4, fuel_consumption=8.5)
    
    # On multiplie par le nombre de voitures vues par YOLO
    co2_total = nombre_vehicules * co2_moyen_par_vehicule
    print(f"ÉMISSION TOTALE ESTIMÉE : {co2_total:.2f} g/km de CO2")
else:
    print("La route est vide, aucune émission de CO2 détectée !")