from ultralytics import YOLO
from co2_predictor import CO2Predictor

print("--- INITIALISATION DU JUMEAU NUMÉRIQUE ---")

chemin_donnees = "../data/raw/CO2 Emissions_Canada.xls"
simulateur_co2 = CO2Predictor(chemin_donnees)

# Démarrage de YOLO
model = YOLO('yolov8n.pt')
chemin_image = "../data/raw/traffic_jam.jpg"

profils_vehicules = {
    'car': (2.0, 4, 8.5),         # Voiture standard
    'truck': (5.0, 8, 18.0),      # Gros pick-up ou camion
    'bus': (6.0, 8, 25.0),        # Bus urbain
    'motorcycle': (1.0, 2, 4.5)   # Moto
}

print("\n--- ANALYSE EN TEMPS RÉEL ---")
# YOLO regarde l'image
resultats = model(chemin_image, verbose=False)

# analyse
vehicules_detectes = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
co2_total = 0.0

# On compte les véhicules détectés
nombre_vehicules = 0
for r in resultats:
    boxes = r.boxes
    for box in boxes:
        # On récupère le nom de ce que YOLO a vu
        classe_id = int(box.cls[0])
        nom_classe = model.names[classe_id]
        
        # Si c'est un véhicule routier, on augmente le compteur
        if nom_classe in profils_vehicules:
            vehicules_detectes[nom_classe] += 1
            
            # On récupère ses caractéristiques spécifiques
            moteur, cyl, conso = profils_vehicules[nom_classe]
            
            # On demande à notre IA de calculer le CO2 du véhicule 
            co2_vehicule = simulateur_co2.predict_vehicle(moteur, cyl, conso)
            co2_total += co2_vehicule

# Calcul des émissions totales
for type_vehicule, quantite in vehicules_detectes.items():
    if quantite > 0:
        print(f" - {quantite} {type_vehicule}(s)")

if co2_total > 0:
    print(f"Émission totale estimée : {co2_total:.2f} g/km de CO2")
else:
    print("La route est vide, aucune émission détectée !")