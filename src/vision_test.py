from ultralytics import YOLO

# On charge le modèle YOLO
print("Chargement de l'oeil de l'IA (YOLO)...")
model = YOLO('yolov8n.pt')

# On lui donne l'image à regarder
chemin_image = "../data/raw/traffic_jam.jpg"

# On lance la détection !
print("Analyse de l'image en cours...")
resultats = model(chemin_image)

# On affiche le résultat de l'analyse
for r in resultats:
    # ouvrir une fenêtre avec ton image et les voitures encadrées
    r.show()  
    
    # resultat
    r.save(filename='resultat_yolo.jpg') 

print("Analyse terminée !")