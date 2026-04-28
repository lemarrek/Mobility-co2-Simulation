import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings

# message erreur scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

class CO2Predictor:
    def __init__(self, data_path):
        """
        Initialise l'IA en chargeant le fichier et en entraînant le modèle.
        """
        # Charger et nettoyer les données
        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.strip() 

        # Identifier les colonnes de manière robuste
        col_moteur = [c for c in self.df.columns if 'Engine Size' in c][0]
        col_cylindres = [c for c in self.df.columns if 'Cylinders' in c][0]
        col_conso = [c for c in self.df.columns if 'Fuel Consumption Comb (L/100 km)' in c][0]
        col_co2 = [c for c in self.df.columns if 'CO2 Emissions' in c][0]

        # Préparer les données
        X = self.df[[col_moteur, col_cylindres, col_conso]]
        y = self.df[col_co2]

        # Entraîner le modèle
        self.modele = LinearRegression()
        self.modele.fit(X, y)
        print("[Système prêt] Modèle IA chargé et entraîné avec succès !")

    def predict_vehicle(self, engine_size, cylinders, fuel_consumption):
        """
        Reçoit les caractéristiques d'UN véhicule et retourne son émission de CO2.
        """
        nouvelle_donnee = [[engine_size, cylinders, fuel_consumption]]
        prediction = self.modele.predict(nouvelle_donnee)
        return prediction[0]

# test
if __name__ == "__main__":
    chemin_fichier = "../data/raw/CO2 Emissions_Canada.xls" 

    print("Initialisation...")
    # On allume notre moteur d'IA
    simulateur = CO2Predictor(chemin_fichier)

    # On simule l'arrivée d'une donnée en temps réel
    print("\n[Simulation] Nouveau véhicule détecté sur la route...")
    co2_estime = simulateur.predict_vehicle(engine_size=3.5, cylinders=6, fuel_consumption=11.1)
    
    print(f"Émission calculée en direct : {co2_estime:.2f} g/km de CO2")