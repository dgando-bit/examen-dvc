import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_final_model():
    input_dir = 'data/processed_data'
    model_dir = 'models'
    best_model_path = os.path.join(model_dir, 'best_model.pkl')
    final_model_path = os.path.join(model_dir, 'trained_model.pkl')

    # Vérification
    if not os.path.exists(best_model_path):
        print("Erreur : Le modèle issu du GridSearch est introuvable.")
        return

    # Chargement du meilleur modèle pour récupérer ses paramètres
    best_estimator = joblib.load(best_model_path)
    params = best_estimator.get_params()
 
    # Chargement des données d'entraînement (scalées)
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv')).values.ravel()


    # Entraînement du modèle final avec les meilleurs paramètres
    print("Entraînement du modèle final en cours...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Sauvegarde du modèle
    joblib.dump(model, final_model_path)
    
    print(f"Modèle entraîné sauvegardé : {final_model_path}")

if __name__ == "__main__":
    train_final_model()