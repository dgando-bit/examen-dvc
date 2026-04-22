import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def tune_and_save_model():
    # Configuration des dossiers et fichiers
    input_dir = 'data/processed'
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    train_features_path = os.path.join(input_dir, 'X_train_scaled.csv')
    train_target_path = os.path.join(input_dir, 'y_train.csv')

    if not os.path.exists(train_features_path) or not os.path.exists(train_target_path):
        print("Erreur : Données d'entraînement introuvables. Lancez d'abord les scripts précédents.")
        return

    # Chargement des données
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_target_path).values.ravel() # Convertit en 1D array

    print(f"Lancement du GridSearch sur {X_train.shape[0]} échantillons...")

    rf = RandomForestRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200], 
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'bootstrap': [True]
    }

    # Configuration du GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='r2'
    )

    # Recherche
    grid_search.fit(X_train, y_train)

    # Affichage des résultats
    print("\n" + "="*30)
    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    print(f"Meilleur score R² : {grid_search.best_score_:.4f}")
    print("="*30)

    # Sauvegarde du meilleur modèle
    model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(grid_search.best_estimator_, model_path)
    
    print(f"\nModèle sauvegardé avec succès dans : {model_path}")

if __name__ == "__main__":
    tune_and_save_model()