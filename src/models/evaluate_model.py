import pandas as pd
import os
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

def evaluate_and_predict():
    processed_dir = 'data/processed'
    data_dir = 'data/predictions'
    model_dir = 'models'
    metrics_dir = 'metrics'
    
    os.makedirs(metrics_dir, exist_ok=True)

    # Chemins des fichiers
    model_path = os.path.join(model_dir, 'trained_model.pkl')
    X_test_path = os.path.join(processed_dir, 'X_test_scaled.csv')
    y_test_path = os.path.join(processed_dir, 'y_test.csv')

    # Vérification
    if not all(os.path.exists(p) for p in [model_path, X_test_path, y_test_path]):
        print("Erreur : Fichiers nécessaires manquants (modèle ou données de test).")
        return

    # Chargement du modèle et des données
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Prédictions
    print("Calcul des prédictions sur l'ensemble de test...")
    predictions = model.predict(X_test)

    # Calcul des métriques
    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    scores = {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "r2_score": r2
    }

    # Sauvegarde des métriques en JSON
    scores_path = os.path.join(metrics_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=4)
    
    print(f"Métriques sauvegardées dans : {scores_path}")
    print(f"Score R² : {r2:.4f}")

    # Sauvegarde des prédictions dans data/predictions.csv
    results_df = pd.DataFrame({
        'target_real': y_test,
        'target_predicted': predictions
    })
    
    predictions_path = os.path.join(data_dir, 'predictions.csv')

    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    results_df.to_csv(predictions_path, index=False)
    
    print(f"Tableau des prédictions sauvegardé dans : {predictions_path}")

if __name__ == "__main__":
    evaluate_and_predict()