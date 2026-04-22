import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def scale_data():
    input_dir = 'data/processed'
    train_path = os.path.join(input_dir, 'X_train.csv')
    test_path = os.path.join(input_dir, 'X_test.csv')
    
    # Vérification
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Erreur : Les fichiers X_train/X_test sont introuvables dans {input_dir}.")
        print("Assurez-vous d'avoir exécuté split_data.py d'abord.")
        return

    # Chargement des données
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    
    #print(f"Normalisation de {X_train.shape[1]} caractéristiques...")

    # Initialisation du scaler
    scaler = StandardScaler()

    # Fit
    X_train_scaled_values = scaler.fit_transform(X_train)
    X_test_scaled_values = scaler.transform(X_test)

    # Reconversion en DataFrame et sauvegarde
    X_train_scaled = pd.DataFrame(X_train_scaled_values, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_values, columns=X_test.columns)

    X_train_scaled.to_csv(os.path.join(input_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(input_dir, 'X_test_scaled.csv'), index=False)

    print(f"Succès de la normalisation des données")

if __name__ == "__main__":
    scale_data()