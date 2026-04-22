import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_processed_data():
    input_path = 'data/raw/raw.csv'
    output_dir = 'data/processed'
    target_col = 'silica_concentrate'

    if not os.path.exists(input_path):
        print(f"Erreur : Le fichier {input_path} est absent. Lancez d'abord import_raw_data.py")
        return

    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)

    # Séparation Features (X) et Cible (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Sauvegarde de 4 datasets
    datasets = {
        'X_train.csv': X_train,
        'X_test.csv': X_test,
        'y_train.csv': y_train,
        'y_test.csv': y_test
    }
    for name, data in datasets.items():
        data.to_csv(os.path.join(output_dir, name), index=False)
    
    print(f"Split terminé. 4 fichiers créés dans {output_dir}")

if __name__ == "__main__":
    split_processed_data()