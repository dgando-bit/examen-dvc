import os
import requests

def download_raw_data():
    raw_dir = 'data/raw'
    raw_file_path = os.path.join(raw_dir, 'raw.csv')
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

    os.makedirs(raw_dir, exist_ok=True)

    if not os.path.exists(raw_file_path) or os.path.getsize(raw_file_path) == 0:
        print(f"Téléchargement des données depuis {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(raw_file_path, 'wb') as f:
                f.write(response.content)
            print("Données brutes importées avec succès.")
        except Exception as e:
            print(f"Erreur lors de l'import : {e}")
    else:
        print("Les données existent déjà localement.")

if __name__ == "__main__":
    download_raw_data()