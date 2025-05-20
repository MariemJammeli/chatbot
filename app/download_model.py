import zipfile
import os
import gdown 

# Liste des fichiers à télécharger (id, nom de sortie)
files = [
    ("159Y8TcaaikeDhwp6GTK9IHYRHfICI1QJ", "app/dataset.zip")
]

for file_id, output in files:
    if os.path.exists(output):
        print(f"{output} existe déjà. Passage au fichier suivant.")
        continue
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Téléchargement de {output} ...")
    gdown.download(url, output, quiet=False)

zip_path = "app/dataset.zip"  
extracted_file = "app/ml.csv"  # 🔧 CORRIGÉ ICI

# Extraire si le fichier CSV n'est pas encore là
if not os.path.exists(extracted_file):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("app")

# Affiche le contenu de 'app' pour debug
print("Contenu du dossier app :", os.listdir("app"))
