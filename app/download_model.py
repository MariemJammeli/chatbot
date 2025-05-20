import zipfile
import os
import gdown 
# Liste des fichiers à télécharger (id, nom de sortie)
files = [
    ("159Y8TcaaikeDhwp6GTK9IHYRHfICI1QJ", "dataset_chatbot.zip")
]
for file_id, output in files:
    if os.path.exists(output):
        print(f"{output} existe déjà. Passage au fichier suivant.")
        continue
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Téléchargement de {output} ...")
    gdown.download(url, output, quiet=False)
zip_path = "app/dataset.zip"
extracted_file = "ml.csv"  # or whatever the actual CSV file is called

if not os.path.exists(extracted_file):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
import os
print("Contenu du dossier app :", os.listdir("app"))
