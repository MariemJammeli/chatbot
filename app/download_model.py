import gdown
import os
import zipfile

# Liste des fichiers à télécharger (ID Google Drive, nom de sortie)
files = [
    ("159Y8TcaaikeDhwp6GTK9IHYRHfICI1QJ", "dataset_chatbot.zip")
]

for file_id, output in files:
    if os.path.exists(output):
        print(f"{output} existe déjà. Passage au fichier suivant.")
        continue

    print(f"Téléchargement de {output} ...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=output, quiet=False)

    if output.endswith(".zip"):
        print(f"Décompression de {output} ...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"{output} décompressé.")
