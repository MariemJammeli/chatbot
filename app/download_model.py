import gdown
import os
import zipfile

# Liste des fichiers à télécharger (URL de partage Google Drive, nom de sortie)
files = [
    ("https://drive.google.com/file/d/1cjIbw4Ng44VMqLFYchVTcLA1ZHWbCOwI/view?usp=sharing", "modele_chatbot.zip"),
    ("https://drive.google.com/file/d/159Y8TcaaikeDhwp6GTK9IHYRHfICI1QJ/view?usp=sharing", "dataset_chatbot.zip"),
]

for url, output in files:
    if os.path.exists(output):
        print(f"{output} existe déjà. Passage au fichier suivant.")
        continue

    print(f"Téléchargement de {output} ...")
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)

    if output.endswith(".zip"):
        print(f"Décompression de {output} ...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"{output} décompressé.")
