import gdown
import os

# Liste des fichiers à télécharger (id, nom de sortie)
files = [
    ("1cjIbw4Ng44VMqLFYchVTcLA1ZHWbCOwI", "modele_chatbot.zip"),
    ("159Y8TcaaikeDhwp6GTK9IHYRHfICI1QJ", "dataset_chatbot.zip"),
]


for file_id, output in files:
    if os.path.exists(output):
        print(f"{output} existe déjà. Passage au fichier suivant.")
        continue
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Téléchargement de {output} ...")
    gdown.download(url, output, quiet=False)
