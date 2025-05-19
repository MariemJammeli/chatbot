import zipfile
import os

zip_path = "dataset.zip"
extracted_file = "dataset_chatbot.csv"  # or whatever the actual CSV file is called

if not os.path.exists(extracted_file):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
