import zipfile
import os

zip_path = "app/dataset.zip"
extracted_file = "ml.csv"  # or whatever the actual CSV file is called

if not os.path.exists(extracted_file):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
