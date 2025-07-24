# Source: https://opacus.ai/tutorials/building_text_classifier
import zipfile
import urllib.request
import os

def download_and_extract(dataset_url="https://nlp.stanford.edu/projects/snli/snli_1.0.zip", data_dir="datasets"):
    filename = "snli.zip"
    urllib.request.urlretrieve(dataset_url, filename)
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(filename)