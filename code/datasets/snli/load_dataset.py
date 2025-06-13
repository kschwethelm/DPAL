# Source: https://opacus.ai/tutorials/building_text_classifier
import zipfile
import urllib.request
import os

def download_and_extract(dataset_url, data_dir):
    print("Downloading and extracting...")
    filename = "snli.zip"
    urllib.request.urlretrieve(dataset_url, filename)
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(filename)
    print("Completed!")

# ----------------------------------
if __name__ == '__main__':
    data_dir = "data"
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    download_and_extract(dataset_url=url, data_dir=data_dir)