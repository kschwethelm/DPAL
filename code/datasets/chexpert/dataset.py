import os

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd

class CheXpert(Dataset):
    def __init__(self, root, size, split='train', transform=None, rgb=True, label_smoothing=0.2, memmap_path=None):
        """ CheXpert Dataset
        
        First, run split_resize.py to generate the resized PNG images and split the dataset.

        Args:
            root (string): Parent directory (e.g. data/chexpert).
            split (string): 'train', 'test', or 'valid'
            transform (callable, optional): Optional transform to be applied
                on a sample.
            rgb (bool): Convert image to RGB
        """
        assert split in ['train', 'test', 'valid'], "Split must be either 'train', 'test', or 'valid'"
        assert os.path.exists(os.path.join(root, f'files_{size}')), f"Resize image folder does not exist: {os.path.join(root, f'files_{size}')}. Run split_resize.py first."

        self.root = root
        self.size = size
        self.split = split
        self.rgb = rgb
        self.transform = transform

        self.label_names, uncertain_pos, uncertain_neg = get_chexpert_constants(split)

        annotation = pd.read_csv(os.path.join(root, f'{split}.csv'))

        if memmap_path:
            num_channels = 3 if rgb else 1
            self.memmap_file = np.memmap(memmap_path, dtype='float32', mode='r', shape=(len(annotation), num_channels, size, size))
        else:
            self.memmap_file = None
            self.paths = annotation["Path"].apply(lambda x: make_new_path(x, root, size, split)).tolist()

        raw_labels = annotation[self.label_names].fillna(0)
        if self.split == "train":
            raw_labels[raw_labels[uncertain_pos]==-1] = 1-label_smoothing
            raw_labels[raw_labels[uncertain_neg]==-1] = label_smoothing
        self.labels = torch.tensor(raw_labels.values, dtype=torch.float32)

    def dataset_name(self):
        return "chexpert"

    def get_label_name(self, idx):
        return self.label_names[idx]

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):

        if self.memmap_file is None:
            img_path = self.paths[idx]
            image = Image.open(img_path)
            if self.rgb:
                image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        else: 
            image = torch.from_numpy(self.memmap_file[idx].copy())

        labels = self.labels[idx]

        return image, labels

def make_new_path(file_path, root, size, split):
    patient_id, study_id, img_id = file_path.split("/")[-3:]
    return f"{root}/files_{size}/{split}/{patient_id}_{study_id}_{img_id}"

def get_chexpert_constants(split):
    # Ensure the overlapping classes are in the same order and the same class id
    if split == "train":
        label_names = [ 
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Lung Opacity",
            "Lung Lesion",
            "Pneumonia",
            "Pneumothorax",
            "Pleural Other",
            "Fracture",
            "Support Devices"
        ]

        uncertain_pos = [
            "Atelectasis",
            "Edema",
            "Pleural Effusion"
        ]
        uncertain_neg = [
            "Cardiomegaly",
            "Consolidation",
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Lung Opacity",
            "Lung Lesion",
            "Pneumonia",
            "Pneumothorax",
            "Pleural Other",
            "Fracture",
            "Support Devices"
        ]

    else:
        label_names = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion"
        ]
        uncertain_pos = None
        uncertain_neg = None

    return label_names, uncertain_pos, uncertain_neg
