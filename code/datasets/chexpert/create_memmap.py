import os

import torchvision.transforms as T

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def make_new_path(file_path, root, size, split):
    patient_id, study_id, img_id = file_path.split("/")[-3:]
    return f"{root}/files_{size}/{split}/{patient_id}_{study_id}_{img_id}"


def create_memmap_file(root, size, split, transform, rgb):
    assert split in ["train", "test", "valid"], (
        "Split must be either 'train', 'test', or 'valid'"
    )
    assert os.path.exists(os.path.join(root, f"files_{size}")), (
        f"Resize image folder does not exist: {os.path.join(root, f'files_{size}')}. Run split_resize.py first."
    )

    annotation = pd.read_csv(os.path.join(root, f"{split}.csv"))
    paths = (
        annotation["Path"].apply(lambda x: make_new_path(x, root, size, split)).tolist()
    )

    for i, img_path in tqdm(enumerate(paths)):
        image = Image.open(img_path)
        if rgb:
            image = image.convert("RGB")
        image = transform(image)

        if i == 0:
            shape = (len(paths), *image.shape)
            memmap = np.memmap(
                f"{root}/memmap_{size}_{split}.memmap",
                dtype="float32",
                mode="w+",
                shape=shape,
            )
        memmap[i] = image.numpy()

    memmap.flush()


def main():
    data_root = "PATH_TO_DATA"

    base_folder = os.path.join(data_root, "chexpert")

    size = 192
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    create_memmap_file(base_folder, size, "train", transforms, rgb=True)
    create_memmap_file(base_folder, size, "valid", transforms, rgb=True)
    create_memmap_file(base_folder, size, "test", transforms, rgb=True)


if __name__ == "__main__":
    main()
