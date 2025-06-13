import os
from math import floor
from tqdm import tqdm

import numpy as np
from PIL import Image
import pandas as pd

def resize_short(im, target_size):
    width, height = im.size
    if width == height:
        size = (target_size, target_size)
    elif width > height:
        ratio = float(width) / float(height)
        newwidth = int(floor(ratio * target_size))
        size = (newwidth, target_size)

    elif height > width:
        ratio = float(height) / float(width)
        newheight = int(floor(ratio * target_size))
        size = (target_size, newheight)

    im = im.resize(size, resample=Image.BICUBIC)
    return im

def resize_image(im, target_size, squared=True):
    h, w = im.size
    min_size = min(h, w)
    if min_size>320:
        resize_short(im, 320)

    if squared:
        """ Resize and center crop like https://github.com/openai/improved-diffusion """
        im.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*im.size) >= 2 * target_size:
            im = im.resize(
                tuple(x // 2 for x in im.size), resample=Image.BOX
            )

        scale = target_size / min(*im.size)
        im = im.resize(
            tuple(round(x * scale) for x in im.size), resample=Image.BICUBIC
        )

        arr = np.array(im.convert("RGB"))
        crop_y = (arr.shape[0] - target_size) // 2
        crop_x = (arr.shape[1] - target_size) // 2
        arr = arr[crop_y : crop_y + target_size, crop_x : crop_x + target_size]

        im = Image.fromarray(arr)

    else:
        """ Resize shorter side to target_size (keep aspect ratio) """
        resize_short(im, target_size)

    return im

def iterate_images(df, base_folder, target_size, split):

    print(f"Saving images to: {base_folder}/files_{target_size}/{split}/<patient_id>_<study_id>_<img_id>.jpg")

    num_imgs = 0
    num_imgs_skipped = 0
    for i, row in tqdm(df.iterrows()):
        file_path = row['Path']
        new_file_path = make_new_path(file_path, split)

        file_path = os.path.join(base_folder, file_path)

        # Save new image
        target_folder = f"{base_folder}/files_{target_size}/{split}"
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        new_path = f"{base_folder}/files_{target_size}/{new_file_path}"

        if not os.path.exists(new_path):
            num_imgs += 1
            im = Image.open(file_path)
            im = resize_image(im, target_size)
            im.save(new_path)
        else:
            num_imgs_skipped += 1
            print(f"Skipping {new_path} as it already exists")
        
    print(f"Processed {num_imgs} images, skipped {num_imgs_skipped} images")

def make_new_path(file_path, split):
    patient_id, study_id, img_id = file_path.split("/")[-3:]
    return f"{split}/{patient_id}_{study_id}_{img_id}"


def remove_parent_folder(df, parent_folder, out_path=None):
    df['Path'] = df['Path'].apply(lambda x: x.replace(parent_folder, ""))
    if out_path:
        df.to_csv(out_path, index=False)
    return df

def main():
    target_size = 192
    data_path = "PATH_TO_PARENT_FOLDER"

    base_folder = os.path.join(data_path, "chexpert")

    df_train = pd.read_csv(os.path.join(base_folder, f'train.csv'))
    df_train = remove_parent_folder(df_train, "CheXpert-v1.0-small/", out_path=os.path.join(base_folder, f'train.csv'))
    df_valid = pd.read_csv(os.path.join(base_folder, f'valid.csv'))
    df_valid = remove_parent_folder(df_valid, "CheXpert-v1.0-small/", out_path=os.path.join(base_folder, f'valid.csv'))
    df_test = pd.read_csv(os.path.join(base_folder, f'test_labels.csv'))
    df_test = remove_parent_folder(df_test, "CheXpert-v1.0-small/", out_path=os.path.join(base_folder, f'test.csv'))

    iterate_images(df_train, base_folder, target_size, split="train")
    iterate_images(df_valid, base_folder, target_size, split="valid")
    iterate_images(df_test, base_folder, target_size, split="test")
    
    
if __name__ == '__main__':
    main()