import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tensorflow import keras
from ocr4all_pixel_classifier.lib.dataset import Dataset
import albumentations as albu
import random
from matplotlib import pyplot as plt
from ocr4all_pixel_classifier.lib.model import default_preprocess
from ocr4all_pixel_classifier.lib.util import gray_to_rgb
from typing import List


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset: Dataset, batch_size=1, n_channels=1, type='train',
                 augmentation: List[albu.OneOf] = None, preprocess=default_preprocess, rgb=False, shuffle=False,
                 foreground_mask=False):
        self.data_set: Dataset = dataset
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.type = type
        self.augmentation = compose(augmentation)
        self.preprocess = preprocess
        self.rgb = rgb
        self.foreground_mask = foreground_mask
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_set.data) / self.batch_size))

    def __getitem__(self, index):
        data = self.data_set.data[index*self.batch_size:(index+1)*self.batch_size]
        image_batch, mask_batch, binary_batch = [], [], []

        for x in data:
            image = x.image
            binary = x.binary
            mask = x.mask
            if binary is None:
                binary = np.full(image.shape, 1, dtype=np.uint8)
                assert (image.dtype == np.uint8)
            if self.rgb:
                image = gray_to_rgb(image)
            if self.foreground_mask:
                mask[binary != 1] = 0
            if self.augmentation is not None and self.type == 'train':
                temp = self.augmentation(image=image, binary=binary, mask=mask)
                image_batch.append(temp['image'])
                binary_batch.append(temp['binary'])
                mask_batch.append(temp['mask'])
            else:
                image_batch.append(image)
                mask_batch.append(mask)
                binary_batch.append(binary)

        image = np.asarray(image_batch) if np.asarray(image_batch).ndim == 4 else np.expand_dims(np.asarray(image_batch), axis=-1)
        binary = np.asarray(binary_batch) if np.asarray(binary_batch).ndim == 4 else np.expand_dims(np.asarray(binary_batch), axis=-1)
        mask = np.asarray(mask_batch) if np.asarray(mask_batch).ndim == 4 else np.expand_dims(np.asarray(mask_batch), axis=-1)

        return ({'input_1': self.preprocess(image), 'input_2': binary}), {'logits': mask}


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = albu.Compose([
        albu.CoarseDropout(p=0.5, max_height=50, max_width=50, max_holes=30),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        #albu.GridDistortion(p=0.3, border_mode=0, value=255, mask_value=[0, 0, 0]),
    ])
    result2 = albu.Compose([
                albu.GaussNoise(p=0.5),
                albu.RandomGamma(p=0.5)
                ])
    result = [
        albu.OneOf([
            result,
            result2,
        ], p=1)
    ]
    return result


def compose(transforms_to_compose):
    if transforms_to_compose is None:
        return None
    # combine all augmentations into one single pipeline
    # convenient if ypu want to add extra targets, e.g. binary input
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ], additional_targets={"binary": "mask"})
    return result


def show_examples(name: str, image: np.ndarray, binary: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")

    plt.subplot(1, 3, 3)
    plt.imshow(binary)
    plt.title(f"Binary: {name}")


def show(name, image, mask, binary, transforms=None) -> None:

    if transforms is not None:
        temp = transforms(image=image, binary=binary, mask=mask)
        image = temp['image']
        binary = temp['binary']
        mask = temp['mask']

    show_examples(name, image, binary, mask)


def show_random(dataset:Dataset, transforms=None) -> None:
    import os
    length = len(dataset.data)
    index = random.randint(0, length - 1)
    image = dataset.data[index].image
    mask = dataset.data[index].mask
    binary = dataset.data[index].binary
    name = os.path.basename(dataset.data[index].image_path)

    show(name, image, mask, binary, transforms)
    plt.show()


if __name__ == "__main__":
    from ocr4all_pixel_classifier.lib.dataset import DatasetLoader
    from ocr4all_pixel_classifier.scripts.generate_image_map import load_image_map_from_file
    import os

    dataset_dir = '/home/alexander/Dokumente/virutal_stafflines/'
    image_map = load_image_map_from_file(os.path.join(dataset_dir, 'image_map.json'))
    dataset_loader = DatasetLoader(8, color_map=image_map)
    train_data = dataset_loader.load_data_from_json(
        [os.path.join(dataset_dir, 't.json')], "train")
    x = DataGenerator(train_data, augmentation=hard_transforms())
    t = x.__getitem__(5)
    z = t[0].get('input_1')[0, :, :, 0]
    train_transforms = compose([
        hard_transforms(),
    ])
    valid_transforms = compose([pre_transforms()])

    show_transforms = compose([hard_transforms()])
    while True:
        show_random(train_data, transforms=show_transforms)