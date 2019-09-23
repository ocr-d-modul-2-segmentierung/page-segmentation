from imgaug.augmentables.kps import KeypointsOnImage
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
from pagesegmentation.lib.dataset import Dataset
from typing import Generator
from imgaug.augmentables.segmaps import SegmentationMapOnImage


class DataAugmentor:

    def __init__(self, data_augmentor: iaa.Augmenter =None, classes=-1):
        self.augmentor: iaa.Augmenter = data_augmentor
        self.classes: int = classes
        assert self.classes != -1

    def augment_image(self, image, mask, binary):
        assert iaa.Sequential is not None
        ia.seed(np.random.randint(0, 9999999))
        mask = SegmentationMapOnImage(mask, nb_classes=self.classes, shape=image.shape)
        augmentor_det = self.augmentor.to_deterministic()

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            images_aug, segmap_aug = augmentor_det(image=image, segmentation_maps=mask)
            image_aug = images_aug[:, :, 0]
        else:
            binary = np.expand_dims(binary, axis=-1)
            image_aug, segmap_aug = augmentor_det(image=image, segmentation_maps=mask)

        binary = np.expand_dims(binary, axis=-1)
        bin_aug = augmentor_det(image=binary)
        bin_aug = bin_aug[:, :, 0]
        bin_aug[bin_aug > 0] = 1
        return image_aug, segmap_aug.get_arr_int(), bin_aug



    @staticmethod
    def get_default_augmentor(binary=False):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally
                iaa.Flipud(0.5),  # vertically
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode='constant',
                    pad_cval=0
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    rotate=(-2.5, 2.5),
                    shear=(-1.5, 1.5),
                    order=0 if binary else 1,
                    cval=0,
                    mode='constant'
                )),
                # execute 0 to 3 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong

                iaa.SomeOf((0, 3),
                           [
                               iaa.PiecewiseAffine(scale=(0.001, 0.01)),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.CoarseDropout((0.003, 0.015), size_percent=(0.02, 0.05)),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return seq

    def set_augmentor(self, augmentor: iaa.Augmenter):
        self.augmentor = augmentor

    def augment_dataset(self, dataset: Dataset) -> Generator:
        for data_idx, d in enumerate(dataset):
            b, i, m = d.binary, d.image, d.mask
            yield self.augment_image(i, m, b)

    def show_augmented_images(self, row, col, dataset: Dataset, mask=True):
        if mask is True:
            assert row == 2
        from matplotlib import pyplot as plt
        f, ax = plt.subplots(row, col)
        gen = self.augment_dataset(dataset)
        if mask:
            c = 0
            while True:
                try:
                    y, z, b = next(gen)
                    ax[0][c].imshow(y)
                    ax[1][c].imshow(z)
                    c += 1

                    if c == col:
                        c = 0
                        plt.show()
                        f, ax = plt.subplots(row, col)
                except StopIteration:
                    plt.show()
                    break
        else:
            c = 0
            r = 0
            while True:
                try:
                    y, z, b = next(gen)
                    ax[r][c].imshow(y)
                    c += 1
                    if c % col == 0:
                        r += 1
                        c = 0

                    if c*r == col*row:
                        c = 0
                        r = 0
                        plt.show()
                        f, ax = plt.subplots(row, col)

                except StopIteration:
                    plt.show()
                    break


if __name__ == "__main__":
    from pagesegmentation.lib.dataset import DatasetLoader
    from pagesegmentation.scripts.generate_image_map import load_image_map_from_file
    import os
    dataset_dir = '/home/alexander/Dokumente/virutal_stafflines/'
    image_map = load_image_map_from_file(os.path.join(dataset_dir, 'image_map.json'))
    dataset_loader = DatasetLoader(8, color_map=image_map)
    train_data = dataset_loader.load_data_from_json(
        [os.path.join(dataset_dir, 't.json')], "train")
    test_data = dataset_loader.load_data_from_json(
        [os.path.join(dataset_dir, 't.json')], "test")
    eval_data = dataset_loader.load_data_from_json(
        [os.path.join(dataset_dir, 't.json')], "eval")

    aug = DataAugmentor(DataAugmentor.get_default_augmentor(), classes=len(image_map))
    aug.show_augmented_images(2, 5, train_data)
