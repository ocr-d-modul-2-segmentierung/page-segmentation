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

    def augment_image(self, image, mask):
        assert iaa.Sequential is not None
        ia.seed(np.random.randint(0, 9999999))
        mask = SegmentationMapOnImage(mask, nb_classes=self.classes, shape=image.shape)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            image_aug, segmap_aug = self.augmentor(image=image, segmentation_maps=mask)
            image_aug = image_aug[:, :, 0]
        else:
            image_aug, segmap_aug = self.augmentor(image=image, segmentation_maps=mask)

        return image_aug, segmap_aug.get_arr_int()

    def set_default_augmentor(self):
        seq = iaa.Sequential([
            iaa.CoarseDropout(0.1, size_percent=0.2),
            iaa.Affine(rotate=(-30, 30)),
            iaa.ElasticTransformation(alpha=10, sigma=1)
        ])
        self.augmentor = seq

    def augment_dataset(self, dataset: Dataset) -> Generator:
        for data_idx, d in enumerate(dataset):
            b, i, m = d.binary, d.image, d.mask
            yield self.augment_image(i, m)

    def show_augmented_images(self, row, col, dataset: Dataset, mask=False):
        if mask is True:
            assert row == 2
        from matplotlib import pyplot as plt
        f, ax = plt.subplots(row, col)
        gen = self.augment_dataset(dataset)
        if mask:
            c = 0
            while True:
                try:
                    y, z = next(gen)
                    ax[0][c].imshow(y)
                    ax[1][c].imshow(z)
                    c += 1

                    if c == col:
                        c = 0
                        plt.show()
                        f, ax = plt.subplots(row, col)
                except:
                    plt.show()
                    break
        else:
            c = 0
            r = 0
            while True:
                try:
                    y, z = next(gen)
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

                except:
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

    aug = DataAugmentor(classes=len(image_map))
    aug.set_default_augmentor()
    aug.show_augmented_images(3, 5, train_data)
