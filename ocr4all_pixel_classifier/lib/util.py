import numpy as np


def gray_to_rgb(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        img = img[..., np.newaxis]
        return np.concatenate(3 * (img,), axis=-1)
    else:
        return img


def image_to_batch(img):
    if len(img.shape) == 2:
        return np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
    else:

        assert img.shape != 3
        return np.expand_dims(img, axis=0)

