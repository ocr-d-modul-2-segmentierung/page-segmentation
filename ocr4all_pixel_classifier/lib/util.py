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


def preserving_resize(image: np.ndarray, target_shape) -> np.ndarray:
    """
    Resizes given image to target_shape while preserving values (i.e. no anti aliasing or range change)
    :param image: input image
    :param target_shape: target_shape
    :return: resized array
    """
    from skimage.transform import resize
    return resize(image, target_shape, order=0, anti_aliasing=False, preserve_range=True)
