import numpy as np
from abc import ABC, abstractmethod
from scipy.misc import imrotate, imresize

from pagesegmentation.lib.image_ops import calculate_padding


class DataAugmenterBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply(self, binary, image, mask):
        return binary, image, mask


class DefaultAugmenter(DataAugmenterBase):
    def __init__(self, angle=None, offset=None, scale=None, flip=None, brightness=None, contrast=None):
        super().__init__()
        self.angle = angle
        self.offset = offset
        self.scale = scale
        self.flip = flip if flip else (0, 0)
        self.brightness = brightness
        self.contrast = contrast

    def apply(self, binary, image, mask):
        if self.angle:
            mi, ma = self.angle
            angle = np.random.random() * (ma - mi) + mi
        else:
            angle = 0

        if self.offset:
            # x_mi, x_ma, y_mi, y_ma = self.offset
            # offset_x = np.random.random() * (x_ma - x_mi) + x_mi
            # offset_y = np.random.random() * (y_ma - y_mi) + y_mi
            width = int((np.random.random() * 0.5 + 0.5) * image.shape[0])
            height = int((np.random.random() * 0.5 + 0.5) * image.shape[1])
            offset_x = np.random.randint(0, image.shape[0] - width)
            offset_y = np.random.randint(0, image.shape[1] - height)
        else:
            offset_x, offset_y = 0, 0

        if self.scale:
            x_mi, x_ma, y_mi, y_ma = self.scale
            scale_x = 2 ** (np.random.random() * (x_ma - x_mi) + x_mi)
            scale_y = 2 ** (np.random.random() * (y_ma - y_mi) + y_mi)
        else:
            scale_x, scale_y = 1, 1

        flip_x = False if self.flip[0] == 0 else np.random.random() < self.flip[0]
        flip_y = False if self.flip[1] == 0 else np.random.random() < self.flip[1]

        brightness = 0 if not self.brightness else np.random.random() * self.brightness
        contrast = 1 if not self.contrast else 2 ** (np.random.random() * self.contrast - self.contrast / 2)

        # noinspection PyUnusedLocal
        def crop(img):
            return img[offset_x:offset_x + width, offset_y:offset_y + width]

        def apply(img, interp, is_img=False):
            # img = crop(img)
            if is_img:
                img = brightness + contrast * img

            if flip_x:
                img = img[::-1][:]

            if flip_y:
                img = img[:][::-1]

            if scale_x != 1 or scale_y != 1:
                img = imresize(img, (int(img.shape[0] * scale_x), int(img.shape[1] * scale_y)), interp=interp)

            pad = calculate_padding(img, 2 ** 3)
            img = np.pad(img, pad, 'edge')

            return imrotate(img, angle, interp=interp)

        return \
            apply(binary, interp='nearest'), \
            apply(image, interp="bilinear", is_img=True), \
            apply(mask, interp='nearest')
