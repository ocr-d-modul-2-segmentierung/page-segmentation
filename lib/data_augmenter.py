import numpy as np
from scipy.misc import imrotate, imresize
from scipy.ndimage.interpolation import shift, rotate

import matplotlib.pyplot as plt

def pad_to_power_shape(img):
    x, y = img.shape

    f = 2 ** 3
    tx = (((x // 2) // 2) // 2) * 8
    ty = (((y // 2) // 2) // 2) * 8

    if x % f != 0:
        px = tx - x + f
        x = x + px
    else:
        px = 0

    if y % f != 0:
        py = ty - y + f
        y = y + py
    else:
        py = 0

    pad = ((px, 0), (py, 0))
    img = np.pad(img, pad, 'edge')

    return img

def augment(binary, image, mask):
    angle = np.random.random() * 10 - 5
    offset_x = np.random.random() * 50 - 25
    offset_y = np.random.random() * 50 - 25
    offset = offset_x, offset_y
    scale_x = 2 ** (np.random.random() * 0.6 - 0.3)
    scale_y = 2 ** (np.random.random() * 0.6 - 0.3)
    scale = scale_x, scale_y

    flip_x = np.random.random() > 0.8 and False
    flip_y = np.random.random() > 0.8 and False

    brightness = np.random.random() * 10
    contrast = 2 ** (np.random.random() * 0.2 - 0.1)

    width = int((np.random.random() * 0.5 + 0.5) * image.shape[0])
    height = int((np.random.random() * 0.5 + 0.5) * image.shape[1])
    offset_x = np.random.randint(0, image.shape[0] - width)
    offset_y = np.random.randint(0, image.shape[1] - height)

    def crop(img):
        return img[offset_x:offset_x+width, offset_y:offset_y+width]

    def apply(img, interp, is_img=False):
        # img = crop(img)
        if is_img:
            img = brightness + contrast * img

        if flip_x:
            img = img[::-1][:]

        if flip_y:
            img = img[:][::-1]


        img = imresize(img, (int(img.shape[0] * scale_x), int(img.shape[1] * scale_y)), interp=interp)
        img = pad_to_power_shape(img)
        return imrotate(img, angle, interp=interp)

    return apply(binary, interp='nearest'), apply(image, interp="bilinear", is_img=True), apply(mask, interp='nearest')



