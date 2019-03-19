def calculate_padding(image, scaling_factor):
    def scale(i, f):
        return (f - i % f) % f

    x, y = image.shape
    px = scale(x, scaling_factor)
    py = scale(y, scaling_factor)

    pad = ((px, 0), (py, 0))
    return pad
