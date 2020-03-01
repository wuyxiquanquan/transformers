import cv2
import numpy as np

__all__ = ['HorizontalFlip', 'VerticalFlip', 'Rotate', 'RandomRotate', 'Flip_Rotate', 'WarpAffine', 'Translate',
           'Resize', 'Fixed_Ratio_Resize',
           ]


class _Transform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


def hFlip(img):
    return img[:, ::-1]


def vFlip(img):
    return img[::-1, :]


def rotate90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate270(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def keep(img):
    return img


_method1 = [keep, hFlip]
_method2 = [keep, rotate270, rotate90, rotate180]


class HorizontalFlip(_Transform):

    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return np.random.choice([hFlip, keep])(img)

    def __repr__(self):
        return self.__class__.__name__


class VerticalFlip(_Transform):

    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return np.random.choice([vFlip, keep])(img)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(_Transform):
    """
    0, rotate -90 degree
    1, rotate 90 degree
    2, rotate 180 degree
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def __call__(self, img):
        return np.random.choice([_method2[self.mode + 1], _method2[0]])(img)

    def __repr__(self):
        return self.__class__.__name__


class RandomRotate(_Transform):
    """
    """

    def __init__(self):
        super().__init__()
        self.rotates = _method2

    def __call__(self, img):
        return np.random.choice(self.rotates)(img)

    def __repr__(self):
        return self.__class__.__name__


class Flip_Rotate(_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return np.random.choice(_method2)(np.random.choice(_method1)(img))

    def __repr__(self):
        return self.__class__.__name__


class WarpAffine(_Transform):
    """
    """

    def __init__(self, low_degree, high_degree):
        super().__init__()
        assert -180 <= low_degree < high_degree <= 180
        self.range = (low_degree, high_degree)

    def __call__(self, img):
        angle = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
        m1 = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), angle, scale=1)
        return cv2.warpAffine(img, m1, img.shape[:2])

    def __repr__(self):
        return self.__class__.__name__


class Translate(object):

    def __init__(self, translate_x=0.2, translate_y=0.2, random=False):
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.random = random

        assert 0 < self.translate_x and self.translate_y < 1

    def __call__(self, img):
        # Chose a random digit to scale by
        height, width = img.shape[:2]
        if self.random:
            translate_x = np.random.random() * self.translate_x * width
            translate_y = np.random.random() * self.translate_y * height
        else:
            translate_x = self.translate_x * width
            translate_y = self.translate_y * height

        getTranslateMatrix = lambda y, x: np.array([[1, 0, y], [0, 1, x]], dtype=np.float32)

        # translate the image
        return cv2.warpAffine(img, getTranslateMatrix(translate_y, translate_x), (width, height))


class Resize(_Transform):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def __call__(self, img):
        return cv2.resize(img, (self.height, self.width))

    def __repr__(self):
        return self.__class__.__name__


class Fixed_Ratio_Resize(_Transform):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def __call__(self, img):
        img_height, img_width = img.shape[:2]
        scale_height, scale_width = self.height / img_height, self.width / img_width
        if len(img.shape) == 3:
            resized_img = np.zeros((self.height, self.width, img.shape[2]), dtype=img.dtype)
        else:
            resized_img = np.zeros((self.height, self.width), dtype=img.dtype)
        if scale_height < scale_width:
            new_sides = (self.height, int(np.round(scale_height * img_width)))
            translate_x = np.random.randint(0, img_width - new_sides[1] + 1)
            resized_img[:, translate_x:translate_x + new_sides[1]] = cv2.resize(img, new_sides[::-1])
        elif scale_height > scale_width:
            new_sides = (int(np.round(scale_width * img_height)), self.width)
            translate_y = np.random.randint(0, img_height - new_sides[0] + 1)
            resized_img[translate_y:translate_y + new_sides[0], :] = cv2.resize(img, new_sides[
                                                                                     ::-1])  # resize: (width, height)
        else:
            resized_img = cv2.resize(img, (self.height, self.width))
        return resized_img

    def __repr__(self):
        return self.__class__.__name__


# TODO
class RandomCropping(_Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return img

    def __repr__(self):
        return self.__class__.__name__


# TODO
class Blur(_Transform):
    # cv2.GaussianBlur()
    pass


# TODO
class Noise(_Transform):
    pass
