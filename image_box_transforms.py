import cv2
import torch
from itertools import product
from torch import functional as F

import cv2
import numpy as np

__all__ = ['HorizontalFlip', 'VerticalFlip', 'Rotate', 'RandomRotate', 'Flip_Rotate', 'WarpAffine', 'Translate',
           'Resize', 'Fixed_Ratio_Resize',
           ]


class _Transform_BOX(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        """

        :param img:
        :param boxes: a box is [x, y, w, h, ...], at least len(box) >= 4
        :return:
        """
        assert len(boxes.shape) == 2 and boxes.shape[1] >= 4
        boxes = np.array(boxes, dtype=np.float32)
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


def transform_boxes(m1, boxes):
    xmins, ymins, ws, hs = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xmaxs = xmins + ws + 1
    ymaxs = ymins + hs + 1
    points = list()
    for ps in product([xmins, xmaxs], [ymins, ymaxs], [1, ]):
        points.append(np.dot(m1, np.array(ps)))
    points = np.vstack(points).astype(np.int32)
    return cv2.boundingRect(points)


def hFlip(img, boxes):
    return img[:, ::-1], img.shape[1] - boxes[:, 0] - 1


def vFlip(img, boxes):
    return img[::-1, :], img.shape[0] - boxes[:, 1] - 1


def rotate90(img, boxes):
    m1 = cv2.getRotationMatrix2D([img.shape[0] // 2, img.shape[1] // 2], 90, 1)
    return cv2.warpAffine(img, m1, dsize=(img.shape[1], img.shape[0])), transform_boxes(m1, boxes)


def rotate180(img, boxes):
    m1 = cv2.getRotationMatrix2D([img.shape[0] // 2, img.shape[1] // 2], 180, 1)
    return cv2.warpAffine(img, m1, dsize=(img.shape[1], img.shape[0])), transform_boxes(m1, boxes)


def rotate270(img, boxes):
    m1 = cv2.getRotationMatrix2D([img.shape[0] // 2, img.shape[1] // 2], -90, 1)
    return cv2.warpAffine(img, m1, dsize=(img.shape[1], img.shape[0])), transform_boxes(m1, boxes)


def keep(img, boxes):
    return img, boxes


_method1 = [keep, hFlip]
_method2 = [keep, rotate270, rotate90, rotate180]


class HorizontalFlip(_Transform_BOX):

    def __init__(self):
        super().__init__()

    def __call__(self, img, boxes):
        return np.random.choice([hFlip, keep])(img, boxes)

    def __repr__(self):
        return self.__class__.__name__


class VerticalFlip(_Transform_BOX):

    def __init__(self):
        super().__init__()

    def __call__(self, img, boxes):
        return np.random.choice([vFlip, keep])(img, boxes)

    def __repr__(self):
        return self.__class__.__name__


class Rotate(_Transform_BOX):
    """
    0, rotate -90 degree
    1, rotate 90 degree
    2, rotate 180 degree
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def __call__(self, img, boxes):
        return np.random.choice([_method2[self.mode + 1], _method2[0]])(img, boxes)

    def __repr__(self):
        return self.__class__.__name__


class RandomRotate(_Transform_BOX):
    """
    """

    def __init__(self):
        super().__init__()
        self.rotates = _method2

    def __call__(self, img, boxes):
        return np.random.choice(self.rotates)(img, boxes)

    def __repr__(self):
        return self.__class__.__name__


class Flip_Rotate(_Transform_BOX):
    def __init__(self):
        super().__init__()

    def __call__(self, img, boxes):
        return np.random.choice(_method2)(np.random.choice(_method1)(img, boxes))

    def __repr__(self):
        return self.__class__.__name__


class WarpAffine(_Transform_BOX):
    """
    """

    def __init__(self, low_degree, high_degree):
        super().__init__()
        assert -180 <= low_degree < high_degree <= 180
        self.range = (low_degree, high_degree)

    def __call__(self, img, boxes):
        angle = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]
        m1 = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), angle, scale=1)
        return cv2.warpAffine(img, m1, img.shape[:2]), transform_boxes(m1, boxes)

    def __repr__(self):
        return self.__class__.__name__


class Translate(object):

    def __init__(self, translate_x=0.2, translate_y=0.2, random=False):
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.random = random

        assert 0 < self.translate_x and self.translate_y < 1

    def __call__(self, img, boxes):
        # Chose a random digit to scale by
        height, width = img.shape[:2]
        if self.random:
            translate_x = np.random.random() * self.translate_x * width
            translate_y = np.random.random() * self.translate_y * height
        else:
            translate_x = self.translate_x * width
            translate_y = self.translate_y * height

        getTranslateMatrix = lambda y, x: np.array([[1, 0, y], [0, 1, x]], dtype=np.float32)
        m1 = getTranslateMatrix(translate_y, translate_x)
        # translate the image
        return cv2.warpAffine(img, m1, (width, height)), transform_boxes(m1, boxes)


class Resize(_Transform_BOX):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def __call__(self, img, boxes):
        height_scale, width_scale = self.height / img.shape[0], self.width / img.shape[1]
        boxes[:, 0], boxes[:, 2] = int(boxes[:, 0] * width_scale), int(boxes[:, 2] * width_scale)
        boxes[:, 1], boxes[:, 3] = int(boxes[:, 1] * height_scale), int(boxes[:, 3] * height_scale)
        return cv2.resize(img, None, fx=width_scale, fy=height_scale), boxes

    def __repr__(self):
        return self.__class__.__name__


class Fixed_Ratio_Resize(_Transform_BOX):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def __call__(self, img, boxes):
        img_height, img_width = img.shape[:2]
        scale_height, scale_width = self.height / img_height, self.width / img_width
        if len(img.shape) == 3:
            resized_img = np.zeros((self.height, self.width, img.shape[2]), dtype=img.dtype)
        else:
            resized_img = np.zeros((self.height, self.width), dtype=img.dtype)
        if scale_height < scale_width:
            new_sides = (self.height, int(np.round(scale_height * img_width)))
            translate_x = np.random.randint(0, img_width - new_sides[1] + 1)
            resized_img[:, translate_x:translate_x + new_sides[1]] = cv2.resize(img, None, fx=scale_height,
                                                                                fy=scale_height)
            boxes[:, 0], boxes[:, 2] = int(boxes[:, 0] * scale_height), int(boxes[:, 2] * scale_height)
            boxes[:, 1], boxes[:, 3] = int(boxes[:, 1] * scale_height), int(boxes[:, 3] * scale_height)
            boxes[:, 0] += translate_x
        elif scale_height > scale_width:
            new_sides = (int(np.round(scale_width * img_height)), self.width)
            translate_y = np.random.randint(0, img_height - new_sides[0] + 1)
            resized_img[translate_y:translate_y + new_sides[0], :] = cv2.resize(img, None, fx=scale_width,
                                                                                fy=scale_width)  # resize: (width, height)
            boxes[:, 0], boxes[:, 2] = int(boxes[:, 0] * scale_width), int(boxes[:, 2] * scale_width)
            boxes[:, 1], boxes[:, 3] = int(boxes[:, 1] * scale_width), int(boxes[:, 3] * scale_width)
            boxes[:, 1] += translate_y
        else:
            scale = self.height / img_height
            resized_img = cv2.resize(img, None, fx=scale, fy=scale)
            boxes[:, 0], boxes[:, 2] = int(boxes[:, 0] * scale), int(boxes[:, 2] * scale)
            boxes[:, 1], boxes[:, 3] = int(boxes[:, 1] * scale), int(boxes[:, 3] * scale)

        return resized_img

    def __repr__(self):
        return self.__class__.__name__


# TODO
class RandomCropping(_Transform_BOX):
    def __init__(self):
        super().__init__()

    def __call__(self, img, boxes):
        return img

    def __repr__(self):
        return self.__class__.__name__


# TODO
class Blur(_Transform_BOX):
    # cv2.GaussianBlur()
    pass


# TODO
class Noise(_Transform_BOX):
    pass
