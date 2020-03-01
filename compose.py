from collections import Iterable
__all__ = ["Compose4Image", "Compose4Image_Box", "Compose4Image_Point", "Compose4Image_Seg"]


class _Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Iterable)
        self.transforms = transforms

    def __call__(self, img):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Compose4Image(_Compose):
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Compose4Image_Box(_Compose):
    def __call__(self, image, boxes):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class Compose4Image_Seg(_Compose):
    def __call__(self, image, segs):
        for t in self.transforms:
            img, segs = t(img, segs)
        return img, segs


class Compose4Image_Point(_Compose):
    def __call__(self, image, points):
        for t in self.transforms:
            img, points = t(img, points)
        return img, points
