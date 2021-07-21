import cv2
import numpy as np
from PIL import Image

from transform import change_transform_origin

SIZE = (640, 480)


def load_data(image_path, label_path, train=True):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(label_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    conf = mask

    return image, mask, conf


def val_resize_crop_padding(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    target_dim = SIZE  # width x height
    scale_factor = float(target_dim[0]) / image.shape[1]
    # resize to fit target width
    image = cv2.resize(image, (target_dim[0], int(image.shape[0] * scale_factor)), interpolation=cv2.INTER_NEAREST)
    if image.shape[0] < target_dim[1]:
        pad_size = abs(target_dim[1] - image.shape[0]) // 2
        image = cv2.copyMakeBorder(image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    image = cv2.resize(image, target_dim, interpolation=cv2.INTER_NEAREST)
    return image


def normalize_image(x):
    """ Normalize the image based on training mean and std.
    Args
        x: np.array of shape (None, None, depth)
    Returns
        The data normalized
    """
    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)
    x /= 255.0
    # h, w, d = x.shape
    # for i in range(d):
    #     x[..., i] -= MEAN[i]
    #     x[..., i] /= STD[i]
    # x=normalizeMeanVariance(x)
    return x


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """

    def __init__(
            self,
            fill_mode='constant',
            interpolation='nearest',
            cval=0,
            relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    Args:
        image:        an image
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval,
    )

    return output
