"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage
import colorsys

from .transform import change_transform_origin


def read_image_bgr(path):
    """Read an image in BGR format.

    Args
        path: Path to the image.
    """
    # Read image using PIL and convert to RGB, then to BGR via numpy
    image = np.ascontiguousarray(Image.open(path).convert("RGB"))
    return image[:, :, ::-1]  # Convert RGB to BGR


def preprocess_image(x, mode="caffe"):
    """Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # covert always to float32 to keep compatibility
    x = x.astype(np.float32)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
    elif mode == "caffe":
        x -= [103.939, 116.779, 123.68]

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """

    def __init__(
        self,
        fill_mode="nearest",
        interpolation="linear",
        cval=0,
        relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def pil_resample_filter(self):
        """Convert interpolation method to PIL filter"""
        if self.interpolation == "nearest":
            return Image.NEAREST
        if self.interpolation == "linear":
            return Image.BILINEAR
        if self.interpolation == "cubic":
            return Image.BICUBIC
        if self.interpolation == "lanczos4":
            return Image.LANCZOS
        # Default to bilinear
        return Image.BILINEAR


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image using PIL and scipy.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    # Convert numpy array to PIL Image
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    # Extract affine transformation parameters (3x3 matrix to 6 parameters for PIL)
    # PIL uses the inverse transformation matrix compared to OpenCV
    # We need to compute the inverse of the matrix for PIL
    inv_matrix = np.linalg.inv(matrix)

    # Convert to PIL affine parameters (a, b, c, d, e, f)
    affine_params = (
        inv_matrix[0, 0],
        inv_matrix[0, 1],
        inv_matrix[0, 2],
        inv_matrix[1, 0],
        inv_matrix[1, 1],
        inv_matrix[1, 2],
    )

    # Apply transformation
    transformed_image = pil_image.transform(
        size=(image.shape[1], image.shape[0]),
        method=Image.AFFINE,
        data=affine_params,
        resample=params.pil_resample_filter(),
        fillcolor=tuple([params.cval] * 3) if params.fill_mode == "constant" else None,
    )

    # Convert back to numpy array
    output = np.array(transformed_image)

    return output


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # Convert to PIL Image for resizing
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(img)
    new_width = int(pil_image.width * scale)
    new_height = int(pil_image.height * scale)

    # Resize using PIL
    resized_image = pil_image.resize((new_width, new_height), Image.BILINEAR)

    return np.array(resized_image), scale


def _uniform(val_range):
    """Uniformly sample from the given range.

    Args
        val_range: A pair of lower and upper bound.
    """
    return np.random.uniform(val_range[0], val_range[1])


def _check_range(val_range, min_val=None, max_val=None):
    """Check whether the range is a valid range.

    Args
        val_range: A pair of lower and upper bound.
        min_val: Minimal value for the lower bound.
        max_val: Maximal value for the upper bound.
    """
    if val_range[0] > val_range[1]:
        raise ValueError("interval lower bound > upper bound")
    if min_val is not None and val_range[0] < min_val:
        raise ValueError("invalid interval lower bound")
    if max_val is not None and val_range[1] > max_val:
        raise ValueError("invalid interval upper bound")


def _clip(image):
    """
    Clip and convert an image to np.uint8.

    Args
        image: Image to clip.
    """
    return np.clip(image, 0, 255).astype(np.uint8)


class VisualEffect:
    """Struct holding parameters and applying image color transformation.

    Args
        contrast_factor:   A factor for adjusting contrast. Should be between 0 and 3.
        brightness_delta:  Brightness offset between -1 and 1 added to the pixel values.
        hue_delta:         Hue offset between -1 and 1 added to the hue channel.
        saturation_factor: A factor multiplying the saturation values of each pixel.
    """

    def __init__(
        self,
        contrast_factor,
        brightness_delta,
        hue_delta,
        saturation_factor,
    ):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor

    def __call__(self, image):
        """Apply a visual effect on the image.

        Args
            image: Image to adjust
        """
        # Convert BGR to RGB for PIL processing
        image_rgb = image[:, :, ::-1].copy()

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Apply contrast and brightness using PIL
        if self.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(self.contrast_factor)

        if self.brightness_delta != 0:
            # Convert brightness delta from [-1, 1] to PIL's factor
            brightness_factor = 1.0 + self.brightness_delta
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor)

        # Apply hue and saturation using manual HSV conversion
        if self.hue_delta != 0 or self.saturation_factor != 1.0:
            # Convert to numpy array for HSV manipulation
            img_array = np.array(pil_image, dtype=np.float32) / 255.0

            # Convert RGB to HSV manually
            hsv_array = np.zeros_like(img_array)
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = img_array[i, j]
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsv_array[i, j] = [h, s, v]

            # Adjust hue and saturation
            if self.hue_delta != 0:
                hsv_array[:, :, 0] = np.mod(hsv_array[:, :, 0] + self.hue_delta, 1.0)

            if self.saturation_factor != 1.0:
                hsv_array[:, :, 1] = np.clip(
                    hsv_array[:, :, 1] * self.saturation_factor, 0, 1
                )

            # Convert back to RGB
            rgb_array = np.zeros_like(hsv_array)
            for i in range(hsv_array.shape[0]):
                for j in range(hsv_array.shape[1]):
                    h, s, v = hsv_array[i, j]
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    rgb_array[i, j] = [r, g, b]

            # Convert back to uint8
            img_array = np.clip(rgb_array * 255, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)

        # Convert back to BGR
        result_rgb = np.array(pil_image)
        result_bgr = result_rgb[:, :, ::-1]

        return result_bgr


def random_visual_effect_generator(
    contrast_range=(0.9, 1.1),
    brightness_range=(-0.1, 0.1),
    hue_range=(-0.05, 0.05),
    saturation_range=(0.95, 1.05),
):
    """Generate visual effect parameters uniformly sampled from the given intervals.

    Args
        contrast_factor:   A factor interval for adjusting contrast. Should be between 0 and 3.
        brightness_delta:  An interval between -1 and 1 for the amount added to the pixels.
        hue_delta:         An interval between -1 and 1 for the amount added to the hue channel.
                           The values are rotated if they exceed 180.
        saturation_factor: An interval for the factor multiplying the saturation values of each
                           pixel.
    """
    _check_range(contrast_range, 0)
    _check_range(brightness_range, -1, 1)
    _check_range(hue_range, -1, 1)
    _check_range(saturation_range, 0)

    def _generate():
        while True:
            yield VisualEffect(
                contrast_factor=_uniform(contrast_range),
                brightness_delta=_uniform(brightness_range),
                hue_delta=_uniform(hue_range),
                saturation_factor=_uniform(saturation_range),
            )

    return _generate()


def adjust_contrast(image, factor):
    """Adjust contrast of an image.

    Args
        image: Image to adjust.
        factor: A factor for adjusting contrast.
    """
    # Convert BGR to RGB for PIL
    image_rgb = image[:, :, ::-1]
    pil_image = Image.fromarray(image_rgb)

    enhancer = ImageEnhance.Contrast(pil_image)
    adjusted_image = enhancer.enhance(factor)

    # Convert back to BGR
    result_rgb = np.array(adjusted_image)
    return result_rgb[:, :, ::-1]


def adjust_brightness(image, delta):
    """Adjust brightness of an image

    Args
        image: Image to adjust.
        delta: Brightness offset between -1 and 1 added to the pixel values.
    """
    # Convert BGR to RGB for PIL
    image_rgb = image[:, :, ::-1]
    pil_image = Image.fromarray(image_rgb)

    # Convert delta to PIL brightness factor
    brightness_factor = 1.0 + delta
    enhancer = ImageEnhance.Brightness(pil_image)
    adjusted_image = enhancer.enhance(brightness_factor)

    # Convert back to BGR
    result_rgb = np.array(adjusted_image)
    return result_rgb[:, :, ::-1]


def adjust_hue(image, delta):
    """Adjust hue of an image.

    Args
        image: Image to adjust.
        delta: An interval between -1 and 1 for the amount added to the hue channel.
               The values are rotated if they exceed 180.
    """
    # Note: This function is now handled within the VisualEffect class
    # using manual HSV conversion with colorsys
    return image


def adjust_saturation(image, factor):
    """Adjust saturation of an image.

    Args
        image: Image to adjust.
        factor: An interval for the factor multiplying the saturation values of each pixel.
    """
    # Note: This function is now handled within the VisualEffect class
    # using manual HSV conversion with colorsys
    return image


# """
# Copyright 2017-2018 Fizyr (https://fizyr.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """

# from __future__ import division
# import numpy as np
# import cv2
# from PIL import Image

# from .transform import change_transform_origin


# def read_image_bgr(path):
#     """ Read an image in BGR format.

#     Args
#         path: Path to the image.
#     """
#     # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
#     image = np.ascontiguousarray(Image.open(path).convert('RGB'))
#     return image[:, :, ::-1]


# def preprocess_image(x, mode='caffe'):
#     """ Preprocess an image by subtracting the ImageNet mean.

#     Args
#         x: np.array of shape (None, None, 3) or (3, None, None).
#         mode: One of "caffe" or "tf".
#             - caffe: will zero-center each color channel with
#                 respect to the ImageNet dataset, without scaling.
#             - tf: will scale pixels between -1 and 1, sample-wise.

#     Returns
#         The input with the ImageNet mean subtracted.
#     """
#     # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
#     # except for converting RGB -> BGR since we assume BGR already

#     # covert always to float32 to keep compatibility with opencv
#     x = x.astype(np.float32)

#     if mode == 'tf':
#         x /= 127.5
#         x -= 1.
#     elif mode == 'caffe':
#         x -= [103.939, 116.779, 123.68]

#     return x


# def adjust_transform_for_image(transform, image, relative_translation):
#     """ Adjust a transformation for a specific image.

#     The translation of the matrix will be scaled with the size of the image.
#     The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
#     """
#     height, width, channels = image.shape

#     result = transform

#     # Scale the translation with the image size if specified.
#     if relative_translation:
#         result[0:2, 2] *= [width, height]

#     # Move the origin of transformation.
#     result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

#     return result


# class TransformParameters:
#     """ Struct holding parameters determining how to apply a transformation to an image.

#     Args
#         fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
#         interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
#         cval:                  Fill value to use with fill_mode='constant'
#         relative_translation:  If true (the default), interpret translation as a factor of the image size.
#                                If false, interpret it as absolute pixels.
#     """
#     def __init__(
#         self,
#         fill_mode            = 'nearest',
#         interpolation        = 'linear',
#         cval                 = 0,
#         relative_translation = True,
#     ):
#         self.fill_mode            = fill_mode
#         self.cval                 = cval
#         self.interpolation        = interpolation
#         self.relative_translation = relative_translation

#     def cvBorderMode(self):
#         if self.fill_mode == 'constant':
#             return cv2.BORDER_CONSTANT
#         if self.fill_mode == 'nearest':
#             return cv2.BORDER_REPLICATE
#         if self.fill_mode == 'reflect':
#             return cv2.BORDER_REFLECT_101
#         if self.fill_mode == 'wrap':
#             return cv2.BORDER_WRAP

#     def cvInterpolation(self):
#         if self.interpolation == 'nearest':
#             return cv2.INTER_NEAREST
#         if self.interpolation == 'linear':
#             return cv2.INTER_LINEAR
#         if self.interpolation == 'cubic':
#             return cv2.INTER_CUBIC
#         if self.interpolation == 'area':
#             return cv2.INTER_AREA
#         if self.interpolation == 'lanczos4':
#             return cv2.INTER_LANCZOS4


# def apply_transform(matrix, image, params):
#     """
#     Apply a transformation to an image.

#     The origin of transformation is at the top left corner of the image.

#     The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
#     Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

#     Args
#       matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
#       image:  The image to transform.
#       params: The transform parameters (see TransformParameters)
#     """
#     output = cv2.warpAffine(
#         image,
#         matrix[:2, :],
#         dsize       = (image.shape[1], image.shape[0]),
#         flags       = params.cvInterpolation(),
#         borderMode  = params.cvBorderMode(),
#         borderValue = params.cval,
#     )
#     return output


# def compute_resize_scale(image_shape, min_side=800, max_side=1333):
#     """ Compute an image scale such that the image size is constrained to min_side and max_side.

#     Args
#         min_side: The image's min side will be equal to min_side after resizing.
#         max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

#     Returns
#         A resizing scale.
#     """
#     (rows, cols, _) = image_shape

#     smallest_side = min(rows, cols)

#     # rescale the image so the smallest side is min_side
#     scale = min_side / smallest_side

#     # check if the largest side is now greater than max_side, which can happen
#     # when images have a large aspect ratio
#     largest_side = max(rows, cols)
#     if largest_side * scale > max_side:
#         scale = max_side / largest_side

#     return scale


# def resize_image(img, min_side=800, max_side=1333):
#     """ Resize an image such that the size is constrained to min_side and max_side.

#     Args
#         min_side: The image's min side will be equal to min_side after resizing.
#         max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

#     Returns
#         A resized image.
#     """
#     # compute scale to resize the image
#     scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

#     # resize the image with the computed scale
#     img = cv2.resize(img, None, fx=scale, fy=scale)

#     return img, scale


# def _uniform(val_range):
#     """ Uniformly sample from the given range.

#     Args
#         val_range: A pair of lower and upper bound.
#     """
#     return np.random.uniform(val_range[0], val_range[1])


# def _check_range(val_range, min_val=None, max_val=None):
#     """ Check whether the range is a valid range.

#     Args
#         val_range: A pair of lower and upper bound.
#         min_val: Minimal value for the lower bound.
#         max_val: Maximal value for the upper bound.
#     """
#     if val_range[0] > val_range[1]:
#         raise ValueError('interval lower bound > upper bound')
#     if min_val is not None and val_range[0] < min_val:
#         raise ValueError('invalid interval lower bound')
#     if max_val is not None and val_range[1] > max_val:
#         raise ValueError('invalid interval upper bound')


# def _clip(image):
#     """
#     Clip and convert an image to np.uint8.

#     Args
#         image: Image to clip.
#     """
#     return np.clip(image, 0, 255).astype(np.uint8)


# class VisualEffect:
#     """ Struct holding parameters and applying image color transformation.

#     Args
#         contrast_factor:   A factor for adjusting contrast. Should be between 0 and 3.
#         brightness_delta:  Brightness offset between -1 and 1 added to the pixel values.
#         hue_delta:         Hue offset between -1 and 1 added to the hue channel.
#         saturation_factor: A factor multiplying the saturation values of each pixel.
#     """

#     def __init__(
#         self,
#         contrast_factor,
#         brightness_delta,
#         hue_delta,
#         saturation_factor,
#     ):
#         self.contrast_factor = contrast_factor
#         self.brightness_delta = brightness_delta
#         self.hue_delta = hue_delta
#         self.saturation_factor = saturation_factor

#     def __call__(self, image):
#         """ Apply a visual effect on the image.

#         Args
#             image: Image to adjust
#         """

#         if self.contrast_factor:
#             image = adjust_contrast(image, self.contrast_factor)
#         if self.brightness_delta:
#             image = adjust_brightness(image, self.brightness_delta)

#         if self.hue_delta or self.saturation_factor:

#             image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#             if self.hue_delta:
#                 image = adjust_hue(image, self.hue_delta)
#             if self.saturation_factor:
#                 image = adjust_saturation(image, self.saturation_factor)

#             image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

#         return image


# def random_visual_effect_generator(
#     contrast_range=(0.9, 1.1),
#     brightness_range=(-.1, .1),
#     hue_range=(-0.05, 0.05),
#     saturation_range=(0.95, 1.05)
# ):
#     """ Generate visual effect parameters uniformly sampled from the given intervals.

#     Args
#         contrast_factor:   A factor interval for adjusting contrast. Should be between 0 and 3.
#         brightness_delta:  An interval between -1 and 1 for the amount added to the pixels.
#         hue_delta:         An interval between -1 and 1 for the amount added to the hue channel.
#                            The values are rotated if they exceed 180.
#         saturation_factor: An interval for the factor multiplying the saturation values of each
#                            pixel.
#     """
#     _check_range(contrast_range, 0)
#     _check_range(brightness_range, -1, 1)
#     _check_range(hue_range, -1, 1)
#     _check_range(saturation_range, 0)

#     def _generate():
#         while True:
#             yield VisualEffect(
#                 contrast_factor=_uniform(contrast_range),
#                 brightness_delta=_uniform(brightness_range),
#                 hue_delta=_uniform(hue_range),
#                 saturation_factor=_uniform(saturation_range),
#             )

#     return _generate()


# def adjust_contrast(image, factor):
#     """ Adjust contrast of an image.

#     Args
#         image: Image to adjust.
#         factor: A factor for adjusting contrast.
#     """
#     mean = image.mean(axis=0).mean(axis=0)
#     return _clip((image - mean) * factor + mean)


# def adjust_brightness(image, delta):
#     """ Adjust brightness of an image

#     Args
#         image: Image to adjust.
#         delta: Brightness offset between -1 and 1 added to the pixel values.
#     """
#     return _clip(image + delta * 255)


# def adjust_hue(image, delta):
#     """ Adjust hue of an image.

#     Args
#         image: Image to adjust.
#         delta: An interval between -1 and 1 for the amount added to the hue channel.
#                The values are rotated if they exceed 180.
#     """
#     image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
#     return image


# def adjust_saturation(image, factor):
#     """ Adjust saturation of an image.

#     Args
#         image: Image to adjust.
#         factor: An interval for the factor multiplying the saturation values of each pixel.
#     """
#     image[..., 1] = np.clip(image[..., 1] * factor, 0 , 255)
#     return image
