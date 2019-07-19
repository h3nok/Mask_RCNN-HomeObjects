import os
from typing import Any, Union

from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import math
import numpy as np
import io
import glob
import cv2


def _list_to_file(file_path, items_list) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'w+') as f:
        for item in items_list:
            f.write("{}\n".format(item))


class Augmenter(object):
    frame = None

    def __init__(self, x: tf.Tensor):
        """Initialize object with frame data
        :param x:
        """
        self.frame = x

    def rotate(self, degrees: int = 90) -> tf.Tensor:
        """Rotation augmentation
        Args:
            x: Image

        Returns:
            Augmented image
            :param x:
            :param degrees:
        """
        if degrees == 90:
            rotated = tf.image.rot90(self.frame)
            return rotated

        rotated = tf.contrib.image.rotate(self.frame, math.radians(degrees))
        return rotated

    def flip_up_down(self) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip
        Returns:
            Augmented image
        """
        return tf.image.flip_up_down(self.frame)

    def flip_left_right(self) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """
        return tf.image.flip_left_right(self.frame)

    def central_crop(self, fraction=0.7) -> tf.Tensor:
        """ crop from the center
        :param x: image in tf.Tensor format
        :param fraction: amount of crop from center
        :return: cropped image
        """
        return tf.image.central_crop(self.frame, central_fraction=fraction)

    def resize(self, size=(224, 224)):
        """ resize image
        :param x: image
        :param size: (new width, new height)
        :return: resized image
        """
        return tf.image.resize(self.frame, size)

    def adjust_contrast(self, contrast_factor=0.7) -> tf.Tensor:
        """ adjust contrast
        :param x: image
        :param contrast_factor: amount of contrast shift
        :return: adjusted image
        """
        return tf.image.adjust_contrast(self.frame, contrast_factor=contrast_factor)

    def adjust_hue(self, delta=0.4) -> tf.Tensor:
        """ adjust hue
        :param x:
        :param delta:
        :return:
        """
        return tf.image.adjust_hue(self.frame, delta=delta)

    def adjust_saturation(self, factor=10):
        """ adjust saturation
        :param x:
        :param factor:
        :return:
        """
        return tf.image.adjust_saturation(self.frame, factor)

    def adjust_brightness(self, delta=0.2) -> tf.Tensor:
        """adjust_brightness

        :param x:
        :param delta:
        :return:
        """
        return tf.image.adjust_brightness(self.frame, delta=delta)

    def adjust_gamma(self, gamma=0.2) -> tf.Tensor:
        """adjust gamma, gamma > 1, brighter, gamma < 1 darker

        :param x:
        :param gamma:
        :return:
        """
        return tf.image.adjust_gamma(self.frame, gamma=gamma)


def visualize_n_save_annotated_images(images: list, titles: list, filename=None, output_dir=None, plot=False):
    """
    :param images:
    :param titles:
    :param filename:
    :param output_dir:
    :param plot:
    """
    assert len(images) >= 1, "Must supply a list containing at least one image."
    images_arr = []
    font = ImageFont.truetype("arial.ttf", 40)
    j = 0
    for im in images:
        with tf.Session() as sess:
            im = im.eval()
        image_arr = Image.fromarray(im)
        image_arr = image_arr.resize((800, 600), Image.ANTIALIAS)
        image_arr.save(os.path.join(output_dir, filename + "_" + titles[j].upper() + ".JPG"), quality=100)
        images_arr.append(image_arr)
        j += 1

    if plot:
        i = 0
        for title in titles:
            draw = ImageDraw.Draw(images_arr[i])
            draw.text((0, 0), title, (255, 255, 255), font=font)
            i += 1
        widths, heights = zip(*(i.size for i in images_arr))
        w = sum(widths)
        h = max(heights)
        new_im = Image.new('RGB', (w, h))
        x_offset = 0
        for im in images_arr:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save('augmented.png')


def images_to_video(dir, file_name='test_image.avi', frame_rate=3):
    """ convert test images to video for color splash
    :param dir:
    :param file_name:
    :param frame_rate:
    """
    img_array = []
    for image in glob.glob(dir + '\\*.jpg'):
        img = cv2.imread(image)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps=frame_rate, frameSize=size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def augment(sample):
    """ augment supplied frame

    :param sample:
    :return:
    """
    sample = tf.gfile.Open(sample, 'rb').read()
    assert isinstance(sample, bytes), "Must supply frame data in raw bytes, supplied type: {}".format(type(sample))
    image = np.array(Image.open(io.BytesIO(sample)))[..., :3]
    image_tf = tf.convert_to_tensor(image)
    aug = Augmenter(image_tf)

    flipped_up_down = aug.flip_up_down()
    flipped_left_right = aug.flip_left_right()
    cropped = aug.central_crop()
    rot180 = aug.rotate(degrees=180)
    rot90 = aug.rotate()
    contrast = aug.adjust_contrast()
    hue = aug.adjust_hue()
    bright = aug.adjust_brightness()

    return [flipped_up_down, flipped_left_right, cropped, rot90, rot180, contrast, hue, bright]


def test_augment():
    home_object_dataset = ""
    output_dir = ""
    samples = glob.glob(home_object_dataset + "\\*.JPG")

    assert len(samples) > 0, "Supplied dataset directory contains no samples\n"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sample in samples:
        augmented_samples = augment(sample)
        filename = os.path.basename(sample)
        filename, ext = os.path.splitext(filename)
        visualize_n_save_annotated_images(augmented_samples, ['flipupdown', "flipleftright",
                                                              "cropped", "rotate_90", "rotate_180",
                                                              "contrast", "hue",
                                                              "brightness"], filename=filename, output_dir=output_dir)


if __name__ == '__main__':
    images_to_video("E:\\Datasets\\HomeObjects06\\Test")
