import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import os


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def aug_func(img):
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.1, 2.0])
    a_img = load_img('{}.png'.format(img))
    x = img_to_array(a_img)
    x = x.reshape((1,) + x.shape)
    i = 0
    create_dir(img)
    for batch in datagen.flow(x, batch_size=1, save_to_dir="{}".format(img), save_prefix='img', save_format='jpeg'):
        i += 1
        if i > 300:
            break
    print("Files Augmented")


if __name__ == "__main__":
    aug_func("Red")
    aug_func("Blue")
    aug_func("Green")