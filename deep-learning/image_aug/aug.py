from pathlib import Path
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img
from ctypes import resize
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array

# image = tf.keras.preprocessing.image.load_img(
#     "cat.jpg",  # 500x500
#     grayscale=False,
#     color_mode="rgb",
#     target_size=None,
#     interpolation="nearest",
#     resize=(64, 64)
# )

# x = np.array([img_to_array(image)])
# print(x.shape)  # (1, 500, 500, 3)

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     featurewise_center=False, samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False, zca_epsilon=1e-06,
#     rotation_range=40, width_shift_range=0.2,
#     height_shift_range=0.2, brightness_range=None,
#     shear_range=0.2, zoom_range=0.2,
#     channel_shift_range=0.0, fill_mode="nearest", cval=0.0,
#     horizontal_flip=True, vertical_flip=False,
#     rescale=None / 255, preprocessing_function=None,
#     data_format=None, validation_split=0.0, dtype=None,
# )
# print(len(datagen.flow(
#     x, batch_size=1, save_to_dir="augmented",
#     save_prefix="cat", save_format="png"
# )))
# i = 0
# for batch in datagen.flow(
#     x, batch_size=1, save_to_dir="augmented",
#     save_prefix="cat", save_format="png"
# ):
#     i += 1
#     if i > 25:
#         break
