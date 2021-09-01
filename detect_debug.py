import os
import csv
import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image
from six import BytesIO

saved_model = 'model/ssd_mobilenet_v2_fpnlite/model/saved/saved_model'
image_path = '/Users/jx/MTC/技术小组/test/000324615c13cf01e55e85873201bef2.png'
# output_dir = 'output'


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


category_index = {
    1: {'id': 1, 'name': 'A'},
    2: {'id': 2, 'name': 'B'},
    3: {'id': 3, 'name': 'C'},
    4: {'id': 4, 'name': 'D'},
    5: {'id': 5, 'name': 'E'},
    6: {'id': 6, 'name': 'F'},
    7: {'id': 7, 'name': 'G'},
    8: {'id': 8, 'name': 'H'},
    9: {'id': 9, 'name': 'I'},
    10: {'id': 10, 'name': 'J'},
    11: {'id': 11, 'name': 'K'},
    12: {'id': 12, 'name': 'L'},
    13: {'id': 13, 'name': 'M'},
    14: {'id': 14, 'name': 'N'},
    15: {'id': 15, 'name': 'O'},
    16: {'id': 16, 'name': 'P'},
    17: {'id': 17, 'name': 'Q'},
    18: {'id': 18, 'name': 'R'},
    19: {'id': 19, 'name': 'S'},
    20: {'id': 20, 'name': 'T'},
    21: {'id': 21, 'name': 'U'},
    22: {'id': 22, 'name': 'V'},
    23: {'id': 23, 'name': 'W'},
    24: {'id': 24, 'name': 'X'},
    25: {'id': 25, 'name': 'Y'},
    26: {'id': 26, 'name': 'Z'},
    27: {'id': 27, 'name': '0'},
    28: {'id': 28, 'name': '1'},
    29: {'id': 29, 'name': '2'},
    30: {'id': 30, 'name': '3'},
    31: {'id': 31, 'name': '4'},
    32: {'id': 32, 'name': '5'},
    33: {'id': 33, 'name': '6'},
    34: {'id': 34, 'name': '7'},
    35: {'id': 35, 'name': '8'},
    36: {'id': 36, 'name': '9'},
}


tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(saved_model)

file_ext = pathlib.Path(image_path).suffix
file_stem = pathlib.Path(image_path).stem

image_np = load_image_into_numpy_array(image_path)
input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

print(detections)


