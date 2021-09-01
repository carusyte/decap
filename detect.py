import os
import csv
import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image
from six import BytesIO

saved_model = 'model/ssd_mobilenet_v2_fpnlite/model/saved/saved_model'
image_dir = '/Users/jx/MTC/技术小组/test'
output_dir = 'output'


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

answers = []

for filename in os.listdir(image_dir):
    file_ext = pathlib.Path(filename).suffix
    file_stem = pathlib.Path(filename).stem
    if file_ext.lower() != ".jpg" and file_ext.lower() != ".png":
        continue
    image_path = os.path.join(image_dir, filename)
    if not os.path.isfile(image_path):
        continue

    image_np = load_image_into_numpy_array(image_path)
    input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int32)

    classes = detections['detection_classes']
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    # anchors = detections['detection_anchor_indices']

    # get the first 4 detected classes and order them by x coordinate
    classes = classes[:4]
    xmins = boxes[:4][:, 1]
    # sort the classes based on xmin and get the labels
    labels = [category_index[c]['name']
              for _, c in sorted(zip(xmins, classes))]

    answers.append({
        'file': filename,
        'labels': ''.join(labels),
    })

    # print(''.join(labels))

    # plt.rcParams['figure.figsize'] = [42, 21]
    # label_id_offset = 1
    # image_np_with_detections = image_np.copy()
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #     image_np_with_detections,
    #     detections['detection_boxes'],
    #     classes + label_id_offset,
    #     scores,
    #     category_index,
    #     use_normalized_coordinates=True,
    #     max_boxes_to_draw=200,
    #     min_score_thresh=.40,
    #     agnostic_mode=False)

    # # plt.subplot(2, 1, i+1)
    # plt.figure()
    # plt.imshow(image_np_with_detections)
    # # plt.show()  ## UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
    # detected_img_path = os.path.join(
    #     output_dir, filename)
    # plt.savefig(detected_img_path)

answers = sorted(answers, key=lambda kv: kv['file'])
keys = answers[0].keys()
with open(os.path.join(output_dir, 'answers.csv'), 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(answers)
