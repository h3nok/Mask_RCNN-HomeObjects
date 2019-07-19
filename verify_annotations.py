import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import home_objects


import utils as utils
import visualize as visualize
from visualize import display_images
import model as modellib
from model import log

ROOT_DIR = os.path.abspath(".")
DATASET_DIR = "D:\\HomeObjects06"
ANNOTATION_DIR = os.path.join(DATASET_DIR, 'Annotated')
sys.path.append(ROOT_DIR)

dataset = home_objects.HomeObjectDataset()
subset = 'Train'
dataset.load_homeobject(DATASET_DIR, subset)
dataset.prepare()
ANNOTATION_DIR = os.path.join(ANNOTATION_DIR, subset)
if not os.path.exists(ANNOTATION_DIR):
    os.makedirs(ANNOTATION_DIR)
# #
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
image_ids = np.random.choice(dataset.image_ids, 4)
# for image_id in image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     print(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# Load random image and mask.
# image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in dataset.image_ids:
    print(image_id)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    path = os.path.join(ANNOTATION_DIR, os.path.basename(dataset.image_reference(image_id)))
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, path)
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    log("file", )
    # Display image and instances
    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, output=path)
