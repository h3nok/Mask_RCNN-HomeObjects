"""
Verison Assignment;
Mask R-CNN for home objects dataset
------------------------------------------------------------

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from dataset_utils import _list_to_file
import imgaug
from enum import Enum
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import Config
import model as modellib
from visualize import display_instances
import utils
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class HomeObjectConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "homeobject"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 56 + 1  # Background + toy
    # backbone
    BACKBONE = 'resnet50'
    # Training epochs
    EPOCHS = 10
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

def _read_classifications(file):
    assert os.path.exists(file), "Unable to load classification list file, file:{}.".format(file)
    with open(file) as f:
        return list(f)


class HomeObjectDataset(utils.Dataset):
    unique_classes = set()
    unique_categories = set()
    classification_to_id_map = dict()

    def load_homeobject(self, dataset_dir, subset):
        """Load the dataset.
        :param dataset_dir: Root directory of the dataset
        :param subset: Subset to load_homeobject: Train or Test
        """
        """ VIA 2.x ann format 
        {"filename":"P1020590.JPG","size":201990,
                "regions":[
                    {"shape_attributes":{"name":"polyline","all_points_x":
                    [87,58,66,103,223,239,263,296,590,689,697,680,463,336,289,250,230,160,120,87],
                    "all_points_y":[212,288,358,393,393,412,456,477,472,421,253,216,156,137,156,202,212,201,197,213]},
                    "region_attributes":{"Name":"Detergent","Type":"Detergent","Size":"Medium",
                    "Image Quality":"Good\n"}}],
                "file_attributes":{}}
        """
        assert subset in ["Train", "Val"]
        classifications = _read_classifications('56_class_list.txt')
        assert len(classifications) == HomeObjectConfig.NUM_CLASSES - 1, \
            "{} != {} Classification list doesn't match NUM_CLASSES\n".format(len(classifications),
                                                                              HomeObjectConfig.NUM_CLASSES - 1)
        for c in range(len(classifications)):
            self.add_class('homeobject', c + 1, classifications[c])
            self.classification_to_id_map[classifications[c].strip().lower()] = c + 1

        _list_to_file('class_to_id_map.txt', self.classification_to_id_map)
        dataset_dir = os.path.join(dataset_dir, subset)
        # load annotations
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations1.values())  # don't need the dict keys
        # skip images without annotations
        annotations = [a for a in annotations if a['regions']]
        # unique_class_counter = 0
        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [r['region_attributes'] for r in a['regions']]
            for cat in objects:
                self.unique_categories.add(cat['Type'].strip().lower())
                self.unique_classes.add(cat['Name'].strip().lower())
            class_ids = [self.classification_to_id_map[n['Type'].strip().lower()] for n in objects]
            # # load_mask() needs the image size to convert polygons to masks.
            # # Unfortunately, VIA doesn't include it in JSON, so we must read
            # # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path).astype(np.bool)
            height, width = image.shape[:2]
            self.add_image(
                'homeobject',  # add sample class label
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

        _list_to_file(subset + "_categories.txt", sorted(self.unique_categories))
        _list_to_file(subset + "_classes.txt", sorted(self.unique_classes))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        :type image_id: object
        :param masks: A bool array of shape [height, width, instance count] with one mask per instance.
        :param class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not homeobject dataset, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != 'homeobject':
            print(
                "Warn: \'{}\'  label not found. Processing with parent load_mask.".format(image_info["source"]))
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        class_ids = image_info['class_ids']
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            # modify dirt mask if it resides outside of image boundary
            rr[rr > mask.shape[0] - 1] = mask.shape[0] - 1
            cc[cc > mask.shape[1] - 1] = mask.shape[1] - 1

            mask[rr, cc, i] = 1
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_ids = np.array(class_ids, dtype=np.int32)
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"].strip().lower() == 'homeobject':
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class AugmentType(Enum):
    Sequential = "Augment and feed samples for training sequentially"
    Sometimes = "Choose one from a list"


def train(m, layers='all', augment=False, augment_type: AugmentType = AugmentType.Sequential) -> None:
    """Train the model.
        Strategies (initial):
        1. Initialize network weights with coco or imagenet
        2. Train network heads without augmentation
        3. Train all layers without augmentation
        4. Train only heads with augmentation
        5. Train all layers with augmentation
    """
    # Load test dataset
    dataset_train = HomeObjectDataset()
    dataset_train.load_homeobject(args.dataset, "Train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = HomeObjectDataset()
    dataset_val.load_homeobject(args.dataset, "Val")
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network, layers: {}".format(layers))
    augmentation = None
    if augment_type == AugmentType.Sequential:
        augmentation = imgaug.augmenters.Sequential(
            [
                imgaug.augmenters.Fliplr(1),  # horizontally flip all images
                imgaug.augmenters.Flipud(1),  # vertically flip all images
                imgaug.augmenters.Affine(rotate=(-90, 90)),  # rotate
                imgaug.augmenters.Affine(rotate=(-45, 45)),  # rotate
                imgaug.augmenters.Affine(scale=(0.8, 1.2)),  # scale images to 80-120% of their size
                imgaug.augmenters.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of
                # original value)
                imgaug.augmenters.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            ]
        )
    elif augment_type == AugmentType.Sometimes:
        augmentation = imgaug.augmenters.Sometimes(1,
            [
                imgaug.augmenters.Fliplr(1),  # horizontally flip all images
                imgaug.augmenters.Flipud(1),  # vertically flip all images
                imgaug.augmenters.Affine(rotate=(-90, 90)),  # rotate
                imgaug.augmenters.Affine(rotate=(-45, 45)),  # rotate
                imgaug.augmenters.Affine(scale=(0.8, 1.2)),  # scale images to 80-120% of their size
                imgaug.augmenters.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of
                # original value)
                imgaug.augmenters.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            ]
        )
    if augment:
        m.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                layers=layers,
                augmentation=augmentation)

    m.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=config.EPOCHS,
            layers=layers)


