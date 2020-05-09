import os
import sys
import random
import math
import numpy as np
import skimage.io

# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/nevera"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.abspath("../../")
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn import model as modellib

from samples.nevera import nevera

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == '__main__':
    # Modify the class_names with your classes 
    class_names = ['BG', 'nevera']
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect neveras.')
    parser.add_argument('--dir', required=False,
                        metavar="/path/to/nevera/dataset/",
                        help='Directory of the images to predict')
    parser.add_argument('--img', required=True,
                        metavar="/path/to/img.jpg",
                        help="Name of image, only .jpg")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    
    args = parser.parse_args()

    
    assert args.dir, "Argument --dir is required for predict like: --dir=/content/images"
    assert args.dir, "Argument --weights is required for predict like: --dir=/content/weights/mask_rcnn_nevera_0030.h5"

    print("Weights: ", args.weights)
    print("Dir: ", args.dir)
    print("Img: ", args.img)

    class InferenceConfig(nevera.NeveraConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


    # Load a random image from the images folder
    image = skimage.io.imread(os.path.join(args.dir, args.img))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
