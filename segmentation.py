import os
import shutil

import PIL
import numpy as np
from PIL import Image
from joblib import load
from numpy import array
from skimage import future
from timeit import default_timer as timer

PIL.Image.MAX_IMAGE_PIXELS = 10000000000  # allow huge images


def segment_tiles(tiles_folder, segmentation_folder, model_path):
    # Load model
    clf, features_func = load(model_path)

    # Process each image in input folder
    for image_filename in os.listdir(tiles_folder):
        if not image_filename.endswith('.tif'):
            if image_filename.endswith('.txt'):
                shutil.copyfile(os.path.join(tiles_folder, image_filename),
                                os.path.join(segmentation_folder, image_filename))
            continue
        print(f"Processing image: {image_filename}")
        start = timer()

        # Load input image
        image_path = os.path.join(tiles_folder, image_filename)
        with Image.open(image_path, 'r') as im:
            data: array = np.array(im)

        # Segment image
        features = features_func(data)
        segmentation = future.predict_segmenter(features, clf)

        # Save output image
        output_path = os.path.join(segmentation_folder, image_filename)
        Image.fromarray(segmentation).save(output_path)
        print(f"Saved segmentation for image: {image_filename} calculated in {round((timer() - start), 2)} seconds.")
