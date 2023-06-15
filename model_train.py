import os

import numpy as np
from PIL import Image
from numpy import array

from joblib import dump, load
from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial

from notify import notify
from settings import *

masks_path = 'training/mask_bi/'
images_path = 'training/tiles/'

mask_files = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]
image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

if len(image_files) != len(mask_files):
    print("Number of images and masks do not match")
    exit()

sigma_min = 1
sigma_max = 32
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

shape = None

full_data = None
full_mask = None

for image in image_files:
    image_path = images_path + image
    mask_path = masks_path + image
    im = Image.open(image_path, 'r')
    mask = Image.open(mask_path, 'r')

    data: array = np.array(im)
    mask_data: array = np.array(mask)

    if shape is None:
        shape = data.shape

    if shape != data.shape:
        print(f"Shape mismatch to previous images: {image}")
        exit()

    # if data.shape != mask_data.shape:
    #    print(f"Mask does not match image shape: {image}")
    #    exit()

    if full_data is None:
        full_data = data
        full_mask = mask_data
    else:
        full_data = np.concatenate((full_data, data), axis=0)
        full_mask = np.concatenate((full_mask, mask_data), axis=0)


features = features_func(full_data)
clf = future.fit_segmenter(full_mask, features, clf)
dump((clf, features_func), MODEL_PATH)

notify('Model trained.')
