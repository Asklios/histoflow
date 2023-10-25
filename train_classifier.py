import numpy as np
from PIL import Image
from numpy import array

from joblib import dump, load
from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial


def train_classifier(image_path, mask_path, model_path):

    sigma_min = 1
    sigma_max = 32
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

    im = Image.open(image_path, 'r')
    mask = Image.open(mask_path, 'r')

    data: array = np.array(im)
    mask_data: array = np.array(mask)

    # check if shape matches
    data_shape = data.shape
    mask_shape = mask_data.shape
    if data_shape[0] != mask_shape[0] or data_shape[1] != mask_shape[1]:
        print(f"ERROR: Mask does not match image shape: {image_path}")
        exit()

    features = features_func(data)
    clf = future.fit_segmenter(mask_data, features, clf)
    dump((clf, features_func), model_path)
