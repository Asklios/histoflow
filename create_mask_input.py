import os

import PIL
import cv2
import numpy as np
from PIL import Image
from numpy import array

from notify import notify
from settings import *

PIL.Image.MAX_IMAGE_PIXELS = 10000000000

# Loop through input directory
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".tif"):
        # Load image
        im = Image.open(os.path.join(INPUT_FOLDER, filename), 'r')
        img: array = np.array(im)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale down by 0.5
        scaled = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Apply Gaussian blur with kernel size 101
        blurred = cv2.GaussianBlur(scaled, (101, 101), 0)

        # Save to output directory
        cv2.imwrite(os.path.join(MASK_INPUT_FOLDER, filename), blurred)

notify("Server is done creating mask input.")
