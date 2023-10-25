import os
import PIL
from PIL import Image
import cv2
import numpy as np
from numpy import array
from numba import jit, cuda, prange
from scipy import ndimage as nd
from timeit import default_timer as timer

from helper import request_yes_no

PIL.Image.MAX_IMAGE_PIXELS = 10000000000


def create_mask_input(input_folder, mask_input_folder):

    # check if mask_input folder is empty
    if len(os.listdir(mask_input_folder)) != 0:
        print('Warning: mask input folder is not empty. Continuing will remove existing files. Continue? (y/n)')

        if not request_yes_no():
            print('Exiting...')
            exit(0)

        # delete all files in mask folder
        for file in os.listdir(mask_input_folder):
            os.remove(os.path.join(mask_input_folder, file))

    for filename in [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]:
        # Load image
        im = Image.open(os.path.join(input_folder, filename), 'r')
        img: array = np.array(im)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale down by 0.5
        scaled = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Apply Gaussian blur with kernel size 101
        blurred = cv2.GaussianBlur(scaled, (101, 101), 0)

        # Save to output directory
        cv2.imwrite(os.path.join(mask_input_folder, filename), blurred)


@jit(target_backend='cuda', nopython=True, cache=True, parallel=True)
def select_threshold(input_data: array, median, mean):
    width = input_data.shape[0]
    height = input_data.shape[1]
    o = np.zeros((width, height), dtype=np.uint8)

    threshold = median - mean * 0.0001

    for x in prange(0, width - 1):
        # print('width: ' + str(x))
        for y in range(0, height - 1):
            if threshold < input_data[x][y]:
                o[x][y] = 0
            else:
                o[x][y] = 255

    return o


@jit(target_backend='cuda', parallel=True, nopython=True, cache=True)
def la_to_int(la):
    for i in prange(0, len(la) - 1):
        la[i] = la[i].astype(np.int8)
    return la


@jit(target_backend='cuda', nopython=True, cache=True, parallel=True)
def separate(input_data: array, mask: array, median):
    width = input_data.shape[0]
    height = input_data.shape[1]
    o = np.zeros((width, height), dtype=np.uint8)

    for x in prange(0, width - 1):
        for y in range(0, height - 1):
            if mask[x][y] == 0:
                o[x][y] = 255
            else:
                if input_data[x][y] < median - 10:
                    o[x][y] = 0
                elif input_data[x][y] < median:
                    o[x][y] = 40
                elif input_data[x][y] < median + 10:
                    o[x][y] = 140
                else:
                    o[x][y] = 200

    return o


def create_mask(mask_input_folder: str, mask_folder: str):

    # check if mask folder is empty
    if len(os.listdir(mask_folder)) != 0:
        print('Warning: mask folder is not empty. Continuing will remove existing files. Continue? (y/n)')
        if not request_yes_no():
            print('Exiting...')
            exit(0)

        # delete all files in mask folder
        for file in os.listdir(mask_folder):
            os.remove(os.path.join(mask_folder, file))

    files = [f for f in os.listdir(mask_input_folder) if os.path.isfile(os.path.join(mask_input_folder, f))]

    for file_name in files:
        print('Creating mask for: ' + file_name)

        im = Image.open(os.path.join(mask_input_folder, file_name), 'r')
        data: array = np.array(im)

        start = timer()
        nd_median = nd.median(data)
        nd_mean = nd.mean(data)

        output = select_threshold(data, nd_median, nd_mean)

        print('stage 1 took: ' + str(round((timer() - start), 2)))
        start = timer()

        s = nd.generate_binary_structure(2, 2)  # consider diagonal neighbors
        labeled_array, num_features = nd.label(output, structure=s)
        output = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)

        if num_features > 1:
            for index in prange(1, num_features):
                if np.sum(labeled_array == index) > data.size * 0.1:
                    la = la_to_int(labeled_array == index)
                    output += la
        else:
            output = labeled_array
        output[output > 0] = 255

        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        file = Image.fromarray(output)
        file.save(os.path.join(mask_folder, file_name.removesuffix('.tif') + '_mask.tif'))
        print('stage 2 took: ' + str(round((timer() - start), 2)))
