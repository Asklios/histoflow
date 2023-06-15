import os
from numba import jit, cuda, prange
import numpy as np
from numpy import array
from scipy import ndimage as nd
import PIL
from PIL import Image
from notify import notify

from timeit import default_timer as timer

from settings import *

PIL.Image.MAX_IMAGE_PIXELS = 10000000000


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


files = [f for f in os.listdir(MASK_INPUT_FOLDER) if os.path.isfile(os.path.join(MASK_INPUT_FOLDER, f))]

for file_name in files:
    if file_name.endswith('.tif'):
        print(file_name)
        notify('Creating mask for ' + file_name)

        im = Image.open(os.path.join(MASK_INPUT_FOLDER, file_name), 'r')
        data: array = np.array(im)

        start = timer()
        nd_median = nd.median(data)
        nd_mean = nd.mean(data)

        output = select_threshold(data, nd_median, nd_mean)

        print('stage 1 took: ' + str(timer() - start))
        start = timer()

        s = nd.generate_binary_structure(2, 2)  # consider diagonal neighbors
        labeled_array, num_features = nd.label(output, structure=s)
        output = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)

        if num_features > 1:
            for index in prange(1, num_features):
                print(index)
                if np.sum(labeled_array == index) > data.size * 0.1:
                    la = la_to_int(labeled_array == index)
                    output += la
        else:
            output = labeled_array
        output[output > 0] = 255

        if not os.path.exists(MASK_FOLDER):
            os.makedirs(MASK_FOLDER)

        file = Image.fromarray(output)
        file.save(os.path.join(MASK_FOLDER, file_name.removesuffix('.tif') + '_mask.tif'))
        print('stage 2 took: ' + str(timer() - start))
        # cv2.imwrite(os.path.join(MASK_FOLDER, file_name.removesuffix('.tif') + '_mask.tif'), output)

print('done')
notify('Server finished creating masks')
