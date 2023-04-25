import os

import PIL
import numpy as np
from PIL import Image
from numpy import array

from settings import *

PIL.Image.MAX_IMAGE_PIXELS = 10000000000  # allow huge images

info = {}

for info_filename in os.listdir(TILES_FOLDER):
    if info_filename.endswith(".txt"):
        with open(os.path.join(TILES_FOLDER, info_filename), "r") as f:
            tile_size, overlap_size = [int(x) for x in f.readline().split()]
            image_width, image_height, final_width, final_height = [int(x) for x in f.readline().split()]
            info[info_filename[:-4]] = {
                "tile_size": tile_size,
                "overlap_size": overlap_size,
                "image_width": image_width,
                "image_height": image_height,
                "final_width": final_width,
                "final_height": final_height
            }

for image_filename in info.keys():
    print(f"Processing image: {image_filename}")
    image_path = os.path.join(OUTPUT_FOLDER, image_filename)
    data = np.zeros((info[image_filename]["final_width"], info[image_filename]["final_height"], 3), dtype=np.uint8)
    for tile_filename in os.listdir(TILES_FOLDER):
        if not tile_filename.startswith(image_filename.removesuffix('.tif')) or not tile_filename.endswith(".tif"):
            continue

        tile_path = os.path.join(TILES_FOLDER, tile_filename)

        with Image.open(tile_path, "r") as im:
            tile: array = np.array(im)
        i, j = [int(x) for x in tile_filename[len(image_filename.removesuffix('.tif')) + 1:-4].split("_")]
        x_start = i * (info[image_filename]["tile_size"] - info[image_filename]["overlap_size"])
        y_start = j * (info[image_filename]["tile_size"] - info[image_filename]["overlap_size"])
        x_end = min(x_start + info[image_filename]["tile_size"], info[image_filename]["image_width"])
        y_end = min(y_start + info[image_filename]["tile_size"], info[image_filename]["image_height"])
        data[x_start:x_end, y_start:y_end, :] = tile[:x_end - x_start, :y_end - y_start, :]

    data = data[:info[image_filename]["image_width"], :info[image_filename]["image_height"], :]

    with Image.fromarray(data) as im:
        im.save(image_path)
