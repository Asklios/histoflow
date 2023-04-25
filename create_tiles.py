import os

import PIL
import numpy as np
from PIL import Image
from numpy import array

from settings import *

PIL.Image.MAX_IMAGE_PIXELS = 10000000000  # allow huge images

# Process each image in input folder
for image_filename in os.listdir(INPUT_FOLDER):
    if not image_filename.endswith(".tif"):
        continue
    print(f"Processing image: {image_filename}")
    # Load input image
    image_path = os.path.join(INPUT_FOLDER, image_filename)
    with Image.open(image_path, "r") as im:
        data: array = np.array(im)

    image_width = data.shape[0]
    image_height = data.shape[1]

    tile_rows = (image_width + OVERLAP_SIZE) // (TILE_SIZE - OVERLAP_SIZE)
    tile_cols = (image_height + OVERLAP_SIZE) // (TILE_SIZE - OVERLAP_SIZE)

    final_width = tile_rows * (TILE_SIZE - OVERLAP_SIZE) + OVERLAP_SIZE
    final_height = tile_cols * (TILE_SIZE - OVERLAP_SIZE) + OVERLAP_SIZE

    with open(f"tiles/{image_filename}.txt", "w") as f:
        f.write(f"{TILE_SIZE} {OVERLAP_SIZE}\n")
        f.write(f"{image_width} {image_height} {final_width} {final_height}")

    for i in range(tile_rows):
        for j in range(tile_cols):
            x_start = i * (TILE_SIZE - OVERLAP_SIZE)
            y_start = j * (TILE_SIZE - OVERLAP_SIZE)
            x_end = min(x_start + TILE_SIZE, image_width)
            y_end = min(y_start + TILE_SIZE, image_height)
            tile = data[x_start:x_end, y_start:y_end, :]
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                # pad tile with zeros
                tile = np.pad(tile, ((0, TILE_SIZE - tile.shape[0]), (0, TILE_SIZE - tile.shape[1]), (0, 0)),
                              mode="constant")
            tile_filename = f"{image_filename.removesuffix('tif')}_{i}_{j}.tif"
            tile_path = os.path.join(TILES_FOLDER, tile_filename)
            with Image.fromarray(tile) as im:
                im.save(tile_path)







