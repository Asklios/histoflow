import os.path

import PIL
from PIL import Image
import numpy as np
from skimage import measure

from notify import notify
from settings import *


PIL.Image.MAX_IMAGE_PIXELS = 10000000000


def add_overlap(tiles: [()]):
    # add overlap to the tiles
    new_tiles = []
    for t in tiles:
        x, y, w, h = t
        x -= OVERLAP
        y -= OVERLAP
        w += 2 * OVERLAP
        h += 2 * OVERLAP

        # check if x or y is negative
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0

        new_tiles.append((x, y, w, h))
    return new_tiles


def create_tiles(image_path, mask_path):

    # check if files exist
    if not os.path.exists(mask_path):
        print(f"Mask {mask_path} does not exist.")
        exit(0)

    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.repeat(2, axis=0).repeat(2, axis=1)  # double the size of the mask since it is 1/2 the size of the image

    orig_image_width, orig_image_height = mask.shape[:2]

    # crop the mask to the contours
    contour = measure.find_contours(mask, 1)[0]
    con_y_min, con_x_min = np.min(contour, axis=0).astype(int)
    con_y_max, con_x_max = np.max(contour, axis=0).astype(int)

    image_width = con_x_max - con_x_min
    image_height = con_y_max - con_y_min

    tile_cols = image_width // TILE_SIZE + 1
    tile_rows = image_height // TILE_SIZE + 1

    new_width = tile_cols * TILE_SIZE
    new_height = tile_rows * TILE_SIZE

    left_minus = (new_width - image_width) // 2
    top_minus = (new_height - image_height) // 2

    x_correction = con_x_min - left_minus
    y_correction = con_y_min - top_minus

    # Generate min tiles
    tiles = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            x = col * TILE_SIZE + x_correction
            y = row * TILE_SIZE + y_correction

            w = TILE_SIZE
            h = TILE_SIZE
            tile = mask[y:y + h, x:x + w]
            if tile.sum() > 0:
                tiles.append((x, y, w, h))

    tiles = add_overlap(tiles)

    image_name = os.path.basename(image_path).split(".")[0]
    image = np.array(Image.open(image_path))

    with open(f'{TILES_FOLDER}/{image_name}.txt', 'w') as file:
        file.write(f'{image_path}\n')
        file.write(f'overlap: {OVERLAP}\n')
        file.write(f'{orig_image_width},{orig_image_height}\n')
        for i, tile in enumerate(tiles):
            file.write(f'{image_name}_{i}.tif: {tile[0]},{tile[1]},{tile[2]},{tile[3]}\n')

            # save tile as mask
            tile_image = image[tile[1]:tile[1] + tile[3], tile[0]:tile[0] + tile[2]]
            tile_image = Image.fromarray(tile_image)
            tile_image.save(f'{TILES_FOLDER}/{image_name}_{i}.tif')


for im in os.listdir(INPUT_FOLDER):
    if im.endswith(".tif"):
        create_tiles(os.path.join(INPUT_FOLDER, im), os.path.join(MASK_FOLDER, im.removesuffix(".tif") + "_mask.tif"))


notify("Server finished creating tiles.")
