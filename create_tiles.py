import os.path

import PIL
from PIL import Image
import numpy as np
from skimage import measure

from helper import filter_supported_files

PIL.Image.MAX_IMAGE_PIXELS = 10000000000


def add_overlap(tiles: [()], overlap: int):
    # add overlap to the tiles
    new_tiles = []
    for t in tiles:
        x, y, w, h = t
        x -= overlap
        y -= overlap
        w += 2 * overlap
        h += 2 * overlap

        # check if x or y is negative
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0

        new_tiles.append((x, y, w, h))
    return new_tiles


def create_tiles(input_folder, mask_folder, tiles_folder, tile_size, overlap):

    for im in filter_supported_files(input_folder):
        print('Creating tiles for ' + im + '...')
        image_path = os.path.join(input_folder, im)
        mask_path = os.path.join(mask_folder, os.path.splitext(im)[0] + '_mask' + os.path.splitext(im)[1])
        mask_path = mask_path.replace('_normalized', '')

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

        tile_cols = image_width // tile_size + 1
        tile_rows = image_height // tile_size + 1

        new_width = tile_cols * tile_size
        new_height = tile_rows * tile_size

        left_minus = (new_width - image_width) // 2
        top_minus = (new_height - image_height) // 2

        x_correction = con_x_min - left_minus
        y_correction = con_y_min - top_minus

        # Generate min tiles
        tiles = []
        for row in range(tile_rows):
            for col in range(tile_cols):
                x = col * tile_size + x_correction
                y = row * tile_size + y_correction

                w = tile_size
                h = tile_size
                tile = mask[y:y + h, x:x + w]
                if tile.sum() > 0:
                    tiles.append((x, y, w, h))

        tiles = add_overlap(tiles, overlap)

        image_name = os.path.basename(image_path).split(".")[0]
        image = np.array(Image.open(image_path))

        with open(f'{tiles_folder}/{image_name}.txt', 'w') as file:
            file.write(f'{image_path}\n')
            file.write(f'tile_size: {tile_size}\n')
            file.write(f'overlap: {overlap}\n')
            file.write(f'{orig_image_width},{orig_image_height}\n')
            for i, tile in enumerate(tiles):
                file.write(f'{image_name}_{i}.tif: {tile[0]},{tile[1]},{tile[2]},{tile[3]}\n')

                # save tile as mask
                tile_image = image[tile[1]:tile[1] + tile[3], tile[0]:tile[0] + tile[2]]
                tile_image = Image.fromarray(tile_image)
                tile_image.save(f'{tiles_folder}/{image_name}_{i}.tif')
