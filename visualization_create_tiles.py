import os.path

import PIL
from PIL import Image, ImageDraw
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

    # crop the mask to the contours
    contour = measure.find_contours(mask, 1)[0]
    con_y_min, con_x_min = np.min(contour, axis=0).astype(int)
    con_y_max, con_x_max = np.max(contour, axis=0).astype(int)

    image_width = con_x_max - con_x_min
    image_height = con_y_max - con_y_min

    tile_cols = image_width // TILE_SIZE + 1
    tile_rows = image_height // TILE_SIZE + 1

    # Generate min tiles
    tiles = []
    all_tiles = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            x = col * TILE_SIZE + con_x_min
            y = row * TILE_SIZE + con_y_min

            w = TILE_SIZE
            h = TILE_SIZE
            tile = mask[y:y + h, x:x + w]

            if tile.sum() > 0:
                tiles.append((x, y, w, h))
            all_tiles.append((x, y, w, h))

    image_name = os.path.basename(image_path).split(".")[0]
    image = np.array(Image.open(image_path))

    visualize_tiles(image, mask, tiles, all_tiles, image_name)


def visualize_tiles(original_image, mask, tiles, all_tiles, image_name):
    # Save the visualized images
    output_folder = "visualizations"
    os.makedirs(output_folder, exist_ok=True)

    # Overlay the mask on the original image
    overlay_image = Image.fromarray(original_image)

    mask_new = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
    mask_new[0:mask.shape[0], 0:mask.shape[1]] = mask

    overlay_image.paste(Image.fromarray(mask), (0, 0, original_image.shape[1], original_image.shape[0]), mask=Image.fromarray(255 - mask_new))
    overlay_image.save(f"{output_folder}/{image_name}_overlay.png")

    # Create an image with red borders around all tiles
    border_image = overlay_image.copy()
    draw = ImageDraw.Draw(border_image)
    for tile in all_tiles:
        x, y, w, h = tile
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=15)
    border_image.save(f"{output_folder}/{image_name}_tile_borders.png")

    # Create an image with only the selected tiles as red borders
    selected_tiles_image = overlay_image.copy()
    draw = ImageDraw.Draw(selected_tiles_image)
    for tile in tiles:
        x, y, w, h = tile
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=15)
    selected_tiles_image.save(f"{output_folder}/{image_name}_selected_tiles.png")

    # Visualize the overlap with green borders
    draw = ImageDraw.Draw(selected_tiles_image)
    tiles = add_overlap(tiles)
    for tile in tiles:
        x, y, w, h = tile
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=15)
    selected_tiles_image.save(f"{output_folder}/{image_name}_selected_overlap_tiles.png")


for im in os.listdir(INPUT_FOLDER):
    if im.endswith(".tif"):
        create_tiles(os.path.join(INPUT_FOLDER, im), os.path.join(MASK_FOLDER, im.removesuffix(".tif") + "_mask.tif"))

notify("Server finished visualizing tiles.")
