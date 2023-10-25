import os.path

import PIL
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure

PIL.Image.MAX_IMAGE_PIXELS = 10000000000


def add_overlap(tiles: [()], overlap):
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


def visualize_tiles(original_image, mask, tiles, all_tiles, image_name, overlap, output_folder):
    # Save the visualized images
    os.makedirs(output_folder, exist_ok=True)

    # handle small missmatch in size between mask and image due to resizing
    mask_new = np.zeros_like(original_image[:, :, 0], dtype=np.uint8)
    rows, cols = mask.shape
    rows = min(rows, mask_new.shape[0])
    cols = min(cols, mask_new.shape[1])
    mask_new[:rows, :cols] = mask[:rows, :cols]

    original_image_shape = original_image.shape
    original_image = Image.fromarray(original_image)

    original_image.paste(Image.fromarray(mask_new), (0, 0, original_image_shape[1], original_image_shape[0]),
                         mask=Image.fromarray(255 - mask_new))
    original_image.save(f"{output_folder}/{image_name}_overlay.png")

    # Create an image with red borders around all tiles
    border_image = original_image.copy()
    draw = ImageDraw.Draw(border_image)
    for tile in all_tiles:
        x, y, w, h = tile
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=15)
    border_image.save(f"{output_folder}/{image_name}_tile_borders.png")

    # Create an image with only the selected tiles as red borders
    selected_tiles_image = original_image.copy()
    draw = ImageDraw.Draw(selected_tiles_image)
    for tile in tiles:
        x, y, w, h = tile
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=15)
    selected_tiles_image.save(f"{output_folder}/{image_name}_selected_tiles.png")

    # Visualize the overlap with green borders
    draw = ImageDraw.Draw(selected_tiles_image)
    tiles = add_overlap(tiles, overlap)
    for tile in tiles:
        x, y, w, h = tile
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=15)
    selected_tiles_image.save(f"{output_folder}/{image_name}_selected_overlap_tiles.png")


def create_visualization(input_folder, mask_folder, visualization_folder, tile_size, overlap):
    for im in os.listdir(input_folder):
        if im.endswith(".tif"):
            print(f"Processing image: {im}")
            image_path = os.path.join(input_folder, im)
            mask_name = im.removesuffix(".tif") + "_mask.tif"
            mask_name = mask_name.replace("_normalized", "")
            mask_path = os.path.join(mask_folder, mask_name)

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

            tile_cols = image_width // tile_size + 1
            tile_rows = image_height // tile_size + 1

            # Generate min tiles
            tiles = []
            all_tiles = []
            for row in range(tile_rows):
                for col in range(tile_cols):
                    x = col * tile_size + con_x_min
                    y = row * tile_size + con_y_min

                    w = tile_size
                    h = tile_size
                    tile = mask[y:y + h, x:x + w]

                    if tile.sum() > 0:
                        tiles.append((x, y, w, h))
                    all_tiles.append((x, y, w, h))

            image_name = os.path.basename(image_path).split(".")[0]
            image = np.array(Image.open(image_path))

            visualize_tiles(image, mask, tiles, all_tiles, image_name, overlap, visualization_folder)
