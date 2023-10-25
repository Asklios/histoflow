import os
import numpy as np
from PIL import Image


def revert_tiles(tiles_folder, output_folder):
    for file in os.listdir(tiles_folder):
        if file.endswith(".txt"):
            print(file)
            with open(os.path.join(tiles_folder, file), "r") as f:
                lines = f.readlines()
                image_name = lines[0].strip().removeprefix("./input/")
                overlap = int(lines[2].split(":")[1].strip())
                image_height, image_width = map(int, lines[3].strip().split(","))
                lines = lines[4:]

                image = Image.new("L", (image_width, image_height), color=0)
                for line in lines:
                    tile_name = line.split(':')[0]
                    x, y, w, h = map(int, line.split(':')[1].strip().split(","))

                    # if file does not exist, skip
                    if not os.path.isfile(os.path.join(tiles_folder, tile_name)):
                        continue

                    tile = np.array(Image.open(os.path.join(tiles_folder, tile_name)))

                    # correct for edge tiles
                    if x + w > image_width:
                        w = image_width - x
                        tile = tile[:, :w]
                    if y + h > image_height:
                        h = image_height - y
                        tile = tile[:h, :]

                    try:
                        image.paste(Image.fromarray(tile), (x, y, x + w, y + h))
                    except ValueError:
                        pass  # TODO fix this

            image.save(os.path.join(output_folder, file.removesuffix(".txt") + ".tif"))
