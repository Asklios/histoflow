import os
import numpy as np
from PIL import Image

TILES_FOLDER = "segmentation/"
OUTPUT_FOLDER = "output/"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for file in os.listdir(TILES_FOLDER):
    if file.endswith(".txt"):
        print(file)
        with open(os.path.join(TILES_FOLDER, file), "r") as f:
            lines = f.readlines()
            image_name = lines[0].strip().removeprefix("./input/")
            overlap = int(lines[1].split(":")[1].strip())
            image_height, image_width = map(int, lines[2].strip().split(","))
            lines = lines[3:]

            image = Image.new("L", (image_width, image_height), color=0)
            for line in lines:
                tile_name = line.split(':')[0]
                x, y, w, h = map(int, line.split(':')[1].strip().split(","))

                tile = np.array(Image.open(os.path.join(TILES_FOLDER, tile_name)))

                # correct for edge tiles
                if x + w > image_width:
                    w = image_width - x
                    tile = tile[:, :w]
                if y + h > image_height:
                    h = image_height - y
                    tile = tile[:h, :]

                image.paste(Image.fromarray(tile), (x, y, x + w, y + h))

        image.save(os.path.join(OUTPUT_FOLDER, image_name))
