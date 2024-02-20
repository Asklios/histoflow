import os
import numpy as np
from PIL import Image


def revert_tiles(tiles_folder, output_folder):
    for file in os.listdir(tiles_folder):
        if file.endswith(".txt"):
            print(file)
            with open(os.path.join(tiles_folder, file), "r") as f:
                lines = f.readlines()
                tile_size = int(lines[1].split(":")[1].strip())
                overlap = int(lines[2].split(":")[1].strip())
                image_height, image_width = map(int, lines[3].strip().split(","))
                lines = lines[5:]

                image = Image.new("L", (image_width, image_height), color=0)
                for line in lines:
                    tile_name = line.split(':')[0]
                    x, y, w, h, l, t = map(int, line.split(':')[1].strip().split(","))

                    # if file does not exist, skip
                    if not os.path.isfile(os.path.join(tiles_folder, tile_name)):
                        continue

                    tile = np.array(Image.open(os.path.join(tiles_folder, tile_name)))

                    # cut off overlap and paste
                    tile = tile[t:tile.shape[0]-overlap, l:tile.shape[1]-overlap]

                    x = x + l
                    y = y + t

                    x2 = x + tile.shape[0]
                    y2 = y + tile.shape[1]

                    if x2 > image_width:
                        dif = x2 - image_width
                        tile = tile[:, :-dif]
                        x2 = image_width
                    if y2 > image_height:
                        dif = y2 - image_height
                        tile = tile[:-dif, :]
                        y2 = image_height

                    try:
                        image.paste(Image.fromarray(tile), (x, y, x2, y2))
                    except ValueError:
                        pass  # TODO fix this

            image.save(os.path.join(output_folder, file.removesuffix(".txt") + ".tif"))
