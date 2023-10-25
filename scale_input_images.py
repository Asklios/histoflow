import os
import shutil
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 10000000000


def scale(input_folder, scale_factor):
    os.makedirs(os.path.join(input_folder, 'orig'), exist_ok=True)

    images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # print warning if any images are already scaled
    if any('_scaled' in image for image in images):
        print('WARNING: Some images are already scaled. These will be scaled again. Press any key to continue.')
        input()

    for image in images:
        input_image = Image.open(os.path.join(input_folder, image), 'r')
        input_image = input_image.resize((int(input_image.size[0] * scale_factor), int(input_image.size[1] * scale_factor)))

        # move original image to orig folder if not already there
        if not os.path.exists(os.path.join(input_folder, 'orig', image)):
            shutil.move(os.path.join(input_folder, image), os.path.join(input_folder, 'orig', image))

        # rename scaled image
        image = os.path.splitext(image)[0] + '_scaled.tif'

        input_image.save(os.path.join(input_folder, image))
        input_image.close()
