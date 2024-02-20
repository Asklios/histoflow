import argparse
import os
import PIL
import numpy as np
from PIL import Image

from helper import notify, request_yes_no, filter_supported_files
from normalize_images import normalize_images, normalize_images_using_values
from scale_input_images import scale
from create_mask import create_mask_input, create_mask
from create_tiles import create_tiles
from train_classifier import train_classifier
from segmentation import segment_tiles
from revert_tiles import revert_tiles
from visualization_create_tiles import create_visualization

PIL.Image.MAX_IMAGE_PIXELS = 10000000000

parser = argparse.ArgumentParser(description='Pixel classification using scikit-learns Random Forest Classifier')
parser.add_argument('-d', '--data', nargs='?', const='.', help='Create data directory')
parser.add_argument('--scale', type=float, help='Scale factor for input images')
parser.add_argument('-m', '--mask', nargs='?', const='', help='Create masks from input images',
                    choices=['', 'input', 'mask'])
parser.add_argument('-n', '--normalize', action='store_true', help='Normalize input images')
parser.add_argument('-t', '--tiles', nargs=2, type=int, help='Specify tile and overlap size in pixels as two integer '
                                                             'values')
parser.add_argument('-c', '--classifier', action='store_true', help='Create classifier from training data')
parser.add_argument('-s', '--segmentation', action='store_true', help='Create segmentation from input images')
parser.add_argument('-o', '--output', action='store_true', help='Create output images from segmentation')
parser.add_argument('-v', '--visualization', nargs=2, type=int, help='Create visualization of workflow, specify tile '
                                                                     'and overlap size in pixels as two integer values')

args = parser.parse_args()


# start with input and file validation
if args.tiles:
    tile_size, overlap = args.tiles

    if overlap > tile_size:
        print('Are you sure you want to use an overlap larger than the tile size? (y/n)')
        if not request_yes_no():
            print('Skipping tile creation')
            exit(0)

data_folders = ['input', 'training', 'mask_input', 'mask', 'normalized', 'tiles', 'segmentation', 'output',
                'visualization']

if args.data is None:
    args.data = './data'
    if os.path.exists(args.data):
        print('Data directory found, creating missing subdirectories')
        for folder in data_folders:
            os.makedirs(os.path.join(args.data, folder), exist_ok=True)
    else:
        print('Data directory not found, create it? (y/n)')
        if request_yes_no():
            print('Creating data directory')
            os.makedirs('data', exist_ok=True)
            for folder in data_folders:
                os.makedirs(os.path.join('data', folder), exist_ok=True)
        else:
            print('Exiting')
            exit(0)
elif args.data == '.':
    args.data = './data'
    print('Creating data directory')
    os.makedirs('data', exist_ok=True)
    for folder in data_folders:
        os.makedirs(os.path.join('data', folder), exist_ok=True)
elif os.path.exists(args.data):
    print('Data directory found, creating subdirectories if missing')
    for folder in data_folders:
        os.makedirs(os.path.join(args.data, folder), exist_ok=True)
else:
    print('Data directory not found, check path')
    exit(1)

full_path = os.path.abspath(args.data)
INPUT_FOLDER = os.path.join(full_path, 'input')
TRAINING_FOLDER = os.path.join(full_path, 'training')
MASK_INPUT_FOLDER = os.path.join(full_path, 'mask_input')
MASK_FOLDER = os.path.join(full_path, 'mask')
NORMALIZED_FOLDER = os.path.join(full_path, 'normalized')
TILES_FOLDER = os.path.join(full_path, 'tiles')
SEGMENTATION_FOLDER = os.path.join(full_path, 'segmentation')
OUTPUT_FOLDER = os.path.join(full_path, 'output')
VISUALIZATION_FOLDER = os.path.join(full_path, 'visualization')

if args.scale:
    SCALE = args.scale
    print('Scaling all images by factor {}'.format(SCALE))
    scale(INPUT_FOLDER, SCALE)
    print('Done scaling images')
    notify('Server finished scaling images')

if args.mask is not None:
    if args.mask == '':
        print('Creating mask input')
        create_mask_input(INPUT_FOLDER, MASK_INPUT_FOLDER)
        print('Done creating mask input')
        print('Creating masks')
        create_mask(MASK_INPUT_FOLDER, MASK_FOLDER)
        print('Done creating masks')
        notify('Server finished creating masks')
    elif args.mask == 'input':
        print('Creating mask input')
        create_mask_input(INPUT_FOLDER, MASK_INPUT_FOLDER)
        print('Done creating mask input')
        notify('Server finished creating mask input')
    elif args.mask == 'mask':
        print('Creating masks')
        create_mask(MASK_INPUT_FOLDER, MASK_FOLDER)
        print('Done creating masks')
        notify('Server finished creating masks')
    else:
        print('Unknown mask option. Use -h for help. Skipping mask creation.')

if args.normalize:
    # check if there is a mask for every input image
    if len(filter_supported_files(MASK_FOLDER)) is not len(filter_supported_files(INPUT_FOLDER)):
        print('There must be a mask for every input image to apply normalization. Exiting.')
        exit(1)

    if len(os.listdir(NORMALIZED_FOLDER)) > 0:
        print('Normalized folder is not empty. Do you want to overwrite existing images? (y/n)')
        if not request_yes_no():
            if len(filter_supported_files(INPUT_FOLDER)) is not len(filter_supported_files(NORMALIZED_FOLDER)):
                print('There is not a normalized image for every input image. '
                      'Applying normalization to missing images')

                # load overall median RGB values
                if not os.path.exists(os.path.join(NORMALIZED_FOLDER, "median_rgb_values.txt")):
                    print('There is no file with overall median RGB values. '
                          'Please normalize all images before normalizing missing images. Exiting.')
                    exit(1)
                overall_median_rgb = np.loadtxt(os.path.join(NORMALIZED_FOLDER, "median_rgb_values.txt"))

                # find missing images
                missing_images = []
                for file in filter_supported_files(INPUT_FOLDER):
                    if file not in os.listdir(NORMALIZED_FOLDER):
                        missing_images.append(os.path.join(INPUT_FOLDER, file))

                # find corresponding masks
                missing_masks = []
                for image_path in missing_images:
                    image_filename = os.path.basename(image_path)
                    mask_filename = image_filename.replace('.tif', '_mask.tif')
                    mask_path = os.path.join(MASK_FOLDER, mask_filename)

                    # check if mask exists
                    if not os.path.exists(mask_path):
                        print('No mask found for image {}. Please create mask before normalizing missing images. '
                              'Exiting.'.format(image_path))
                        exit(1)

                    missing_masks.append(mask_path)

                # normalize missing images
                normalize_images_using_values(missing_images, missing_masks, overall_median_rgb, NORMALIZED_FOLDER)
                print('Done normalizing missing images')
                notify('Server finished normalizing missing images')
                INPUT_FOLDER = NORMALIZED_FOLDER

            else:
                print('Skipping normalization. Using existing, normalized images.')
                INPUT_FOLDER = NORMALIZED_FOLDER
        else:
            print('Normalizing images')
            # delete existing images
            for file in os.listdir(NORMALIZED_FOLDER):
                os.remove(os.path.join(NORMALIZED_FOLDER, file))
            normalize_images(INPUT_FOLDER, MASK_FOLDER, NORMALIZED_FOLDER)
            INPUT_FOLDER = NORMALIZED_FOLDER
            print('Done normalizing images')
            notify('Server finished normalizing images')
    else:
        # check if there are images in the INPUT_FOLDER
        if len(filter_supported_files(INPUT_FOLDER)) == 0:
            print('No images found in input folder. Exiting.')
            exit(1)

        print('Normalizing images')
        normalize_images(INPUT_FOLDER, MASK_FOLDER, NORMALIZED_FOLDER)
        print('Done normalizing images')
        notify('Server finished normalizing images')

if args.tiles:
    tile_size, overlap = args.tiles
    print('Creating tiles with size {} and overlap {}'.format(tile_size, overlap))

    # check if tile folder is empty
    if len(os.listdir(TILES_FOLDER)) > 0:
        print('Tile folder is not empty. Do you want to overwrite existing tiles? (y/n)')
        if not request_yes_no():
            print('Skipping tile creation')
        else:
            create_tiles(INPUT_FOLDER, MASK_FOLDER, TILES_FOLDER, tile_size, overlap)
            print('Done creating tiles')
            notify('Server finished creating tiles')
    else:
        create_tiles(INPUT_FOLDER, MASK_FOLDER, TILES_FOLDER, tile_size, overlap)
        print('Done creating tiles')
        notify('Server finished creating tiles')

if args.classifier:
    print('Training classifier')

    # check if training data is available
    if len([file for file in os.listdir(TRAINING_FOLDER) if os.path.isfile(os.path.join(TRAINING_FOLDER, file))]) == 0:
        print('No training data found. Place image and mask in the training folder and try again. Skipping training.')
    elif len([file for file in os.listdir(TRAINING_FOLDER) if os.path.isfile(os.path.join(TRAINING_FOLDER, file))]) > 2:
        print('More than two files found in the training folder. Please check your training data and try again. '
              'The training folder should contain one RGB image and one mask. Skipping training.')
    else:
        # check if one of the files has 3 layers and the other one has 1 layer
        files = [file for file in os.listdir(TRAINING_FOLDER) if os.path.isfile(os.path.join(TRAINING_FOLDER, file))]
        file1 = os.path.join(TRAINING_FOLDER, files[0])
        file2 = os.path.join(TRAINING_FOLDER, files[1])

        im = Image.open(file1, 'r')
        img1 = np.array(im)
        im = Image.open(file2, 'r')
        img2 = np.array(im)

        if len(img1.shape) == 3 and len(img2.shape) == 2:
            image_file = file1
            mask_file = file2
            print('Starting training with image {} and mask {}'.format(image_file, mask_file))
            train_classifier(image_file, mask_file, os.path.join(full_path, 'classifier.joblib'))
            print('Done training classifier')
            notify('Server finished training classifier')
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            image_file = file2
            mask_file = file1
            print('Starting training with image {} and mask {}'.format(image_file, mask_file))
            train_classifier(image_file, mask_file, os.path.join(full_path, 'classifier.joblib'))
            print('Done training classifier')
            notify('Server finished training classifier')
        else:
            print('Could not find image and mask pair. Please check your training data and try again. '
                  'The training folder should contain one RGB image and one mask. Currently found: '
                  '{} with shape {} and {} with shape {}. '
                  'Skipping training.'.format(files[0], img1.shape, files[1], img2.shape))

if args.segmentation:
    print('Starting segmentation')

    classifier = os.path.join(full_path, 'classifier.joblib')

    if not os.path.exists(classifier):
        print('No classifier found. Please train a classifier before segmenting images. Exiting.')
        exit(1)

    segment_tiles(TILES_FOLDER, SEGMENTATION_FOLDER, classifier)
    print('Done segmenting tiles')
    notify('Server finished segmenting tiles')

if args.output:
    print('Stitching tiles together and exporting to output folder')
    revert_tiles(SEGMENTATION_FOLDER, OUTPUT_FOLDER)
    print('Done creating output')
    notify('Server finished creating output')

if args.visualization:
    tile_size, overlap = args.visualization
    print('Creating visualization with tile size {} and an overlap of {}'.format(tile_size, overlap))
    create_visualization(INPUT_FOLDER, MASK_FOLDER, VISUALIZATION_FOLDER, tile_size, overlap)
    print('Done creating visualization')
    notify('Server finished creating visualization')

print('All done!')
