import os
import numpy as np
import PIL
from PIL import Image
from joblib import Parallel, delayed

from helper import filter_supported_files

PIL.Image.MAX_IMAGE_PIXELS = 10000000000


# Function to load an image and its corresponding mask
def load_image_and_mask(image_path, mask_path):
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.repeat(2, axis=0).repeat(2, axis=1)  # double the size of the mask since it is 1/2 the size of the image
    # handle small missmatch in size between mask and image due to resizing
    mask_new = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    rows, cols = mask.shape
    rows = min(rows, mask_new.shape[0])
    cols = min(cols, mask_new.shape[1])
    mask_new[:rows, :cols] = mask[:rows, :cols]
    return image, mask_new


# Function to calculate the median RGB values based on the non-zero mask pixels
def calculate_median_rgb(image, mask):
    masked_pixels = image[mask != 0]
    median_rgb = np.median(masked_pixels, axis=0)
    return median_rgb


# Function to normalize the image using the individual median RGB and the overall median RGB values
def normalize_image(image, median_rgb, overall_median_rgb):
    normalized_image = (image.astype(np.float32) + (overall_median_rgb - median_rgb)) / 255.0
    return (normalized_image * 255.0).astype(np.uint8)


# Main function to normalize a set of images and save normalization values to a file
def normalize_images(images_path, masks_path, output_path):

    image_paths = [os.path.join(images_path, filename) for filename in filter_supported_files(images_path)]
    mask_paths = [os.path.join(masks_path, filename) for filename in filter_supported_files(masks_path)]

    num_images = len(image_paths)
    file_names = []  # List to store the original file names

    # Load images and masks in parallel
    images, masks = zip(
        *Parallel(n_jobs=-1)(delayed(load_image_and_mask)(image_paths[i], mask_paths[i]) for i in range(num_images)))

    # Calculate median RGB values for all images using masks
    median_rgbs = [calculate_median_rgb(image, mask) for image, mask in zip(images, masks)]
    overall_median_rgb = np.median(median_rgbs, axis=0)

    # Save median RGB values to a file
    np.savetxt(os.path.join(output_path, "median_rgb_values.txt"), overall_median_rgb)

    # Normalize each image using its median RGB and the overall median RGB
    normalized_images = [normalize_image(image, median_rgbs[i], overall_median_rgb) for i, image in enumerate(images)]

    # Save normalization values (median differences) and file names to a file
    median_differences = median_rgbs - overall_median_rgb

    for i, img_path in enumerate(image_paths):
        # Extract the original file name without extension
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        file_name = f'{file_name}_normalized.tif'
        file_names.append(file_name)

        Image.fromarray(normalized_images[i]).save(os.path.join(output_path, file_name))

    # Save file names alongside their corresponding median difference values
    with open(os.path.join(output_path, "file_names_with_median_differences.txt"), "w") as file:
        for name, median_diff in zip(file_names, median_differences):
            file.write(f"{name}: {', '.join(map(str, median_diff))}\n")


# Function to normalize an image using previously calculated normalization values
def normalize_images_using_values(image_paths, mask_paths, normalization_values, output_path):
    num_images = len(image_paths)

    # Load images and masks in parallel
    images, masks = zip(
        *Parallel(n_jobs=-1)(delayed(load_image_and_mask)(image_paths[i], mask_paths[i]) for i in range(num_images)))

    # Calculate median RGB values for all images using masks
    median_rgbs = [calculate_median_rgb(image, mask) for image, mask in zip(images, masks)]

    # Normalize each image using its median RGB and the overall median RGB
    normalized_images = [normalize_image(image, median_rgbs[i], normalization_values) for i, image in enumerate(images)]

    # Save normalization values (median differences) and file names to a file
    median_differences = median_rgbs - normalization_values

    file_names = []
    for i, img_path in enumerate(image_paths):
        # Extract the original file name without extension
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        file_name = f'{file_name}_normalized.tif'
        file_names.append(file_name)

        Image.fromarray(normalized_images[i]).save(os.path.join(output_path, file_name))

    # append to file
    with open(os.path.join(output_path, "file_names_with_median_differences.txt"), "a") as file:
        for name, median_diff in zip(file_names, median_differences):
            file.write(f"{name}: {', '.join(map(str, median_diff))}\n")



