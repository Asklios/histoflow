# Histoflow
## Segmentation of histological images through machine learning.
![Languages](https://img.shields.io/github/languages/top/Asklios/histoflow.svg)
![Modified](https://img.shields.io/github/last-commit/Asklios/histoflow.svg)
[![License](https://img.shields.io/github/license/Asklios/histoflow.svg)](https://github.com/Asklios/histoflow/blob/master/LICENSE)

### Available commands:
- `-d` or `--data` to create a new data folder or define an existing one
- `--scale` to scale the images by a defined factor
- `-m` or `--mask` to create a mask for the images
  - add option `input` if you only want to create mask input images
  - add option `mask` if you only want to create masks from existing input images
- `-n` or `--normalize` to normalize the images
- `-t` or `--tiles` to create tiles from the images, this takes two int values
  - the first is the `<tile size>`
  - the second is the `<overlap size>`
- `-c` or `--classifier` to train a classifier. This requires two images in the data/training folder.
One must be RGB and one a mask.
- `-s` or `--segmentation` to segment all created tiles. This requires a trained classifier.
- `-o` or `--output` to stitch the segmented tiles back together.
- `-v` or `--visualize` to visualize the segmentation process, this takes two int values
  - the first is the `<tile size>`
  - the second is the `<overlap size>`

### Suggested workflow:
1. Create a new data folder with the `-d` or `--data` command.
2. Copy your images into the data/input folder.
3. Scale the images with the `--scale` command if necessary.
4. Create a mask for the images with the `-m` or `--mask` command.
5. Normalize the images with the `-n` or `--normalize` command.
6. Create training data out of the normalized images. It consists of two images, one RGB and one binary mask.
7. Copy the training data into the data/training folder. There should be two images, one RGB and one binary mask.
8. Train a classifier with the `-c` or `--classifier` command.
9. Segment the images with the `-s` or `--segmentation` command.
10. Stitch the segmented tiles back together with the `-o` or `--output` command.
11. Optionally visualize the segmentation process with the `-v` or `--visualize` command. 
12. See the results in the data/output folder.

You can combin multiple commands in one call, e.g.:
```
python histoflow.py -d -m -n -t 512 64 -c -s -o -v 512 64
```

### Please cite the author if you use this code:
  ```
  @software{JakobVoerkelius.2023,
    author = {Jakob Voerkelius},
    title = {Histoflow},
    titleaddon = {Segmentation of histological images through machine learning},
    version = {1.0.0},
    year = {2023},
    url = {https://github.com/Asklios/histoflow},
    orcid = {0009-0003-1630-2265},
    license = {MIT}
  }