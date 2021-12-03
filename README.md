# Duplicate (Similar) Images Removal

A simple script that can remove similar images by comparing their ImageHash (Details here: https://github.com/JohannesBuchner/imagehash)

## Installation
This script depends on the following libraries:
```
tqdm
numpy
Pillow
hnswlib
imagehash
```
All of them can be installed via pypi

## Usage 
```
> python imagehash_v2.py -h

usage: imagehash_v2.py [-h] [-c COMPARE] [-t TOLERANCE] [-ef HNSW_EF]
                       [-m HNSW_M] [-b] [-d]
                       [-A | -P | -D | -WH | -WD | -C | -CR]
                       origin

Using different ImageHash method to find the duplicate images in a given
folder

positional arguments:
  origin                the origin folder of all processing files

optional arguments:
  -h, --help            show this help message and exit
  -c COMPARE, --compare COMPARE
                        the folder need to compare with origin
  -t TOLERANCE, --tolerance TOLERANCE
                        Hashing difference tolerance (Default: 1)
  -ef HNSW_EF, --hnsw_ef HNSW_EF
                        hnswlib ef_construction param, ADJUST WITH CAUTION
                        (Default: 200)
  -m HNSW_M, --hnsw_m HNSW_M
                        hnswlib M param, ADJUST WITH CAUTION (Default: 16)
  -b, --backup          create backups for original images (Default: True)
  -d, --delete          delete all duplicates (Default) (otherwise copied to
                        subdirectories)
  -A, --Average         Average hashing
  -P, --Perceptual      Perceptual hashing (Default)
  -D, --Difference      Difference hashing
  -WH, --WaveHaar       Wavelet hashing - haar
  -WD, --WaveDB4        Wavelet hashing - db4
  -C, --Color           HSV color hashing
  -CR, --Crop           Crop-resistant hashing (SLOW)
```

## v2?
Version 2 utilizes hnswlib (Details here: https://github.com/nmslib/hnswlib) for faster indexing and querying speed.
**hnswlib has some params that need to be adjusted (and sometimes it's hard)**

Version 1 is just a brutal force grouping, 
it's included in the `ImageHashProcessor` class, 
but that's not implemented in the command line tool.

You can simply replace all `hnsw_group` with `brutal_force_group` to use the brutal force grouping.

## Something is wrong!
Yes, it's not perfect. Here's some issues you may encouter:

### Extensions:
The hard-coded extensions are: 
```
image_suffixes = ('jpg', 'JPG',
                  'jpeg', 'JPEG',
                  'png', 'PNG',
                  'gif', 'GIF',
                  'bmp', 'BMP')
```
since I believe that's all Pillow (Thus, ImageHash library) supported.

**For Windows Users:**

The default image extensions are case-sensetive, BUT Windows is **NOT**!

You need to manually adjust the extension part, delete all **UPPERCASE** extensions ones.
(It's at the end of the imagehash_v2.py file)

### hnswlib related:
As pointed out in the previous part, you could (maybe should) adjust some default settings (`ef_construction` and `M`) using the arugments.

Please refer to the HNSW algorithm parameters page for their explanations: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md