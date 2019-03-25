# Page Segmentation

## Requirements

Python dependencies are specified in `requirements.txt` / `setup.py`.

You must install the package via pip with either `page_segmentation[tf_cpu]` to
use CPU version of tensorflow or `page_segmentation[tf_gpu]` to use GPU (CUDA)
version of tensorflow. For the latter, your system should be set up with CUDA 9
and CuDNN 7.

## Usage

### Pixel classifier

#### Classification

To run a model on some input images, use `page-segmentation predict`:

```sh
page-segmentation predict --load PATH_TO_MODEL \
	--output OUTPUT_PATH \
	--binary PATH_TO_BINARY_IMAGES \
	--images PATH_TO_SOURCE_IMAGES \
	--norm PATH_TO_NORMALIZATIONS
```
(`ocrd_pixel_classifier` is an alias for `page-segmentation predict`)

This will create three folders at the output path:
- `color`: the classification as color image, with pixel color corresponding to
	the class for that pixel
- `inverted`: inverted binary image with classification of foreground pixels
	only (i.e. background is black, foreground is white or class color)
- `overlay`: classification image layered transparently over the original image

#### Training

For training, you first have to create dataset files. A dataset file is a JSON
file containing three arrays, for train, test and evaluation data (also
called train/validation/test in other publications). The JSON file uses the
following format:

```json
{
	"train": [
		//datasets here
	],
	"test": [
		//datasets here
	],
	"eval": [
		//datasets here
	]
}
```

A dataset describes a single input image and consists of several paths: the
original image, a binarized version and the mask (pixel color corresponds to
class). Furthermore, the line height of the page in pixels must be specified:
```json
{
	"binary_path": "/path/to/image/binary/filename.bin.png",
	"image_path":  "/path/to/image/color/filename.jpg",
	"mask_path":  "/path/to/image/mask/filename_MASK.png",
	"line_height_px": 18
}
```

The generation of dataset files can be automated using `page-segmentation
create-dataset-file`. Refer to the command's `--help` output for further
information.

To start the training:

```sh
page-segmentation train \
    --train DATASET_FILE.json --test DATASET_FILE.json --eval DATASET_FILE.json \
    --output MODEL_TARGET_PATH \
    --n_iter 5000
```
The parameters `--train`, `--test` and `--eval` may be followed by any number of
dataset files or patterns (shell globbing).

Refer to `page-segmentation train --help` for further parameters provided to
affect the training procedure.

You can combine several dataset files into a _split file_. The format of the
split file is:

```json
{
	"label": "name of split",
	"train": [
		"/path/to/dataset1.json",
		"/path/to/dataset2.json",
		...
	],
	"test": [
		//dataset paths here
	],
	"eval": [
		//dataset paths here
	]
}
```
To use a split file, add the `--split_file` parameter.

### `page-segmentation compute-image-normalizations` / `ocrd_compute_normalizations`

Calculate image normalizations, i.e. scaling factors based on average line
height.

Required arguments:

- `--input_dir`: location of images
- `--output_dir`: target location of norm files

Optional arguments:
- `--average_all`: Average height over all images
- `--inverse`
