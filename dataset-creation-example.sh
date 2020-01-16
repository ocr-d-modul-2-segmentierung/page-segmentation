#!/bin/sh

# Example of creating a dataset for ocr4all-pixel-classifier
#
# This assumes your folder structure looks like this:
# base dir
# ├── book1
# │   ├── binary  <- Binarized version of image
# │   ├── jpg     <- Color version of image
# │   └── page    <- PageXML
# ├── ...
# └── bookN
#    ├── binary
#    ├── jpg
#    └── page

for book in book*; do

	# Generate the masks for training
	# --setting defines, which region types will be used:
	#   - all_types: one color for each region type in PageXML
	#   - text_nontext: red for text, green for non_text
	#   - baseline: draw baselines of text lines
	#   - textline: draw polygons of text lines
	# we set --image_map_dir to the current directory, which will overwrite the
	# output in each loop, but the image map generated is constant for each
	# setting and only one file is needed for training
	ocr4all-pixel-classifier gen-masks \
		--input-dir $book/page \
		--output-dir $book/masks \
		--threads $(nproc) \
		--setting text_nontext \
		--image-map_dir ./

	# Estimate the xheight for all pages based on connected components in binary
	# image.
	# When running on images with different dpi/fontsize, don't use --average_all
	ocr4all-pixel-classifier compute-image-normalizations \
		--input-dir $book/binary
		--average-all \
		--output-dir $book/norms 

	# create a json file usable as input for the pixel classifier's train command
	# --n_train and --n_test determine the size of the training and validation set
	# (note that validation set is called test set in the pixel classifier for
	# historical reasons)
	ocr4all-pixel-classifier create-dataset-file \
		--images-dir jpg \
		--binary-dir binary \
		--masks-dir masks \ 
		--normalizations-dir norms \
		--output-file $book/dataset.json \
		--n-train 0.8 --n-test 0.2
		--dataset-path $(realpath $book/)
done
