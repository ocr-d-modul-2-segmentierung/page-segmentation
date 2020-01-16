# Example call for training a model

# This trains a model based on the data in the given json files (see
# dataset-creation-example.sh)
#
# Note that the dataset files already differentiate between train,
# test/validation and evaluation dataset, so even if you specify datasets for
# both train and test, only the image in the relevant section in the JSON will
# be used for each group
#
# --n-epoch / -E specifies the number of epochs for training, i.e. the training
# duration. We also specify the number of iterations a worse performance is
# allowed, when the model does not improve for that long the training stops
# (only possible with test dataset). Of course we keep the best model, not the
# one after the performance drops.
#
# --output specifies the folder where model.h5 and and logs will be placed.
ocr4all-pixel-classifier train \
	--train dataset1.json dataset2.json dataset3.json \
	--test dataset1.json dataset2.json dataset3.json \
	--n-epoch 100 \
	--early-stopping-max-performance-drops 30 \
	--output my-model \
	--color_map image_map.json

# if using a split file:
ocr4all-pixel-classifier train \
	--split-file splits.json \
	-E 100 \
	-S 30 \
	--output my-model \
	--color_map image_map.json


# you can also use --load to specify an existing model on which to continue
# training. There are some minor tuning options available, see the help output
# for the "train" command for those:
ocr4all-pixel-classifier train --help
