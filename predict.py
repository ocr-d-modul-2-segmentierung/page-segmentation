import argparse
import os
import glob
from dataset import DatasetLoader, label_to_colors
import model
import tqdm
import tensorflow as tf
import numpy as np
import skimage.io as img_io

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, required=True,
                    help="Model to load")
parser.add_argument("--data_dir", type=str, required=True,
                    help="Data dir containing binary_images and images")
parser.add_argument("--char_height", type=int, required=True,
                    help="Average height of character m or n, ...")
parser.add_argument("--target_line_height", type=int, default=6,
                    help="Scale the data images so that the line height matches this value (must be the same as in training)")
parser.add_argument("--output", required=True,
                    help="Output dir")
parser.add_argument("--binary_dir", type=str, default="binary_images",
                    help="directory name of the binary images")
parser.add_argument("--images_dir", type=str, default="images",
                    help="directory name of the images on which to train")
args = parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdir(args.output)

images_dir = os.path.join(args.data_dir, args.images_dir)
binary_dir = os.path.join(args.data_dir, args.binary_dir)
if not os.path.exists(images_dir) or not os.path.exists(binary_dir):
    raise Exception("Both {} and {} must exists and contain the desired image files".format(images_dir, binary_dir))

image_file_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))]
binary_file_paths = [os.path.join(binary_dir, f) for f in sorted(os.listdir(binary_dir))]

if len(image_file_paths) != len(binary_file_paths):
    raise Exception("Got {} images but {} binary images".format(len(image_file_paths), len(binary_file_paths)))

print("Loading {} files with character height {}".format(len(image_file_paths), args.char_height))


dataset_loader = DatasetLoader(args.target_line_height, prediction=True)
data = dataset_loader.load_data(
    [{"binary_path": b, "image_path": i, "line_height_px": args.char_height} for b, i in zip(binary_file_paths, image_file_paths)]
)


print("Creating net")

graph = tf.Graph()
graph.as_default()

with tf.Session(graph=graph) as sess:
    saver = tf.train.import_meta_graph(args.load + '.meta')
    saver.restore(sess, args.load)
    inputs = graph.get_tensor_by_name("inputs:0")
    try:
        prediction = graph.get_tensor_by_name("prediction:0")
    except:
        prediction = graph.get_tensor_by_name("ArgMax:0")

    mkdir(os.path.join(args.output, "overlay"))
    mkdir(os.path.join(args.output, "color"))
    mkdir(os.path.join(args.output, "inverted"))

    print("Starting prediction")
    for i, sample in tqdm.tqdm(enumerate(data), total=len(data)):
        pred, = sess.run((prediction, ),
                               {inputs: [sample["image"]]})

        filename = os.path.basename(sample["image_path"])
        color_mask = label_to_colors(pred[0])
        foreground = np.stack([(1 - sample["image"])] * 3, axis=-1)
        inv_binary = np.stack([(sample["binary"])] * 3, axis=-1)
        overlay_mask = np.ndarray.astype(color_mask * foreground, dtype=np.uint8)
        inverted_overlay_mask = np.ndarray.astype(color_mask * inv_binary, dtype=np.uint8)
        img_io.imsave(os.path.join(args.output, "color", filename), color_mask)
        img_io.imsave(os.path.join(args.output, "overlay", filename), overlay_mask)
        img_io.imsave(os.path.join(args.output, "inverted", filename), inverted_overlay_mask)

