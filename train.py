import tensorflow as tf
import model
import numpy as np
from dataset import DatasetLoader, label_to_colors
import matplotlib.pyplot as plt
import skimage.io as img_io
import os
import tqdm
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--l_rate", type=float, default=1e-2)
parser.add_argument("--l_rate_drop_factor", type=float, default=0.1)
parser.add_argument("--n_classes", type=int, default=4)
parser.add_argument("--target_line_height", type=int, default=6,
                    help="Scale the data images so that the line height matches this value")
parser.add_argument("--output", type=str, default='./saved_model/FCN-All-for-GW5064')
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--n_iter", type=int, default=300000)
parser.add_argument("--early_stopping_test_interval", type=int, default=1000)
parser.add_argument("--early_stopping_max_keep", type=int, default=10)
parser.add_argument("--early_stopping_max_l_rate_drops", type=int, default=3)
parser.add_argument("--prediction_dir", type=str, default="prediction-All-for-GW5064")
parser.add_argument("--split_file", type=str, default="split.json",
                    help="Load splits from a json file")
parser.add_argument("--train", type=str, nargs="*", default=[])
parser.add_argument("--test", type=str, nargs="*", default=[],
                    help="Data used for early stopping"
)
parser.add_argument("--eval", type=str, nargs="*", default=[])

args = parser.parse_args()

# json file for splits
if args.split_file:
    with open(args.split_file) as f:
        d = json.load(f)
        args.train += d["train"]
        args.test += d["test"]
        args.eval += d["eval"]



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


l_rate = tf.placeholder(tf.float32, None, "l_rate")
binary_inputs = tf.placeholder(tf.int32, (None, None, None), "binary_inputs")
inputs = tf.placeholder(tf.float32, (None, None, None), "inputs")
masks = tf.placeholder(tf.int32, (None, None, None), "masks")

batch_size = tf.shape(inputs)[0]

prediction, logits, probs = model.model(tf.expand_dims(inputs, -1), args.n_classes)
loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
    labels=masks,
    logits=logits,
    # weights=10 * binary_inputs,
))
optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=0.9)
gradients = optimizer.compute_gradients(loss)
gradients = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gradients]
train_op = optimizer.apply_gradients(gradients, name="train_op")

equals = tf.equal(tf.cast(prediction, tf.int32), masks)
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

fgpa_correct = tf.reduce_sum(tf.reshape(tf.multiply(tf.cast(equals, tf.int32), binary_inputs), (batch_size, -1)), axis=-1)
fgpa_total = tf.reduce_sum(tf.reshape(binary_inputs, (batch_size, -1)), axis=-1)

fgpa = tf.reduce_mean(tf.divide(fgpa_correct, fgpa_total))

print("Loading data")
dataset_loader = DatasetLoader(args.target_line_height)
train_data = dataset_loader.load_data_from_json(args.train, "train")
test_data = dataset_loader.load_data_from_json(args.train, "test")
eval_data = dataset_loader.load_data_from_json(args.eval, "eval")

if len(train_data) == 0 and args.n_iter > 0:
    raise Exception("No training files specified. Maybe set n_iter=0")

debug_plot = False

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    def compute_pgpa(data):
        total_a, total_fg = 0, 0
        for i, sample in tqdm.tqdm(enumerate(data), total=len(data)):
            a, fg, pred = sess.run((accuracy, fgpa, prediction),
                                   {inputs: [sample["image"]], masks: [sample["mask"]],
                                    binary_inputs: [sample["binary"]]})

            total_a += a / len(data)
            total_fg += fg / len(data)

        return total_fg

    if args.load:
        all_var_names = [v for v in tf.global_variables()
                         if "Momentum" not in v.name
                         and "Adam" not in v.name and "beta1_power" not in v.name and "beta2_power" not in v.name]
        print(all_var_names)
        saver = tf.train.Saver(all_var_names)
        saver.restore(sess, args.load)

    saver = tf.train.Saver()

    current_best_fgpa = 0
    current_best_model_iter = 0
    current_best_iters = 0
    l_rate_drops = 0
    avg_loss, avg_acc, avg_fgpa = 10, 0, 0
    for step in range(args.n_iter):
        # only bs one, cause images might have a different shape
        samples = np.random.randint(0, len(train_data), 1).tolist()
        train_binary, train_input, train_mask = \
            zip(*[(t["binary"], t["image"], t["mask"]) for t in [train_data[s] for s in samples]])
        gs, _, l, a, fg = sess.run((gradients, train_op, loss, accuracy, fgpa),
                           {inputs: train_input, masks: train_mask,
                            binary_inputs: train_binary,
                            l_rate: args.l_rate,})
        # m = max([np.abs(np.mean(g)) for g, _ in gs])
        # print(m)
        avg_loss = 0.99 * avg_loss + 0.01 * l
        avg_acc = 0.99 * avg_acc + 0.01 * a
        avg_fgpa = 0.99 * avg_fgpa + 0.01 * fg
        print("#%05d (%.5f): Acc=%.5f FgPA=%.5f" % (step, avg_loss, avg_acc, avg_fgpa))

        if (step + 1) % args.early_stopping_test_interval == 0:
            print("checking for early stopping")
            test_fgpa = compute_pgpa(test_data)
            if test_fgpa > current_best_fgpa:
                current_best_fgpa = test_fgpa
                current_best_model_iter = step + 1
                current_best_iters = 0
                print("New best model at iter {} with FgPA={}".format(current_best_model_iter, current_best_fgpa))

                print("Saving the model")
                saver.save(sess, args.output, global_step=current_best_model_iter)
            else:
                current_best_iters += 1
                print("No new best model found. Current iterations {} with FgPA={}".format(current_best_iters, current_best_fgpa))
                print("{} learning rates drops to go.".format(args.early_stopping_max_l_rate_drops - l_rate_drops))

            if current_best_iters >= args.early_stopping_max_keep:
                if l_rate_drops >= args.early_stopping_max_l_rate_drops:
                    print('early stopping at %d' % (step + 1))
                    break
                else:
                    l_rate_drops += 1
                    args.l_rate *= args.l_rate_drop_factor
                    current_best_iters = 0
                    print('dropping learning rate to {}'.format(args.l_rate))

        if debug_plot:
            if step % 10 != 0:
                continue
            eq, p, lit = sess.run((equals, prediction, logits),
                               {inputs: train_input, masks: train_mask,
                                binary_inputs: train_binary})
            fix, ax = plt.subplots(3, 3)
            ax[0, 0].imshow(train_binary[0], cmap="gray")
            ax[1, 0].imshow(train_input[0], cmap="gray")
            ax[2, 0].imshow(train_mask[0], vmin=0, vmax=2)
            ax[1, 1].imshow(eq[0], cmap="gray")
            ax[2, 1].imshow(p[0], vmin=0, vmax=2)
            ax[0, 2].imshow(lit[0][:,:,0], vmin=0,vmax=1, cmap="gray")
            ax[1, 2].imshow(lit[0][:,:,1], vmin=0,vmax=1, cmap="gray")
            ax[2, 2].imshow(lit[0][:,:,2], vmin=0,vmax=1, cmap="gray")

            plt.show()

    print("Best model at iter {} with fgpa of {}".format(current_best_model_iter, current_best_fgpa))

    if current_best_model_iter > 0:
        print("Loading best model")
        saver.restore(sess, args.output + "-%d" % current_best_model_iter)

    def compute_total(label, data, output_dir=None):
        print("Computing total error of {}".format(label))
        if output_dir:
            mkdir(os.path.join(output_dir, "overlay"))
            mkdir(os.path.join(output_dir, "color"))
            mkdir(os.path.join(output_dir, "inverted"))

        total_a, total_fg = 0, 0
        for i, sample in tqdm.tqdm(enumerate(data), total=len(data)):
            a, fg, pred = sess.run((accuracy, fgpa, prediction),
                            {inputs: [sample["image"]], masks: [sample["mask"]],
                             binary_inputs: [sample["binary"]]})

            total_a += a / len(data)
            total_fg += fg / len(data)

            if output_dir:
                filename = os.path.basename(sample["image_path"])
                color_mask = label_to_colors(pred[0])
                foreground = np.stack([(1 - sample["image"])] * 3, axis=-1)
                inv_binary = np.stack([(sample["binary"])] * 3, axis=-1)
                overlay_mask = np.ndarray.astype(color_mask * foreground, dtype=np.uint8)
                inverted_overlay_mask = np.ndarray.astype(color_mask * inv_binary, dtype=np.uint8)
                img_io.imsave(os.path.join(output_dir, "color", filename), color_mask)
                img_io.imsave(os.path.join(output_dir, "overlay", filename), overlay_mask)
                img_io.imsave(os.path.join(output_dir, "inverted", filename), inverted_overlay_mask)


        print("%s: Acc=%.5f FgPA=%.5f" % (label, total_a, total_fg))

    compute_total("Train", train_data)
    compute_total("Test", test_data)

    if len(eval_data) > 0 and args.prediction_dir:
        compute_total("Eval", eval_data, output_dir=args.prediction_dir)

    else:
        compute_total("Eval", eval_data)

