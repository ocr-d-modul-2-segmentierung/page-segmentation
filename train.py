import tensorflow as tf
import model
import numpy as np
from dataset import load_test, label_to_colors
import matplotlib.pyplot as plt
import skimage.io as img_io
import os
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='./saved_model/FCN-on-OCR-D')
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--n_iter", type=int, default=30000)

args = parser.parse_args()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


l_rate = 0.005

binary_inputs = tf.placeholder(tf.int32, (None, None, None), "binary_inputs")
inputs = tf.placeholder(tf.float32, (None, None, None), "inputs")
masks = tf.placeholder(tf.int32, (None, None, None), "masks")

batch_size = tf.shape(inputs)[0]

prediction, logits, probs = model.model(tf.expand_dims(inputs, -1))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=masks,
    logits=logits))
train_op = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=0.9).minimize(loss)

equals = tf.equal(tf.cast(prediction, tf.int32), masks)
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

fgpa_correct = tf.reduce_sum(tf.reshape(tf.multiply(tf.cast(equals, tf.int32), binary_inputs), (batch_size, -1)), axis=-1)
fgpa_total = tf.reduce_sum(tf.reshape(binary_inputs, (batch_size, -1)), axis=-1)

fgpa = tf.reduce_mean(tf.divide(fgpa_correct, fgpa_total))

train_data, test_data = load_test()

saver = tf.train.Saver()

debug_plot = False

with tf.Session() as sess:
    if args.load:
        saver.restore(sess, args.load)
    else:
        sess.run(tf.global_variables_initializer())

    avg_loss, avg_acc, avg_fgpa = 10, 0, 0
    for step in range(args.n_iter):
        # only bs one, cause images might have a different shape
        samples = np.random.randint(0, len(train_data), 1).tolist()
        train_binary, train_input, train_mask = \
            zip(*[(t["binary"], t["image"], t["mask"]) for t in [train_data[s] for s in samples]])
        _, l, a, fg = sess.run((train_op, loss, accuracy, fgpa),
                           {inputs: train_input, masks: train_mask,
                            binary_inputs: train_binary})
        avg_loss = 0.99 * avg_loss + 0.01 * l
        avg_acc = 0.99 * avg_acc + 0.01 * a
        avg_fgpa = 0.99 * avg_fgpa + 0.01 * fg
        print("#%05d (%.5f): Acc=%.5f FgPA=%.5f" % (step, avg_loss, avg_acc, avg_fgpa))


        if (step + 1) % 5000 == 0:
            print("Saving the model")
            saver.save(sess, args.output, global_step=step + 1)



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

    if len(test_data) > 0:
        compute_total("Test", test_data, output_dir="prediction_GW5064_pretrained-OCR-D")

