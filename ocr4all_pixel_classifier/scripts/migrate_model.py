import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def migrate_model(path_to_meta, n_classes, l_rate, output_path):
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session(graph=graph) as sess:
            # import graph
            saver = tf.compat.v1.train.import_meta_graph(path_to_meta)

            # load weights for graph
            saver.restore(sess, path_to_meta[:-5])

            # get all global variables (including model variables)
            vars_global = tf.compat.v1.global_variables()

            # get their name and value and put them into dictionary
            sess.as_default()
            model_vars = {}
            for var in vars_global:
                model_vars[var.name] = var.eval()

        from ocr4all_pixel_classifier.lib.metrics import fgpa, accuracy, loss

        input_image = tf.keras.layers.Input((None, None, 1))
        input_binary = tf.keras.layers.Input((None, None, 1))

        from ocr4all_pixel_classifier.lib.model import model_fcn_skip
        model = model_fcn_skip([input_image], n_classes)
        optimizer = tf.keras.optimizers.Adam(lr=l_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
        keys = list(model_vars.keys())
        counter = 0
        for l in model.layers:
            if len(l.get_weights()) > 0:
                l.set_weights([model_vars[keys[counter]], model_vars[keys[counter + 1]]])
                counter += 2
                pass
        model.save(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True,
                        help="path to the meta file of the tf1 model to convert")
    parser.add_argument("--output_path", type=str, required=True,
                        help="The output dir for the info files")
    parser.add_argument("--l_rate", default=1e-3, type=float,
                        help="Average height over all images")
    parser.add_argument("--n_classes", required=True, type=int)
    args = parser.parse_args()
    migrate_model(args.meta_path, args.n_classes, args.l_rate, args.output_path)


if __name__ == "__main__":
    main()
