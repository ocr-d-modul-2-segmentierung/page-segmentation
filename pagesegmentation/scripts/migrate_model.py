import tensorflow as tf


def migrate_model(path_to_meta, n_classes, l_rate, output_path):

    with tf.Session() as sess:

        # import graph
        saver = tf.train.import_meta_graph(path_to_meta)

        # load weights for graph
        saver.restore(sess, path_to_meta[:-5])

        # get all global variables (including model variables)
        vars_global = tf.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_vars = {}
        for var in vars_global:
            try:
                model_vars[var.name] = var.eval()
            except:
                print("For var={}, an exception occurred".format(var.name))
    pass

    def loss(y_true, y_pred):
        y_true = tf.keras.backend.reshape(y_true, (-1,))
        y_pred = tf.keras.backend.reshape(y_pred, (-1, n_classes))
        return tf.keras.backend.mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred,
                                                                                     from_logits=True))

    def accuracy(y_true, y_pred):
        y_true = tf.keras.backend.reshape(y_true, (-1,))
        y_pred = tf.keras.backend.reshape(y_pred, (-1, n_classes))
        return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.cast(y_true, 'int64'),
                                                            tf.keras.backend.argmax(y_pred, axis=-1)))

    input_image = tf.keras.layers.Input((None, None, 1))
    input_binary = tf.keras.layers.Input((None, None, 1))

    from pagesegmentation.lib.model import model_fcn_skip
    model = model_fcn_skip([input_image, input_binary], n_classes)
    optimizer = tf.keras.optimizers.Adam(lr=l_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
    keys = list(model_vars.keys())
    counter = 0
    for l in model.layers:
        if len(l.get_weights()) > 0:
            l.set_weights([model_vars[keys[counter]], model_vars[keys[counter + 1]]])
            counter += 2
            pass
    model.save(output_path + '/converted_model.hdf5')


if __name__ == "__main__":
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
    migrate_model(args.input_dir, args.n_classes, args.l_rate, args.output_path)
