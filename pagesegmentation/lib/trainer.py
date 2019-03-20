from .dataset import Dataset
from typing import NamedTuple
from tqdm import tqdm
from pagesegmentation.lib.data_augmenter import DataAugmenterBase
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrainProgressCallback:
    def __init__(self):
        super().__init__()
        self.total_iters = 0
        self.early_stopping_iters = 0

    def init(self, total_iters, early_stopping_iters):
        self.total_iters = total_iters
        self.early_stopping_iters = early_stopping_iters

    def next_iteration(self, iter: int, loss: float, acc: float, fgpa: float):
        pass

    def next_best_model(self, best_iter: int, best_acc: float, best_iters: int):
        pass

    def early_stopping(self):
        pass


class TrainSettings(NamedTuple):
    n_iter: int
    n_classes: int
    l_rate: float
    train_data: Dataset
    validation_data: Dataset
    load: str
    display: int
    output: str
    early_stopping_test_interval: int
    early_stopping_max_keep: int
    early_stopping_on_accuracy: bool
    checkpoint_iter_delta: int
    threads: int
    data_augmentation: DataAugmenterBase = None
    compute_baseline: bool = False


class Trainer:
    def __init__(self, settings: TrainSettings):
        self.settings = settings

        from .network import Network
        from .model import model
        import tensorflow as tf

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph,
                                  config=tf.ConfigProto(
                                      intra_op_parallelism_threads=settings.threads,
                                      inter_op_parallelism_threads=settings.threads,
                                  ))

        self.train_net = Network("train", self.graph, self.session, model, settings.n_classes, l_rate=settings.l_rate, reuse=False, data_augmentation=settings.data_augmentation)
        self.test_net = Network("test", self.graph, self.session, model, settings.n_classes, l_rate=settings.l_rate, reuse=True)

        self.deploy_graph = tf.Graph()
        self.deploy_session = tf.Session(graph=self.deploy_graph)
        self.deploy_net = Network("deploy", self.deploy_graph, self.deploy_session, model, settings.n_classes, l_rate=settings.l_rate)

        self.train_net.set_data(settings.train_data)
        self.test_net.set_data(settings.validation_data)

        if len(settings.train_data) == 0 and settings.n_iter > 0:
            raise Exception("No training files specified. Maybe set n_iter=0")

        if settings.compute_baseline:
            def compute_label_percentage(label):
                return np.sum([np.sum(d.mask == label) for d in settings.train_data]) \
                       / np.sum([d.mask.shape[0] * d.mask.shape[1] for d in settings.train_data])

            logging.info("Computing label percentage for {} files.".format(len(settings.train_data)))
            label_percentage = [compute_label_percentage(l) for l in range(settings.n_classes)]
            logging.info("Label percentage: {}".format(list(zip(range(settings.n_classes), label_percentage))))
            logging.info("Baseline: {}".format(max(label_percentage)))

    def train(self, callback: TrainProgressCallback = None):
        settings = self.settings
        callback = callback if callback is not None else TrainProgressCallback()
        callback.init(settings.n_iter, settings.early_stopping_max_keep)

        def compute_pgpa(net, fg_not_a=not settings.early_stopping_on_accuracy):
            total_a, total_fg = 0, 0
            for logits, a, fg, _ in tqdm(net.test_dataset(), total=net.n_data()):
                total_a += a / net.n_data()
                total_fg += fg / net.n_data()

            return total_fg if fg_not_a else total_a

        self.train_net.prepare()
        if settings.load:
            self.train_net.load_weights(settings.load)

        current_best_fgpa = 0
        current_best_model_iter = 0
        current_best_iters = 0
        avg_loss, avg_acc, avg_fgpa = 10, 0, 0

        cur_checkpoint = 0
        for step in range(settings.n_iter):
            if not (settings.checkpoint_iter_delta is None):
                cur_checkpoint = cur_checkpoint if step < cur_checkpoint else cur_checkpoint + settings.checkpoint_iter_delta
            else:
                cur_checkpoint = None
            l, a, fg = self.train_net.train_dataset()

            # m = max([np.abs(np.mean(g)) for g, _ in gs])
            # print(m)
            avg_loss = 0.99 * avg_loss + 0.01 * l
            avg_acc = 0.99 * avg_acc + 0.01 * a
            avg_fgpa = 0.99 * avg_fgpa + 0.01 * fg

            callback.next_iteration(step, avg_loss, avg_acc, avg_fgpa)
            if step % settings.display == 0:
                print("#%05d (%.5f): Acc=%.5f FgPA=%.5f" % (step, avg_loss, avg_acc, avg_fgpa))

            if (step + 1) % settings.early_stopping_test_interval == 0:
                print("checking for early stopping")
                test_fgpa = compute_pgpa(self.test_net)
                if test_fgpa > current_best_fgpa:
                    current_best_fgpa = test_fgpa
                    current_best_model_iter = step + 1
                    current_best_iters = 0
                    print("New best model at iter {} with FgPA={}".format(current_best_model_iter, current_best_fgpa))

                    print("Saving the model to {}".format(settings.output))
                    self.train_net.save_checkpoint(settings.output)
                    self.deploy_net.load_weights(settings.output, restore_only_trainable=True)
                    self.deploy_net.save_checkpoint(settings.output, checkpoint=cur_checkpoint)
                else:
                    current_best_iters += 1
                    print("No new best model found. Current iterations {} with FgPA={}".format(current_best_iters,
                                                                                               current_best_fgpa))

                callback.next_best_model(current_best_iters, current_best_fgpa, current_best_iters)

                if current_best_iters >= settings.early_stopping_max_keep:
                    callback.early_stopping()
                    print('early stopping at %d' % (step + 1))
                    break

        self.train_net.prepare(False)  # reset the net, required to prevent blocking of tensorflow on shutdown
        self.test_net.prepare(False)   # possibly the garbage collector cant resolve the tf.Dataset

        print("Best model at iter {} with fgpa of {}".format(current_best_model_iter, current_best_fgpa))
        if current_best_iters > 0:
            # restore best model
            self.test_net.load_weights(settings.output, restore_only_trainable=True)
