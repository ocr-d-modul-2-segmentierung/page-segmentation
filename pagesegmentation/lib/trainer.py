from pagesegmentation.lib.dataset import DatasetLoader, Dataset
from pagesegmentation.lib.model import model_by_name
from typing import NamedTuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrainSettings(NamedTuple):
    n_iter: int
    n_classes: int
    l_rate: float
    train_data: Dataset
    validation_data: Dataset
    display: int
    output: str
    threads: int
    early_stopping_max_l_rate_drops: int
    data_augmentation: bool
    n_architecture: str = 'fcn_skip'
    evaluation_data: Dataset = None
    load: str = None
    continue_training: bool = False
    compute_baseline: bool = False
    foreground_masks: bool = False
    tensorboard: bool = False
    reduce_lr_on_plateu = True


class Trainer:
    def __init__(self, settings: TrainSettings):
        self.settings = settings
        print(settings.n_classes)
        from pagesegmentation.lib.network import Network
        self.train_net = Network("train", model_by_name(self.settings.n_architecture), settings.n_classes,
                                 l_rate=settings.l_rate,
                                 foreground_masks=settings.foreground_masks, model=settings.load,
                                 continue_training=settings.continue_training)

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

    def train(self) -> None:
        self.train_net.train_dataset(self.settings.train_data, self.settings.validation_data, self.settings.output,
                                     epochs=self.settings.n_iter, early_stopping=
                                     True if self.settings.early_stopping_max_l_rate_drops != 0 else False,
                                     early_stopping_interval=self.settings.early_stopping_max_l_rate_drops,
                                     tensorboardlogs=self.settings.tensorboard,
                                     augmentation=self.settings.data_augmentation, reduce_lr_on_plateu=
                                     self.settings.reduce_lr_on_plateu)

    def eval(self) -> None:
        if len(self.settings.evaluation_data) > 0:
            self.train_net.evaluate_dataset(self.settings.evaluation_data)
        else:
            print(self.train_net.evaluate_dataset(self.settings.validation_data))


if __name__ == "__main__":
    from pagesegmentation.lib.dataset import DatasetLoader
    from pagesegmentation.scripts.generate_image_map import load_image_map_from_file
    image_map = load_image_map_from_file('/home/alexander/Bilder/datenset2/image_map.json')
    dataset_loader = DatasetLoader(6, color_map=image_map)
    print(dataset_loader.color_map)
    train_data = dataset_loader.load_data_from_json(
        ['/home/alexander/Bilder/datenset2/t.json'], "train")
    test_data = dataset_loader.load_data_from_json(
        ['/home/alexander/Bilder/datenset2/t.json'], "test")
    eval_data = dataset_loader.load_data_from_json(
        ['/home/alexander/Bilder/datenset2/t.json'], "eval")
    settings = TrainSettings(
        n_iter=100,
        n_classes=len(dataset_loader.color_map),
        l_rate=1e-3,
        train_data=train_data,
        validation_data=test_data,
        display=10,
        output='/home/alexander/Bilder/datenset2/',#'/home/alexander/Bilder/test_datenset/', #'/home/alexander/PycharmProjects/PageContent/pagecontent/demo/'
        threads=8,
        foreground_masks=False,
        data_augmentation=True,
        tensorboard=True,
        n_architecture='mobile_net',
        early_stopping_max_l_rate_drops=5,
        load=None#'/home/alexander/Bilder/test_datenset/best_model.hdf5'
    )
    trainer = Trainer(settings)
    trainer.train()
    for x in test_data:
        trainer.train_net.predict_single_data(x)
